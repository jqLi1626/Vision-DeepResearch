from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any, List, Optional

from PIL import Image
import re
from collections import Counter
import json5

# rLLM imports
from rllm.engine.rollout import RolloutEngine

# Constants from original DeepResearch
OBS_START = "<tool_response>"
OBS_END = "\n</tool_response>"
MAX_LLM_CALL_PER_RUN = 50
MAX_PROMPT_LENGTH_PER_RUN = 64000
MAX_RESPONSE_LENGTH_PER_RUN = 4096

DEEPRESEARCH_SYSTEM_PROMPT_TEXT = """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "search", "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "visit", "description": "Visit webpage(s) and return the summary of the content.", "parameters": {"type": "object", "properties": {"url": {"type": "array", "items": {"type": "string"}, "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."}, "goal": {"type": "string", "description": "The specific information goal for visiting webpage(s)."}}, "required": ["url", "goal"]}}}
</tools>

# Response Format

At each step you must output EXACTLY ONE of the following two patterns:

Pattern 1 — use a tool:
<think>
your reasoning
</think>
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Pattern 2 — give the final answer:
<think>
your reasoning
</think>
<answer>
your final answer
</answer>

Rules:
- Always begin with <think>...</think>.
- Output EXACTLY ONE of: a single <tool_call>...</tool_call>  OR  a single <answer>...</answer>.
- NEVER output both <tool_call> and <answer> in the same response — doing so will be rejected as a format error.
- NEVER output more than one <tool_call> in the same response — this will also be rejected.
- The JSON inside <tool_call> must be strictly valid: no trailing braces, no extra characters after the final `}`.

Current date: """


DEEPRESEARCH_SYSTEM_PROMPT = """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"name": "search", "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}}, "required": ["query"]}}
{"name": "crop_and_search", "description": "Crop some important local regions from an image and perform reverse image / visual search to identify objects, text, organizations, or other visual elements.", "parameters": {"type": "object", "properties": {"image_id": {"type": "string", "description": "The path or unique identifier of the image to analyze."}, "bbox": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "description": "Bounding box coordinates [x1, y1, x2, y2]."}, "minItems": 1, "description": "One or more important local regions to be cropped from the image."}, "goal": {"type": "string", "description": "The specific purpose of the visual search."}}, "required": ["image_id", "bbox", "goal"]}}
{"name": "visit", "description": "Visit webpage(s) and return the summary of the content.", "parameters": {"type": "object", "properties": {"url": {"type": "array", "items": {"type": "string"}, "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."}, "goal": {"type": "string", "description": "The specific information goal for visiting webpage(s)."}}, "required": ["url", "goal"]}}
</tools>

# Response Format

At each step you must output EXACTLY ONE of the following two patterns:

Pattern 1 — use a tool:
<think>
your reasoning
</think>
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Pattern 2 — give the final answer:
<think>
your reasoning
</think>
<answer>
your final answer
</answer>

Rules:
- Always begin with <think>...</think>.
- Output EXACTLY ONE of: a single <tool_call>...</tool_call>  OR  a single <answer>...</answer>.
- NEVER output both <tool_call> and <answer> in the same response.
- NEVER output more than one <tool_call> in the same response.

Current date: """


def _extract_first_json_object(text: str) -> str:
    """
    Extract the first complete, balanced JSON object from text.
    Uses brace-depth tracking (correctly handling string literals and escape
    sequences) so that any trailing characters — including extra `}}}` or `/>` —
    are silently discarded.
    Returns the extracted JSON string, or the original text if no object found.
    """
    depth = 0
    in_string = False
    escape_next = False
    start = None

    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : i + 1]

    return text  # fallback: return original if no balanced object found


def today_date():
    """Get today's date in YYYY-MM-DD format."""
    return datetime.now().date().strftime("%Y-%m-%d")


def analyze_repetition_ngram(text: str, n: int = 30, threshold: float = 0.5):
    """
    Use N-grams to detect repetition in a string.

    Args:
        text (str): Input text to analyze.
        n (int): N-gram window size (default 10).
            - For long repetitive sequences, 10-20 is recommended.
        threshold (float): Distinct-N threshold (0~1).
            - Values below this indicate heavy repetition (default 0.5).

    Returns:
        bool: True if repetition is detected, False otherwise.
    """
    if not text or len(text) < n:
        print("text is too short, cannot analyze.")
        return False

    # 1. Generate N-grams (character-level sliding window).
    # List comprehension: slice from index i to i+n.
    ngrams = [text[i : i + n] for i in range(len(text) - n + 1)]

    total_count = len(ngrams)
    if total_count == 0:
        return False

    # 2. Count frequencies.
    ngram_counts = Counter(ngrams)
    unique_count = len(ngram_counts)

    # 3. Compute Distinct-N (unique count / total count).
    # Repetitive text is typically < 0.4.
    distinct_ratio = unique_count / total_count

    # 4. Determine repetition.
    is_repetitive = distinct_ratio < threshold

    return is_repetitive


def count_words(text: str) -> int:
    # Match segments that look like English words.
    # Rule: starts and ends with a letter, may contain letters, apostrophes, or hyphens.
    pattern = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")
    words = pattern.findall(text)
    return len(words)


def build_text_completion_prompt(
    messages: list[dict], allow_special: bool = True
) -> str:
    """
    Build text completion prompt from messages list.
    Adapted from qwen_agent.utils.utils.build_text_completion_prompt

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        allow_special: Whether to allow special tokens (for compatibility)

    Returns:
        Formatted prompt string
    """
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"

    prompt_parts = []

    # Handle system message
    if messages and messages[0]["role"] == "system":
        sys_content = messages[0]["content"]
        prompt_parts.append(f"{im_start}system\n{sys_content}{im_end}")
        messages = messages[1:]

    # Ensure chat completes with assistant
    if messages and messages[-1]["role"] != "assistant":
        messages = messages + [{"role": "assistant", "content": ""}]

    # Format each message
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt_parts.append(f"{im_start}{role}\n{content}{im_end}")

    return "\n".join(prompt_parts)


class MultiTurnReactAgent:
    """
    Multi-turn ReAct Agent adapted from Tongyi DeepResearch.

    This agent implements the core reasoning loop with tool calling capabilities,
    using rLLM's OpenAI engine for model inference.
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        tools: dict = None,
        system_prompt: str | None = None,
        default_max_tries: int = 3,
        **kwargs,
    ):
        """
        Initialize the ReAct agent.

        Args:
            rollout_engine: rLLM OpenAI engine for model inference
            tools: Dictionary of available tools {tool_name: tool_instance}
            system_prompt: Optional custom system prompt
        """
        self.rollout_engine = rollout_engine
        self.tools = tools or {}
        self.system_prompt = system_prompt
        # Configuration from original DeepResearch
        self.max_llm_calls = MAX_LLM_CALL_PER_RUN
        self.default_max_tries = default_max_tries

        # Smart context management using actual API consumption
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        self.max_prompt_tokens = MAX_PROMPT_LENGTH_PER_RUN
        self.max_response_tokens = MAX_RESPONSE_LENGTH_PER_RUN

    def sanity_check_output(self, content: str) -> bool:
        """Check if the model output contains the expected thinking structure."""
        return "<think>" in content and "</think>" in content

    async def call_server(
        self, messages: list[dict], max_tries: Optional[int] = None, **kwargs
    ):
        """Call rollout engine once; assumes XML ReAct format."""
        try:
            # Force per-round limits from DeepResearchAgent without local token estimation.
            if hasattr(self.rollout_engine, "max_prompt_length"):
                self.rollout_engine.max_prompt_length = int(self.max_prompt_tokens)
            if hasattr(self.rollout_engine, "max_response_length"):
                self.rollout_engine.max_response_length = int(self.max_response_tokens)

            kwargs.pop("max_new_tokens", None)
            kwargs["max_tokens"] = int(self.max_response_tokens)
            response = await self.rollout_engine.get_model_response(
                messages=messages, **kwargs
            )

            if hasattr(response, "prompt_length") and hasattr(
                response, "completion_length"
            ):
                self.total_prompt_tokens += response.prompt_length
                self.total_completion_tokens += response.completion_length

            return response
        except Exception as exc:  # noqa: BLE001
            print(f"call_server failed: {exc}")
            raise

    def record_token_usage(self, response) -> None:
        """Record the latest prompt/completion token count from rollout engine."""
        prompt_tokens = getattr(response, "prompt_length", None)
        completion_tokens = getattr(response, "completion_length", None)

        if prompt_tokens is not None:
            try:
                self.total_prompt_tokens = int(prompt_tokens)
            except (TypeError, ValueError):  # noqa: PERF203
                self.total_prompt_tokens = 0

        if completion_tokens is not None:
            try:
                self.total_completion_tokens = int(completion_tokens)
            except (TypeError, ValueError):  # noqa: PERF203
                self.total_completion_tokens = 0

    def get_total_tokens_used(self) -> int:
        """Return the latest prompt + completion token usage reported by the engine."""
        return self.total_prompt_tokens + self.total_completion_tokens

    def _estimate_prompt_tokens(self, messages: list[dict]) -> int:
        """Estimate prompt length for the next call using the rollout engine's tokenizer."""
        tokenizer = getattr(self.rollout_engine, "tokenizer", None)
        chat_parser = getattr(self.rollout_engine, "chat_parser", None)

        if tokenizer is None or chat_parser is None:
            return self.total_prompt_tokens

        try:
            prompt = chat_parser.parse(
                messages,
                add_generation_prompt=True,
                is_first_msg=True,
                tools=[],
                accumulate_reasoning=getattr(
                    self.rollout_engine, "accumulate_reasoning", False
                ),
            )
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            return len(token_ids)
        except Exception as exc:  # noqa: BLE001
            print(f"[TokenEstimator] Failed to estimate prompt tokens: {exc}")
            return self.total_prompt_tokens

    def _build_result(
        self,
        *,
        question: str,
        answer: str | None,
        messages: list[dict],
        prediction: str,
        termination: str,
        rounds: int,
        start_time: float,
        # next_prompt_tokens: int | None = None,
    ) -> dict:
        """Assemble result payload with token usage metadata."""
        token_usage = {
            "prompt": self.total_prompt_tokens,
            "completion": self.total_completion_tokens,
            "max_prompt": self.max_prompt_tokens,
        }

        result = {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination,
            "rounds": rounds,
            "time_taken": time.time() - start_time,
            "token_usage": token_usage,
        }
        return result

    async def _run(
        self,
        question: str,
        answer: str = None,
        images: list = None,
        image_path: str = None,
        **kwargs,
    ) -> dict:
        """
        Main reasoning loop adapted from original DeepResearch.

        This is the core ReAct implementation that handles:
        - Multi-turn conversation
        - Tool calling and execution
        - Context length management
        - Termination conditions

        Args:
            question: The research question to answer
            answer: Ground truth answer (for evaluation)
            images: List of image data URLs (base64 encoded)

        Returns:
            Dictionary with results including messages, prediction, and termination reason
        """
        start_time = time.time()

        system_prompt = (
            self.system_prompt or DEEPRESEARCH_SYSTEM_PROMPT
        ) + today_date()

        if images:
            user_message = {
                "role": "user",
                "content": question,
                "images": images,
            }
        else:
            user_message = {"role": "user", "content": question}

        messages = [
            {"role": "system", "content": system_prompt},
            user_message,
        ]

        if not images:
            messages = [
                {
                    "role": "system",
                    "content": DEEPRESEARCH_SYSTEM_PROMPT_TEXT + today_date(),
                },
                user_message,
            ]

        num_llm_calls_available = self.max_llm_calls
        round = 0
        termination = None
        prediction = ""
        consecutive_bad_steps = 0

        while num_llm_calls_available > 0:
            round += 1
            num_llm_calls_available -= 1

            # Get model response from rollout engine
            try:
                response = await self.call_server(messages, **kwargs)
            except Exception as exc:  # noqa: BLE001
                prediction = "call_server failed"
                termination = "error"
                return self._build_result(
                    question=question,
                    answer=answer,
                    messages=messages,
                    prediction=prediction,
                    termination=termination,
                    rounds=round,
                    start_time=start_time,
                )

            # Synchronize token usage with rollout engine feedback
            self.record_token_usage(response)

            # Extract text content (may be None for pure function calling)
            content = (
                response.text if hasattr(response, "text") and response.text else ""
            )

            if "<tool_call>" in content:
                # Extract tool name for display
                if "python" in content.lower() and "<code>" in content:
                    pass
                elif '"name":' in content:
                    try:
                        tool_text = content.split("<tool_call>")[1].split(
                            "</tool_call>"
                        )[0]
                        tool_data = json5.loads(tool_text)
                        tool_name = tool_data.get("name", "Unknown")
                        if "arguments" in tool_data:
                            args_str = str(tool_data["arguments"])
                            pass
                        else:
                            pass
                    except Exception:
                        pass
                else:
                    pass

            # Clean up content if it contains tool_response
            if "<tool_response>" in content:
                pos = content.find("<tool_response>")
                content = content[:pos]

            # Format violation: both <tool_call> and <answer> in the same response
            has_tool_call = "<tool_call>" in content and "</tool_call>" in content
            has_answer = "<answer>" in content and "</answer>" in content
            multiple_tool_calls = content.count("<tool_call>") > 1

            if (has_tool_call and has_answer) or multiple_tool_calls:
                if multiple_tool_calls:
                    observation = (
                        "Format error: your response contains more than one <tool_call>. "
                        "Output EXACTLY ONE <tool_call> per response. Please try again."
                    )
                else:
                    observation = (
                        "Format error: your response contains both <tool_call> and <answer>. "
                        "Output EXACTLY ONE — either a <tool_call> OR an <answer>, never both. "
                        "Please try again."
                    )
                messages.append(
                    {"role": "assistant", "content": content.strip(), "step_error": True}
                )
                messages.append({"role": "user", "content": observation})
                consecutive_bad_steps += 1
                if consecutive_bad_steps >= 3:
                    prediction = "Too many consecutive step errors."
                    termination = "consecutive_step_errors"
                    return self._build_result(
                        question=question,
                        answer=answer,
                        messages=messages,
                        prediction=prediction,
                        termination=termination,
                        rounds=round,
                        start_time=start_time,
                    )
                continue

            # Only XML ReAct tool calls are supported.
            if "<tool_call>" in content and "</tool_call>" in content:
                # ReAct text format path
                assistant_message = {
                    "role": "assistant",
                    "content": content.strip(),
                    "step_error": False,
                }
                messages.append(assistant_message)
                tool_error = False

                tool_call_text = content.split("<tool_call>")[1].split("</tool_call>")[
                    0
                ]
                # Special handling for Python code (match original logic)
                if "python" in tool_call_text.lower():
                    try:
                        # Extract code from the original content (not just tool_call_text)
                        code_raw = (
                            content.split("<tool_call>")[1]
                            .split("</tool_call>")[0]
                            .split("<code>")[1]
                            .split("</code>")[0]
                            .strip()
                        )
                        result = await self.execute_python(code_raw)
                        if isinstance(result, str) and result.startswith(
                            (
                                "Python execution error:",
                                "PythonInterpreter tool not available",
                                "PythonInterpreter tool is not callable",
                            )
                        ):
                            tool_error = True
                    except Exception:
                        result = (
                            "[Python Interpreter Error]: Python code formatting error."
                        )
                        tool_error = True
                else:
                    try:
                        # Strip any trailing garbage (e.g. extra }}} or />) before parsing
                        clean_json = _extract_first_json_object(tool_call_text)
                        tool_call = json5.loads(clean_json)
                        tool_name = tool_call.get("name", "")
                        tool_args = tool_call.get("arguments", {})
                        if tool_name == "crop_and_search":
                            tool_args["image_id"] = image_path
                        result = await self.custom_call_tool(tool_name, tool_args)
                    except Exception:
                        result = "[Json Parse Error]: Tool call is not a valid JSON."
                        tool_error = True

                if tool_error:
                    assistant_message["step_error"] = True

                # Add tool response in ReAct format
                tool_response = f"<tool_response>\n{result}\n</tool_response>"
                messages.append({"role": "user", "content": tool_response})
                if assistant_message["step_error"]:
                    consecutive_bad_steps += 1
                else:
                    consecutive_bad_steps = 0
                if consecutive_bad_steps >= 3:
                    prediction = "Too many consecutive step errors."
                    termination = "consecutive_step_errors"
                    return self._build_result(
                        question=question,
                        answer=answer,
                        messages=messages,
                        prediction=prediction,
                        termination=termination,
                        rounds=round,
                        start_time=start_time,
                    )

            # Check for final answer AFTER processing tools
            # This allows o3 to execute tools even when it includes answer in same message
            elif "<answer>" in content and "</answer>" in content:
                messages.append(
                    {
                        "role": "assistant",
                        "content": content.strip(),
                        "step_error": False,
                    }
                )
                prediction = content.split("<answer>")[1].split("</answer>")[0].strip()
                termination = "answer"
                consecutive_bad_steps = 0
                break

            # Priority 3: No tool call and answer, just reasoning or format error
            else:
                is_repetitive = analyze_repetition_ngram(content)
                is_overlong = count_words(content) > 2500
                if is_repetitive and is_overlong:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": content.strip(),
                            "step_error": True,
                        }
                    )
                    prediction = "Repetition response"
                    termination = "repetition_detected"
                    return self._build_result(
                        question=question,
                        answer=answer,
                        messages=messages,
                        prediction=prediction,
                        termination=termination,
                        rounds=round,
                        start_time=start_time,
                    )

                observation = "Error: Invalid content format. Content must contain <tool_call> or <answer> tags. Let's try again."
                messages.append(
                    {
                        "role": "assistant",
                        "content": content.strip(),
                        "step_error": True,
                    }
                )
                messages.append({"role": "user", "content": observation})
                consecutive_bad_steps += 1
                if consecutive_bad_steps >= 3:
                    prediction = "Too many consecutive step errors."
                    termination = "consecutive_step_errors"
                    return self._build_result(
                        question=question,
                        answer=answer,
                        messages=messages,
                        prediction=prediction,
                        termination=termination,
                        rounds=round,
                        start_time=start_time,
                    )

            # Determine whether another round is feasible
            if num_llm_calls_available <= 0 and "<answer>" not in content:
                prediction = f"No answer found after {self.max_llm_calls} rounds."
                termination = f"answer not found after {self.max_llm_calls} rounds"
                return self._build_result(
                    question=question,
                    answer=answer,
                    messages=messages,
                    prediction=prediction,
                    termination=termination,
                    rounds=round,
                    start_time=start_time,
                )

        last_message_content = (
            messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        )
        if last_message_content and "<answer>" in last_message_content:
            prediction = last_message_content.split("<answer>")[1].split("</answer>")[0]
            termination = "answer"
        else:
            prediction = "No answer found."
            termination = "answer not found"
            if num_llm_calls_available == 0:
                termination = "exceed available llm calls"

        result = self._build_result(
            question=question,
            answer=answer,
            messages=messages,
            prediction=prediction,
            termination=termination,
            rounds=round,
            start_time=start_time,
        )

        print("\n🏁 DeepResearch completed:")
        print(f"   Rounds: {round}")
        print(f"   Time: {result['time_taken']:.1f}s")
        print(f"   Termination: {termination}")
        print(
            "   Token usage: prompt={prompt}, completion={completion}, max_prompt={max_prompt}".format(
                prompt=self.total_prompt_tokens,
                completion=self.total_completion_tokens,
                max_prompt=self.max_prompt_tokens,
            )
        )
        return result

    async def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs) -> str:
        """
        Execute tool calls with the available tools.

        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments to pass to the tool

        Returns:
            Tool execution result as string
        """
        if tool_name in self.tools:
            try:
                # Call the tool
                if hasattr(self.tools[tool_name], "call"):
                    # Async tool
                    if asyncio.iscoroutinefunction(self.tools[tool_name].call):
                        result = await self.tools[tool_name].call(**tool_args)
                    else:
                        result = self.tools[tool_name].call(**tool_args)
                elif callable(self.tools[tool_name]):
                    # Direct callable
                    result = self.tools[tool_name](**tool_args)
                else:
                    result = f"Tool {tool_name} is not callable"

                return str(result)

            except Exception as e:
                return f"Error calling tool {tool_name}: {e}"
        else:
            available_tools = list(self.tools.keys())
            return f"Tool {tool_name} not found. Available tools: {available_tools}"

    async def execute_python(self, code: str) -> str:
        """
        Execute Python code using the PythonInterpreter tool.

        Args:
            code: Python code to execute

        Returns:
            Execution result as string
        """
        if "PythonInterpreter" in self.tools:
            try:
                # Use the PythonInterpreter tool
                tool = self.tools["PythonInterpreter"]
                if hasattr(tool, "call"):
                    if asyncio.iscoroutinefunction(tool.call):
                        result = await tool.call(code=code)
                    else:
                        result = tool.call(code=code)
                    return str(result)
                else:
                    return "PythonInterpreter tool is not callable"
            except Exception as e:
                return f"Python execution error: {e}"
        else:
            return "PythonInterpreter tool not available"

    def reset(self):
        """Reset the agent state (for compatibility with rLLM workflow)."""
        # Reset token counters for each new task
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    async def run(
        self,
        question: str,
        answer: str = None,
        images: list = None,
        image_path: str = None,
        **kwargs,
    ) -> dict:
        """
        Public interface for running the agent.

        Args:
            question: Research question to answer
            answer: Ground truth answer (optional, for evaluation)

        Returns:
            Result dictionary
        """
        # Reset token counters for each new run
        self.reset()
        return await self._run(question, answer, images, image_path, **kwargs)


DeepResearchAgent = MultiTurnReactAgent


# =============================================================================
# Planner-Executor Architecture
# =============================================================================

PLANNER_SYSTEM_PROMPT = """You are a deep research planner. Your role is to think carefully about how to answer research questions by gathering information from multiple sources.

At each step, you will see the conversation history including any previous tool results. Your job is to REASON about:
1. What information you have gathered so far
2. What you still need to find out
3. What action to take next and WHY

# Tools Available
You can direct the executor to use any of the following tools:
- **search**: Perform web searches with specific queries
- **crop_and_search**: Crop a region from an image and perform reverse visual search to identify objects, text, or entities
- **visit**: Visit specific URLs to extract detailed information

# IMPORTANT: Output Format
You must output EXACTLY ONE <think>...</think> block and NOTHING ELSE.

In your thinking, be explicit about your intent. End your think block with ONE of these directive lines:

If you want to use a tool:
→ ACTION: <tool_name> | <description of what to do and why>
  Examples:
  → ACTION: search | Search for "Eiffel Tower height in meters" to find the exact height
  → ACTION: visit | Visit https://example.com to find the publication date of the article
  → ACTION: crop_and_search | Crop the logo in the top-left corner (approximately 0-200 width, 0-100 height) to identify the brand

If you are ready to answer:
→ ACTION: answer | <your complete final answer>

Rules:
- Always end your <think> block with an → ACTION line
- Be specific and descriptive in your action directive so the executor can work precisely
- For crop_and_search, describe the region in natural language (the executor will determine exact coordinates)
- For search, specify the exact query strings you want
- For answer, include your complete answer in the directive

Current date: """

EXECUTOR_SYSTEM_PROMPT = """You are a precise action executor for a research system. You will receive a planner's thinking and your job is to translate the planner's intent into an exact, well-formatted action.

# Tools

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_definition}
</tools>

# Response Format

Based on the planner's thinking and its → ACTION directive, output EXACTLY ONE of the following:

Pattern 1 — use a tool:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>

Pattern 2 — give the final answer:
<answer>
your final answer here
</answer>

Rules:
- Output EXACTLY ONE tool_call OR answer, never both, never neither
- The JSON inside <tool_call> must be strictly valid
- For crop_and_search bbox: use coordinates in 0-1000 range (where 1000 = full image width/height). Provide precise coordinates based on the planner's description
- For search: extract specific query strings from the planner's directive
- For answer: copy the answer content from the planner's → ACTION: answer directive
"""

EXECUTOR_TOOLS_DEFINITION = """{"name": "search", "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}}, "required": ["query"]}}
{"name": "crop_and_search", "description": "Crop some important local regions from an image and perform reverse image / visual search to identify objects, text, organizations, or other visual elements.", "parameters": {"type": "object", "properties": {"image_id": {"type": "string", "description": "The path or unique identifier of the image to analyze."}, "bbox": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "description": "Bounding box coordinates [x1, y1, x2, y2] in 0-1000 range."}, "minItems": 1, "description": "One or more important local regions to be cropped from the image."}, "goal": {"type": "string", "description": "The specific purpose of the visual search."}}, "required": ["image_id", "bbox", "goal"]}}
{"name": "visit", "description": "Visit webpage(s) and return the summary of the content.", "parameters": {"type": "object", "properties": {"url": {"type": "array", "items": {"type": "string"}, "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."}, "goal": {"type": "string", "description": "The specific information goal for visiting webpage(s)."}}, "required": ["url", "goal"]}}"""


class PlannerExecutorReActAgent(MultiTurnReactAgent):
    """
    Two-model ReAct Agent using a Planner-Executor architecture.

    Motivation:
    - Large models (planners) excel at reasoning but may produce inaccurate bbox
      coordinates and non-standard tool call formats.
    - Small trained models (executors) can be fine-tuned to generate precise,
      well-formatted tool calls given a high-level intent description.

    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  PLANNER (large model)                                      │
    │  Input:  conversation history + question + tool responses   │
    │  Output: <think>reasoning + → ACTION directive</think>      │
    └────────────────────────┬────────────────────────────────────┘
                             │ thinking content
                             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  EXECUTOR (small trained model)                             │
    │  Input:  planner's thinking + image (if vision task)        │
    │  Output: <tool_call>...</tool_call> OR <answer>...</answer>  │
    └────────────────────────┬────────────────────────────────────┘
                             │ tool call or answer
                             ▼
                       Tool Execution
                             │ tool_response
                             ▼
                    Back to PLANNER loop
    """

    def __init__(
        self,
        planner_engine: RolloutEngine,
        executor_engine: RolloutEngine,
        tools: dict = None,
        planner_system_prompt: str | None = None,
        executor_system_prompt: str | None = None,
        executor_tools_definition: str | None = None,
        default_max_tries: int = 3,
        **kwargs,
    ):
        """
        Initialize the Planner-Executor agent.

        Args:
            planner_engine: RolloutEngine for the large planner model
            executor_engine: RolloutEngine for the small executor model
            tools: Dictionary of available tools {tool_name: tool_instance}
            planner_system_prompt: Custom system prompt for the planner
            executor_system_prompt: Custom system prompt template for the executor
            executor_tools_definition: Tool definitions string injected into executor prompt
        """
        # Initialize parent with planner as the primary engine
        super().__init__(
            planner_engine, tools, planner_system_prompt, default_max_tries, **kwargs
        )
        self.executor_engine = executor_engine
        self.planner_system_prompt = planner_system_prompt or PLANNER_SYSTEM_PROMPT

        _tools_def = executor_tools_definition or EXECUTOR_TOOLS_DEFINITION
        _exec_prompt_template = executor_system_prompt or EXECUTOR_SYSTEM_PROMPT
        self.executor_system_prompt = _exec_prompt_template.format(
            tools_definition=_tools_def
        )

    # ------------------------------------------------------------------
    # Planner call
    # ------------------------------------------------------------------

    async def call_planner(
        self, messages: list[dict], **kwargs
    ) -> str:
        """
        Call the large planner model to get its thinking.

        The planner receives the full conversation history and outputs ONLY
        a <think>...</think> block with its reasoning and an → ACTION directive.

        Returns:
            The full thinking text (including <think> tags)
        """
        kwargs.pop("max_new_tokens", None)
        kwargs["max_tokens"] = int(self.max_response_tokens)

        # Use stop tokens to prevent the planner from generating tool_calls
        stop_tokens = kwargs.pop("stop", [])
        if isinstance(stop_tokens, str):
            stop_tokens = [stop_tokens]
        stop_tokens = list(stop_tokens) + ["</think>"]
        kwargs["stop"] = stop_tokens

        try:
            response = await self.rollout_engine.get_model_response(
                messages=messages, **kwargs
            )
            self.record_token_usage(response)
            content = (
                response.text if hasattr(response, "text") and response.text else ""
            )
            # Ensure the <think> block is properly closed
            if "<think>" in content and "</think>" not in content:
                content = content + "</think>"
            return content
        except Exception as exc:
            print(f"call_planner failed: {exc}")
            raise

    # ------------------------------------------------------------------
    # Executor call
    # ------------------------------------------------------------------

    async def call_executor(
        self,
        thinking: str,
        images: list | None = None,
        image_path: str | None = None,
        **kwargs,
    ) -> str:
        """
        Call the small executor model to translate planner's thinking into action.

        The executor receives:
        - The planner's full thinking (including the → ACTION directive)
        - The original image (if available, for accurate bbox generation)

        It outputs EXACTLY ONE of:
        - <tool_call>...</tool_call>
        - <answer>...</answer>

        Args:
            thinking: The planner's <think>...</think> block
            images: List of image data URLs (base64) for vision tasks
            image_path: Path to the image file
        """
        user_content = (
            "Based on the following research planner's thinking, generate the "
            "precise tool call or final answer:\n\n"
            f"{thinking}"
        )

        if images:
            user_message = {
                "role": "user",
                "content": user_content,
                "images": images,
            }
        else:
            user_message = {"role": "user", "content": user_content}

        executor_messages = [
            {"role": "system", "content": self.executor_system_prompt},
            user_message,
        ]

        kwargs.pop("max_new_tokens", None)
        kwargs.pop("stop", None)  # Executor should not inherit planner stop tokens
        kwargs["max_tokens"] = int(self.max_response_tokens)

        try:
            response = await self.executor_engine.get_model_response(
                messages=executor_messages, **kwargs
            )
            content = (
                response.text if hasattr(response, "text") and response.text else ""
            )
            return content
        except Exception as exc:
            print(f"call_executor failed: {exc}")
            raise

    # ------------------------------------------------------------------
    # Main reasoning loop (overrides parent _run)
    # ------------------------------------------------------------------

    async def _run(
        self,
        question: str,
        answer: str = None,
        images: list = None,
        image_path: str = None,
        **kwargs,
    ) -> dict:
        """
        Main reasoning loop for the Planner-Executor architecture.

        Flow per round:
        1. Planner receives conversation history → outputs <think>reasoning + ACTION</think>
        2. Executor receives thinking + image → outputs <tool_call> or <answer>
        3. Tool is executed → result added to conversation as <tool_response>
        4. Loop back to step 1

        The conversation history stored for the planner combines the planner's
        thinking WITH the executor's tool_call in a single assistant message,
        so the planner can see both what it intended and what was actually done.
        """
        start_time = time.time()

        planner_system = self.planner_system_prompt + today_date()

        if images:
            user_message = {
                "role": "user",
                "content": question,
                "images": images,
            }
        else:
            user_message = {"role": "user", "content": question}

        messages = [
            {"role": "system", "content": planner_system},
            user_message,
        ]

        num_llm_calls_available = self.max_llm_calls
        round = 0
        termination = None
        prediction = ""
        consecutive_bad_steps = 0

        while num_llm_calls_available > 0:
            round += 1
            num_llm_calls_available -= 1

            # ── Step 1: Planner generates thinking ──────────────────────────
            try:
                thinking = await self.call_planner(messages, **kwargs)
            except Exception as exc:
                prediction = "call_planner failed"
                termination = "error"
                return self._build_result(
                    question=question,
                    answer=answer,
                    messages=messages,
                    prediction=prediction,
                    termination=termination,
                    rounds=round,
                    start_time=start_time,
                )

            if not thinking or "<think>" not in thinking:
                # Planner produced no valid thinking
                observation = (
                    "Error: Your response must start with <think>...</think>. "
                    "Please output your reasoning inside <think> tags."
                )
                messages.append(
                    {"role": "assistant", "content": thinking.strip(), "step_error": True}
                )
                messages.append({"role": "user", "content": observation})
                consecutive_bad_steps += 1
                if consecutive_bad_steps >= 3:
                    prediction = "Too many consecutive step errors (planner)."
                    termination = "consecutive_step_errors"
                    return self._build_result(
                        question=question,
                        answer=answer,
                        messages=messages,
                        prediction=prediction,
                        termination=termination,
                        rounds=round,
                        start_time=start_time,
                    )
                continue

            # Check for early termination signals in thinking
            if analyze_repetition_ngram(thinking) and count_words(thinking) > 2500:
                messages.append(
                    {"role": "assistant", "content": thinking.strip(), "step_error": True}
                )
                prediction = "Repetition response in planner"
                termination = "repetition_detected"
                return self._build_result(
                    question=question,
                    answer=answer,
                    messages=messages,
                    prediction=prediction,
                    termination=termination,
                    rounds=round,
                    start_time=start_time,
                )

            # ── Step 2: Executor translates thinking into action ─────────────
            try:
                executor_output = await self.call_executor(
                    thinking=thinking,
                    images=images,
                    image_path=image_path,
                    **kwargs,
                )
            except Exception as exc:
                prediction = "call_executor failed"
                termination = "error"
                return self._build_result(
                    question=question,
                    answer=answer,
                    messages=messages,
                    prediction=prediction,
                    termination=termination,
                    rounds=round,
                    start_time=start_time,
                )

            # Clean up executor output if it contains tool_response
            if "<tool_response>" in executor_output:
                pos = executor_output.find("<tool_response>")
                executor_output = executor_output[:pos]

            # Validate executor output format
            has_tool_call = (
                "<tool_call>" in executor_output and "</tool_call>" in executor_output
            )
            has_answer = (
                "<answer>" in executor_output and "</answer>" in executor_output
            )
            multiple_tool_calls = executor_output.count("<tool_call>") > 1

            # Combine planner thinking + executor action into a single assistant message
            combined_content = thinking.strip() + "\n" + executor_output.strip()

            if (has_tool_call and has_answer) or multiple_tool_calls:
                # Executor format error
                if multiple_tool_calls:
                    observation = (
                        "Format error: executor output contains more than one <tool_call>. "
                        "Output EXACTLY ONE <tool_call> per response."
                    )
                else:
                    observation = (
                        "Format error: executor output contains both <tool_call> and <answer>. "
                        "Output EXACTLY ONE — either <tool_call> OR <answer>, never both."
                    )
                messages.append(
                    {
                        "role": "assistant",
                        "content": combined_content,
                        "step_error": True,
                    }
                )
                messages.append({"role": "user", "content": observation})
                consecutive_bad_steps += 1
                if consecutive_bad_steps >= 3:
                    prediction = "Too many consecutive step errors (executor)."
                    termination = "consecutive_step_errors"
                    return self._build_result(
                        question=question,
                        answer=answer,
                        messages=messages,
                        prediction=prediction,
                        termination=termination,
                        rounds=round,
                        start_time=start_time,
                    )
                continue

            # ── Step 3a: Execute tool call ───────────────────────────────────
            if has_tool_call:
                assistant_message = {
                    "role": "assistant",
                    "content": combined_content,
                    "step_error": False,
                }
                messages.append(assistant_message)
                tool_error = False

                tool_call_text = executor_output.split("<tool_call>")[1].split(
                    "</tool_call>"
                )[0]

                try:
                    clean_json = _extract_first_json_object(tool_call_text)
                    tool_call = json5.loads(clean_json)
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("arguments", {})
                    if tool_name == "crop_and_search":
                        tool_args["image_id"] = image_path
                    result = await self.custom_call_tool(tool_name, tool_args)
                except Exception:
                    result = "[Json Parse Error]: Executor tool call is not valid JSON."
                    tool_error = True

                if tool_error:
                    assistant_message["step_error"] = True

                tool_response = f"<tool_response>\n{result}\n</tool_response>"
                messages.append({"role": "user", "content": tool_response})

                if assistant_message["step_error"]:
                    consecutive_bad_steps += 1
                else:
                    consecutive_bad_steps = 0

                if consecutive_bad_steps >= 3:
                    prediction = "Too many consecutive step errors."
                    termination = "consecutive_step_errors"
                    return self._build_result(
                        question=question,
                        answer=answer,
                        messages=messages,
                        prediction=prediction,
                        termination=termination,
                        rounds=round,
                        start_time=start_time,
                    )

            # ── Step 3b: Final answer ────────────────────────────────────────
            elif has_answer:
                messages.append(
                    {
                        "role": "assistant",
                        "content": combined_content,
                        "step_error": False,
                    }
                )
                prediction = (
                    executor_output.split("<answer>")[1].split("</answer>")[0].strip()
                )
                termination = "answer"
                consecutive_bad_steps = 0
                break

            # ── Step 3c: Executor produced no valid output ───────────────────
            else:
                observation = (
                    "Error: Invalid executor output. Must contain <tool_call> or "
                    "<answer> tags. Please try again."
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": combined_content,
                        "step_error": True,
                    }
                )
                messages.append({"role": "user", "content": observation})
                consecutive_bad_steps += 1
                if consecutive_bad_steps >= 3:
                    prediction = "Too many consecutive step errors."
                    termination = "consecutive_step_errors"
                    return self._build_result(
                        question=question,
                        answer=answer,
                        messages=messages,
                        prediction=prediction,
                        termination=termination,
                        rounds=round,
                        start_time=start_time,
                    )

            # Check if we've run out of rounds
            if num_llm_calls_available <= 0 and "<answer>" not in executor_output:
                prediction = f"No answer found after {self.max_llm_calls} rounds."
                termination = f"answer not found after {self.max_llm_calls} rounds"
                return self._build_result(
                    question=question,
                    answer=answer,
                    messages=messages,
                    prediction=prediction,
                    termination=termination,
                    rounds=round,
                    start_time=start_time,
                )

        last_message_content = (
            messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        )
        if last_message_content and "<answer>" in last_message_content:
            prediction = (
                last_message_content.split("<answer>")[1].split("</answer>")[0]
            )
            termination = "answer"
        else:
            prediction = "No answer found."
            termination = "answer not found"
            if num_llm_calls_available == 0:
                termination = "exceed available llm calls"

        result = self._build_result(
            question=question,
            answer=answer,
            messages=messages,
            prediction=prediction,
            termination=termination,
            rounds=round,
            start_time=start_time,
        )

        print("\n🏁 PlannerExecutor DeepResearch completed:")
        print(f"   Rounds: {round}")
        print(f"   Time: {result['time_taken']:.1f}s")
        print(f"   Termination: {termination}")
        print(
            "   Token usage: prompt={prompt}, completion={completion}".format(
                prompt=self.total_prompt_tokens,
                completion=self.total_completion_tokens,
            )
        )
        return result