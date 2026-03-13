"""
Benchmark evaluation runner for Vision DeepResearch (single workflow).

Goals:
- Use the same workflow/tools/reward as training (`DeepResearchWorkflow`, `deepresearch_reward_fn`).
- Default rollout: OpenAI-compatible (can point to local vLLM server via base_url).
- Input: Parquet only with columns question/answer/(images).
- Save full trajectories (sanitized) and metrics under eval/outputs/<timestamp>/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse
from urllib.request import urlopen

# Load .env from the rllm directory (parent of eval/) before anything else,
# so env-var lookups in tool constructors always see the correct values.
try:
    from dotenv import load_dotenv as _load_dotenv

    _env_path = Path(__file__).resolve().parent.parent / ".env"
    _load_dotenv(dotenv_path=_env_path, override=True)
except ImportError:
    pass  # python-dotenv not installed; rely on shell export

from datasets import load_dataset

import yaml
from PIL import Image

from vision_deepresearch_async_workflow.deepresearch_tools_async_executor import (
    get_all_tools,
)
from vision_deepresearch_async_workflow.deepresearch_workflow import (
    DeepResearchWorkflow,
    PlannerExecutorDeepResearchWorkflow,
)
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout import OpenAIEngine
from rllm.rewards.reward_fn import deepresearch_reward_fn


# ---------------------- Data loading ---------------------- #


def _extract_from_content(content: list) -> tuple[str, List[Any]]:
    """Convert OpenAI-style content array (text + image_url) to question + images list."""
    texts: List[str] = []
    images: List[Any] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text" and "text" in item:
            texts.append(str(item["text"]))
        elif item.get("type") == "image_url":
            url = item.get("image_url", {}) or {}
            if isinstance(url, dict) and "url" in url:
                images.append({"url": url["url"]})
    question = "\n".join([t for t in texts if t.strip()])
    return question, images


def _url_to_bytes(url: str) -> dict:
    try:
        parsed = urlparse(url)
        if parsed.scheme in {"http", "https"}:
            with urlopen(url, timeout=10) as resp:
                data = resp.read()
                return {"bytes": data, "origin_url": url}
    except Exception:
        pass
    return {"url": url}


def _normalize_images(images: List[Any]) -> List[Any]:
    normalized: List[Any] = []
    for img in images:
        if isinstance(img, str) and img.startswith(("http://", "https://")):
            normalized.append(_url_to_bytes(img))
        elif isinstance(img, dict) and "url" in img and isinstance(img["url"], str):
            normalized.append(_url_to_bytes(img["url"]))
        else:
            normalized.append(img)
    return normalized


def _record_to_task(record: dict) -> dict:
    """Strict mapping to the unified schema question/answer/images."""
    question = record.get("question", "")
    answer = record.get("answer", "")
    images: List[Any] = []

    if "images" in record and record["images"] is not None:
        imgs = record["images"]
        images.extend(imgs if isinstance(imgs, list) else [imgs])

    if not isinstance(question, str):
        question = str(question)
    if not isinstance(answer, str):
        answer = str(answer)

    return {
        "id": record.get("id") or record.get("idx"),
        "question": question,
        "answer": answer,
        "images": _normalize_images(images),
    }


def _record_from_parquet(rec: dict, idx: int) -> dict:
    # Strict allowed fields
    if "question" not in rec or "answer" not in rec:
        raise ValueError(f"Record {idx} missing required fields 'question'/'answer'")

    question_raw = rec.get("question")
    answer_raw = rec.get("answer")

    question = str(question_raw) if question_raw is not None else ""
    if not question.strip():
        raise ValueError(f"Record {idx} has empty question")

    answer = str(answer_raw) if answer_raw is not None else ""
    if not answer.strip():
        raise ValueError(f"Record {idx} has empty answer")
    images_raw = rec.get("images", [])
    if images_raw is None:
        images_raw = []
    images_list: List[Any] = (
        images_raw if isinstance(images_raw, list) else [images_raw]
    )

    return {
        "id": rec.get("id") or rec.get("idx") or rec.get("_id"),
        "question": question,
        "answer": answer,
        "images": _normalize_images(images_list),
    }


def load_tasks(args) -> List[dict]:
    if not args.parquet:
        raise ValueError(
            "Parquet path is required. Please set --parquet or data.parquet."
        )

    ds_dict = load_dataset("parquet", data_files=str(args.parquet))
    ds = ds_dict["train"]

    allowed_cols = {"question", "answer", "images", "id", "idx", "_id","image_caption","question_original"}
    required_cols = {"question", "answer"}
    cols = set(ds.column_names)

    missing = required_cols - cols
    if missing:
        raise ValueError(f"Parquet file missing required columns: {sorted(missing)}")

    extras = cols - allowed_cols
    if extras:
        raise ValueError(
            f"Parquet file contains unsupported columns (allowed: question/answer/images/id): {sorted(extras)}"
        )

    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    tasks: List[dict] = []
    for idx, rec in enumerate(ds):
        rec_dict = dict(rec)
        task = _record_from_parquet(rec_dict, idx)
        tasks.append(task)

    if not tasks:
        raise ValueError("No valid tasks loaded from Parquet.")
    return tasks


# ---------------------- Rollout setup ---------------------- #


def build_rollout_engine(args):
    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.max_tokens is not None:
        sampling_params["max_tokens"] = args.max_tokens
    # OpenAIEngine also works with local OpenAI-compatible servers (e.g., vLLM)
    return OpenAIEngine(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        sampling_params=sampling_params,
    )


def build_executor_engine(args):
    """Build the small executor model engine for planner-executor mode."""
    sampling_params = {
        "temperature": args.executor_temperature,
        "top_p": args.top_p,
    }
    if args.executor_max_tokens is not None:
        sampling_params["max_tokens"] = args.executor_max_tokens
    return OpenAIEngine(
        model=args.executor_model,
        base_url=args.executor_base_url,
        api_key=args.executor_api_key,
        sampling_params=sampling_params,
    )


# ---------------------- Serialization helpers ---------------------- #


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, Image.Image):
        return {"type": "PIL.Image", "size": obj.size}
    if isinstance(obj, bytes):
        return f"<bytes:{len(obj)}>"
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def episode_to_dict(ep) -> Dict[str, Any]:
    trajectories = []
    for tr in ep.trajectories or []:
        steps = []
        for st in tr.steps or []:
            action_val = getattr(st.action, "action", st.action)
            steps.append(
                {
                    "chat_completions": _sanitize(st.chat_completions),
                    "model_response": _sanitize(st.model_response),
                    "action": _sanitize(action_val),
                    "observation": _sanitize(st.observation),
                    "reward": st.reward,
                }
            )
        trajectories.append(
            {
                "name": tr.name,
                "reward": tr.reward,
                "info": _sanitize(getattr(tr, "info", {})),
                "steps": steps,
            }
        )

    return {
        "id": ep.id,
        "task": _sanitize(ep.task),
        "termination_reason": (
            ep.termination_reason.value if ep.termination_reason else None
        ),
        "is_correct": ep.is_correct,
        "metrics": _sanitize(ep.metrics),
        "info": _sanitize(ep.info),
        "trajectories": trajectories,
    }


# ---------------------- Metrics & IO ---------------------- #


def compute_metrics(episodes: Iterable[Any]) -> dict:
    episodes = list(episodes)
    total = len(episodes)
    correct = sum(1 for ep in episodes if getattr(ep, "is_correct", False))
    termination = {}
    rewards: List[float] = []
    for ep in episodes:
        reason = ep.termination_reason.value if ep.termination_reason else "unknown"
        termination[reason] = termination.get(reason, 0) + 1
        if ep.trajectories:
            rewards.extend(
                [tr.reward for tr in ep.trajectories if tr.reward is not None]
            )
    avg_reward = (sum(rewards) / len(rewards)) if rewards else None
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "termination_distribution": termination,
        "average_reward": avg_reward,
    }


def save_outputs(episodes: List[Any], metrics: dict, output_dir: Path, config: dict):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSONL (one JSON object per line, compact) for programmatic use
    ep_path = output_dir / "episodes.jsonl"
    ep_dicts = [episode_to_dict(ep) for ep in episodes]
    with ep_path.open("w", encoding="utf-8") as f:
        for ep_dict in ep_dicts:
            f.write(json.dumps(ep_dict, ensure_ascii=False) + "\n")

    # Save as pretty-printed JSON array for human readability
    ep_pretty_path = output_dir / "episodes.json"
    with ep_pretty_path.open("w", encoding="utf-8") as f:
        json.dump(ep_dicts, f, indent=2, ensure_ascii=False)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved episodes to {ep_path} (compact) and {ep_pretty_path} (pretty)", flush=True)
    print(f"📊 Metrics: {metrics}", flush=True)


# ---------------------- Arg parsing ---------------------- #


def parse_args():
    parser = argparse.ArgumentParser(
        description="DeepResearch evaluation (Parquet only)"
    )
    parser.add_argument(
        "--config", default="eval/config/eval_hle.yaml", help="YAML config path"
    )

    # Data (Parquet only)
    parser.add_argument("--parquet", default=None, help="Local Parquet path (required)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples")

    # Rollout
    parser.add_argument(
        "--provider",
        default=None,
        help="openai (default) | vllm (still uses OpenAIEngine base_url)",
    )
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument(
        "--temperature", type=float, default=None, help="Sampling temperature"
    )
    parser.add_argument("--top-p", type=float, default=None, help="Top-p")
    parser.add_argument(
        "--max-tokens", type=int, default=None, help="Max tokens for completion"
    )

    # Execution
    parser.add_argument(
        "--parallel-tasks", type=int, default=None, help="Parallel tasks"
    )
    parser.add_argument("--retry-limit", type=int, default=None, help="Retry limit")
    parser.add_argument(
        "--max-rounds", type=int, default=None, help="Max LLM rounds per task (for debugging)"
    )
    parser.add_argument(
        "--output-dir", default=None, help="Output dir (relative to eval/)"
    )
    parser.add_argument(
        "--max-crops-per-call",
        type=int,
        default=None,
        help="Max number of image crops (bboxes) processed per crop_and_search call (default: 1)",
    )

    # ── Planner-Executor mode ──────────────────────────────────────────────────
    parser.add_argument(
        "--agent-mode",
        default="standard",
        choices=["standard", "planner-executor"],
        help=(
            "Agent architecture mode. "
            "'standard': single model handles both thinking and tool calls (default). "
            "'planner-executor': large model handles thinking only; "
            "a small trained model generates precise tool calls and bbox coordinates."
        ),
    )
    parser.add_argument(
        "--executor-model",
        default=None,
        help="[planner-executor] Small executor model name (e.g. Vision-DeepResearch-7B)",
    )
    parser.add_argument(
        "--executor-base-url",
        default=None,
        help="[planner-executor] OpenAI-compatible base URL for the executor model",
    )
    parser.add_argument(
        "--executor-api-key",
        default=None,
        help="[planner-executor] API key for the executor model (default: EMPTY for local vLLM)",
    )
    parser.add_argument(
        "--executor-temperature",
        type=float,
        default=0.0,
        help="[planner-executor] Sampling temperature for executor (default: 0.0 for determinism)",
    )
    parser.add_argument(
        "--executor-max-tokens",
        type=int,
        default=None,
        help="[planner-executor] Max tokens for executor completion",
    )

    return parser.parse_args()


def load_config(path: Path) -> dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def merge_args_with_config(cfg: dict, args) -> argparse.Namespace:
    # Rollout source
    provider = args.provider or cfg.get("provider", "openai")
    rollout_cfg = cfg.get(provider, {}) if isinstance(cfg, dict) else {}

    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    exec_cfg = cfg.get("execution", {}) if isinstance(cfg, dict) else {}
    out_cfg = cfg.get("output", {}) if isinstance(cfg, dict) else {}

    merged = argparse.Namespace(
        provider=provider,
        model=args.model or rollout_cfg.get("model") or "gpt-4o",
        base_url=args.base_url
        or rollout_cfg.get("base_url")
        or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=args.api_key
        or rollout_cfg.get("api_key")
        or os.getenv("OPENAI_API_KEY", ""),
        temperature=(
            args.temperature
            if args.temperature is not None
            else rollout_cfg.get("sampling_params", {}).get("temperature", 0.6)
        ),
        top_p=(
            args.top_p
            if args.top_p is not None
            else rollout_cfg.get("sampling_params", {}).get("top_p", 0.95)
        ),
        max_tokens=(
            args.max_tokens
            if args.max_tokens is not None
            else rollout_cfg.get("sampling_params", {}).get("max_tokens")
        ),
        parquet=args.parquet or data_cfg.get("parquet"),
        max_samples=(
            args.max_samples
            if args.max_samples is not None
            else data_cfg.get("max_samples")
        ),
        parallel_tasks=(
            args.parallel_tasks
            if args.parallel_tasks is not None
            else exec_cfg.get("parallel_tasks", 4)
        ),
        retry_limit=(
            args.retry_limit
            if args.retry_limit is not None
            else exec_cfg.get("retry_limit", 1)
        ),
        max_rounds=(
            args.max_rounds
            if args.max_rounds is not None
            else exec_cfg.get("max_rounds")
        ),
        max_crops_per_call=(
            args.max_crops_per_call
            if args.max_crops_per_call is not None
            else exec_cfg.get("max_crops_per_call")  # None means no limit
        ),
        output_dir=args.output_dir or out_cfg.get("dir") or str(Path(__file__).resolve().parent.parent.parent / "data" / "eval_outputs"),
        # Planner-executor specific (no config file override for these)
        agent_mode=args.agent_mode,
        executor_model=args.executor_model,
        executor_base_url=args.executor_base_url or os.getenv("EXECUTOR_BASE_URL"),
        executor_api_key=args.executor_api_key or os.getenv("EXECUTOR_API_KEY", "EMPTY"),
        executor_temperature=args.executor_temperature,
        executor_max_tokens=args.executor_max_tokens,
    )
    return merged


# ---------------------- Main ---------------------- #


async def main():
    args_cli = parse_args()
    cfg = load_config(Path(args_cli.config))
    args = merge_args_with_config(cfg, args_cli)

    tasks = load_tasks(args)
    if not tasks:
        raise ValueError("No tasks loaded. Please check dataset or JSONL path.")

    # Propagate max_crops_per_call to the tool via environment variable (empty string = no limit)
    os.environ["MAX_CROPS_PER_CALL"] = str(args.max_crops_per_call) if args.max_crops_per_call is not None else ""

    tools = get_all_tools()
    rollout_engine = build_rollout_engine(args)

    workflow_args = {
        "tools": tools,
        "reward_function": deepresearch_reward_fn,
    }
    if args.max_rounds is not None:
        workflow_args["max_llm_calls"] = args.max_rounds

    # Select workflow class based on agent mode
    if args.agent_mode == "planner-executor":
        if not args.executor_model:
            raise ValueError(
                "--executor-model is required when --agent-mode=planner-executor"
            )
        if not args.executor_base_url:
            raise ValueError(
                "--executor-base-url is required when --agent-mode=planner-executor"
            )
        executor_engine = build_executor_engine(args)
        workflow_args["executor_engine"] = executor_engine
        workflow_cls = PlannerExecutorDeepResearchWorkflow
        print(
            f"🔀 Agent mode: planner-executor\n"
            f"   Planner : {args.model} @ {args.base_url}\n"
            f"   Executor: {args.executor_model} @ {args.executor_base_url}",
            flush=True,
        )
    else:
        workflow_cls = DeepResearchWorkflow
        print(f"🤖 Agent mode: standard ({args.model})", flush=True)

    workflow_engine = AgentWorkflowEngine(
        workflow_cls=workflow_cls,
        workflow_args=workflow_args,
        rollout_engine=rollout_engine,
        n_parallel_tasks=args.parallel_tasks,
        retry_limit=args.retry_limit,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build a meaningful directory name:
    #   {timestamp}_{model_short}_{dataset_short}[_r{max_rounds}][_c{max_crops}][_n{max_samples}]
    def _slugify(s: str) -> str:
        """Sanitize a string for use as a directory name component."""
        import re
        s = s.replace("\\", "/")
        s = s.split("/")[-1]          # keep only the last path segment
        s = re.sub(r"[^\w\-.]", "_", s)  # replace non-alphanumeric chars
        s = re.sub(r"_+", "_", s)     # collapse consecutive underscores
        return s.strip("_")

    planner_slug = _slugify(args.model)[:24]
    if args.agent_mode == "planner-executor" and args.executor_model:
        executor_slug = _slugify(args.executor_model)[:24]
        # e.g. "gemini-3.1-pro+VDR-7B"
        model_slug = f"{planner_slug}+{executor_slug}"
    else:
        model_slug = planner_slug

    dataset_slug = _slugify(Path(args.parquet).stem) if args.parquet else "unknown"
    dataset_slug = dataset_slug[:32]

    dir_parts = [timestamp, model_slug, dataset_slug]
    if args.agent_mode == "planner-executor":
        dir_parts.append("pe")           # mark as planner-executor run
    if args.max_rounds is not None:
        dir_parts.append(f"r{args.max_rounds}")
    if args.max_crops_per_call is not None:
        dir_parts.append(f"c{args.max_crops_per_call}")
    if args.max_samples is not None:
        dir_parts.append(f"n{args.max_samples}")

    run_name = "_".join(dir_parts)
    output_dir = Path(args.output_dir) / model_slug / dataset_slug / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config upfront so it's available even if the run is interrupted
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    ep_jsonl_path = output_dir / "episodes.jsonl"

    print(f"🚀 Running DeepResearch evaluation with {len(tasks)} tasks", flush=True)
    print(f"📂 Output dir: {output_dir}", flush=True)

    completed_episodes: List[Any] = []
    done_count = 0

    episodes = await workflow_engine.execute_tasks(tasks)

    # Write each episode immediately after it is collected
    with ep_jsonl_path.open("a", encoding="utf-8") as jsonl_f:
        for ep in episodes:
            ep_dict = episode_to_dict(ep)
            jsonl_f.write(json.dumps(ep_dict, ensure_ascii=False) + "\n")
            jsonl_f.flush()
            completed_episodes.append(ep)
            done_count += 1
            is_correct = getattr(ep, "is_correct", False)
            print(
                f"✅ [{done_count}/{len(tasks)}] ep_id={ep.id} correct={is_correct}",
                flush=True,
            )

    metrics = compute_metrics(completed_episodes)

    # Save pretty JSON and metrics after all tasks finish
    ep_pretty_path = output_dir / "episodes.json"
    ep_dicts = [episode_to_dict(ep) for ep in completed_episodes]
    with ep_pretty_path.open("w", encoding="utf-8") as f:
        json.dump(ep_dicts, f, indent=2, ensure_ascii=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved episodes to {ep_jsonl_path} (compact) and {ep_pretty_path} (pretty)", flush=True)
    print(f"📊 Metrics: {metrics}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
