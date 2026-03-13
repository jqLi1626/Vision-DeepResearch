#!/usr/bin/env python3
"""
轨迹深度分析脚本（增强版）
- 尽量保留完整轨迹信息（减少截断）
- 将原始图片 + 所有 crop_and_search 裁剪区域以图片形式传给 AI
- 对比分析四组实验：
    0_before  : 8b 训练模型 + 反向检索0条图片（困难，网上无匹配）
    21+_before: 8b 训练模型 + 反向检索21+条图片（容易，网上有匹配）
    0         : GPT-5.1    + 反向检索0条图片
    21+       : GPT-5.1    + 反向检索21+条图片
"""

import json
import re
import os
import time
import base64
import io
from collections import defaultdict
from openai import OpenAI

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[警告] PIL 不可用，将跳过图片传入功能")

# ────────────────────────────────────────────────────────────────────
# 配置
# ────────────────────────────────────────────────────────────────────
AI_CLIENT = OpenAI(
    api_key="sk-btCGkgdyFPhB2aowxniX7K3zXChEZzqsdUT82GHdMkCnDtkC",
    base_url="https://aiberm.com/v1"
)
AI_MODEL = "google/gemini-3.1-pro"

FILES = {
    "0_before":   "/home/yangyajie/ljq/Vision-DeepResearch/data/eval_outputs/0_before/episodes.json",
    "21+_before": "/home/yangyajie/ljq/Vision-DeepResearch/data/eval_outputs/21+_before/episodes.json",
    "0":          "/home/yangyajie/ljq/Vision-DeepResearch/data/eval_outputs/0/episodes.json",
    "21+":        "/home/yangyajie/ljq/Vision-DeepResearch/data/eval_outputs/21+/episodes.json",
}

OUTPUT_REPORT = "/home/yangyajie/ljq/Vision-DeepResearch/data/eval_analysis/trajectory_analysis_report.md"

# 最多传入 AI 的裁剪图片数（原图1张 + 裁剪图若干）
# gemini-3.1-pro 限制为请求总大小<20MB，16张图片约600KB，远未达到限制
MAX_CROP_IMAGES = 15

# ────────────────────────────────────────────────────────────────────
# 图片工具
# ────────────────────────────────────────────────────────────────────

def load_pil_image(image_path: str):
    """加载图片，返回 PIL.Image 或 None"""
    if not PIL_AVAILABLE:
        return None
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  [图片加载失败] {image_path}: {e}")
        return None

def crop_image(pil_img, bbox: list):
    """按 bbox [x1, y1, x2, y2] 裁剪图片，修正越界坐标"""
    if pil_img is None or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        w, h = pil_img.size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return pil_img.crop((x1, y1, x2, y2))
    except Exception:
        return None

def pil_to_base64(pil_img, max_size=512) -> str:
    """将 PIL.Image 转为 base64 字符串（适度压缩以节省 token）"""
    if pil_img is None:
        return ""
    try:
        w, h = pil_img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return ""

def build_image_content_parts(pil_img, all_bboxes: list) -> list:
    """
    构造传给 AI 的图片内容列表：
    - 第一张：原始完整图片
    - 后续：每个 crop_and_search bbox 的裁剪区域（去重，最多 MAX_CROP_IMAGES 张）
    返回格式：[{"type": "image_url", "image_url": {...}}, ...]
    """
    parts = []
    if pil_img is None:
        return parts

    # 原始图片
    b64 = pil_to_base64(pil_img, max_size=768)
    if b64:
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    # 裁剪区域：去重
    seen = set()
    crop_count = 0
    for bbox in all_bboxes:
        if crop_count >= MAX_CROP_IMAGES:
            break
        key = tuple(bbox)
        if key in seen:
            continue
        seen.add(key)
        cropped = crop_image(pil_img, bbox)
        if cropped is None:
            continue
        b64c = pil_to_base64(cropped, max_size=384)
        if b64c:
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64c}"}
            })
            crop_count += 1

    return parts

# ────────────────────────────────────────────────────────────────────
# 轨迹解析工具
# ────────────────────────────────────────────────────────────────────

def extract_think(content: str) -> str:
    """提取 <think>...</think>，保留尽量完整内容"""
    m = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    if m:
        return m.group(1).strip()
    m2 = re.search(r'<think\s*(.*?)(?:</think>|<tool_call>|<answer>)', content, re.DOTALL)
    if m2:
        return m2.group(1).strip()
    return ""

def extract_final_answer_text(content: str) -> str:
    m = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""

def parse_tool_call(action: dict) -> dict:
    if action.get("type") != "tool_call":
        return {}
    raw = action.get("tool_call", "{}")
    try:
        tc = json.loads(raw)
        name = tc.get("name", "unknown")
        args = tc.get("arguments", {})
        if name == "crop_and_search":
            bboxes = args.get("bbox", [])
            goal = args.get("goal", "")
            return {"name": name, "bbox_count": len(bboxes), "bboxes": bboxes, "goal": goal}
        elif name == "search":
            return {"name": name, "queries": args.get("query", [])}
        elif name == "visit":
            return {"name": name, "urls": args.get("url", [])}
        else:
            return {"name": name, "args": str(args)[:300]}
    except Exception:
        return {"name": "parse_error", "raw": raw[:200]}

def extract_obs_full(observation: str) -> str:
    """
    保留 observation 中每个 bbox 结果的完整摘要（不做大截断）
    去掉过长的重复内容，保留 Summary + Evidence 段落
    """
    if not observation:
        return ""
    lines = observation.split("\n")
    result_lines = []
    include = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("The search results for bbox"):
            result_lines.append("")
            result_lines.append(stripped)
            include = False
            continue
        if "useful information in [" in stripped:
            result_lines.append(stripped)
            include = True
            continue
        if "Evidence in page:" in stripped or "Summary:" in stripped:
            include = True
        if include and stripped:
            result_lines.append(stripped)
        if stripped == "=======":
            include = False
            result_lines.append("---")
    text = "\n".join(result_lines)
    # 最多保留 3000 字，避免单条 obs 撑爆 token
    if len(text) > 3000:
        text = text[:3000] + f"\n...[观测结果过长，已截断至3000字，原长{len(text)}字]"
    return text

def build_trajectory_full(ep: dict) -> dict:
    """提取 episode 的完整轨迹信息（尽量保留原始内容）"""
    info = ep.get("info", {})
    task = ep.get("task", {})
    question = task.get("question", "")
    correct_answer = task.get("answer", "")
    image_paths = task.get("images", [])
    prediction = info.get("prediction", "")
    is_correct = ep.get("is_correct", False)
    termination = ep.get("termination_reason", "")
    rounds = info.get("rounds", 0)
    masked = ep.get("metrics", {}).get("masked", 0)
    mask_reason = info.get("mask_reason", "")
    token_usage = info.get("token_usage", {})
    reward_meta = info.get("reward_metadata", {})
    judgment = reward_meta.get("judgment", "")

    steps_info = []
    all_bboxes = []  # 收集所有 crop bbox，用于图片裁剪

    for traj in ep.get("trajectories", []):
        for step_idx, step in enumerate(traj.get("steps", [])):
            action = step.get("action", {})
            observation = step.get("observation", "")
            step_error = False

            think_text = ""
            final_ans_text = ""
            completions = step.get("chat_completions", [])
            for comp in completions:
                if comp.get("role") == "assistant":
                    content = comp.get("content", "")
                    step_error = comp.get("step_error", False)
                    t = extract_think(content)
                    if t:
                        think_text = t
                    fa = extract_final_answer_text(content)
                    if fa:
                        final_ans_text = fa

            tool_info = parse_tool_call(action)
            obs_text = extract_obs_full(observation)

            # 收集 bbox 供后续图片裁剪
            if tool_info.get("name") == "crop_and_search":
                for bbox in tool_info.get("bboxes", []):
                    all_bboxes.append(bbox)

            steps_info.append({
                "step": step_idx + 1,
                "think": think_text,        # 不截断
                "tool": tool_info,
                "observation": obs_text,    # 保留完整摘要
                "final_answer_in_step": final_ans_text,
                "step_error": step_error,
            })

    return {
        "question": question,
        "correct_answer": correct_answer,
        "image_paths": image_paths,
        "prediction": prediction,           # 不截断
        "is_correct": is_correct,
        "termination": termination,
        "rounds": rounds,
        "masked": masked,
        "mask_reason": mask_reason,
        "prompt_tokens": token_usage.get("prompt", 0),
        "completion_tokens": token_usage.get("completion", 0),
        "judge_judgment": judgment,         # 不截断
        "steps": steps_info,
        "all_bboxes": all_bboxes,
    }

def format_trajectory_for_ai(group_name: str, ep_idx: int, traj: dict) -> str:
    """格式化轨迹文本，发给 AI 时附图片"""
    lines = []
    lines.append(f"=== 轨迹 [{group_name}] 第{ep_idx+1}条 ===")
    lines.append(f"问题: {traj['question']}")
    lines.append(f"正确答案: {traj['correct_answer']}")
    lines.append(f"模型最终预测: {traj['prediction']}")
    lines.append(f"是否正确: {'✓ 正确' if traj['is_correct'] else '✗ 错误'}")
    lines.append(f"终止原因: {traj['termination']} | masked={traj['masked']} | mask_reason={traj['mask_reason']}")
    lines.append(f"总轮数: {traj['rounds']} | prompt_tokens={traj['prompt_tokens']} | completion_tokens={traj['completion_tokens']}")
    if traj["judge_judgment"]:
        lines.append(f"Judge评判: {traj['judge_judgment']}")
    lines.append("")

    # 图片说明
    bboxes = traj["all_bboxes"]
    if bboxes:
        lines.append(f"[图片说明] 原始图片已附上，另附 {min(len(bboxes), MAX_CROP_IMAGES)} 张裁剪区域图片。")
        lines.append(f"图片顺序：第1张=原图，第2~{1+min(len(bboxes),MAX_CROP_IMAGES)}张=裁剪区域（按步骤顺序）")
        lines.append(f"所有 crop bbox 坐标（[x1,y1,x2,y2]）: {bboxes}")
    else:
        lines.append("[图片说明] 无 crop_and_search 操作，仅附原始图片。")
    lines.append("")
    lines.append("=" * 50)
    lines.append("步骤详情（完整轨迹）")
    lines.append("=" * 50)

    for s in traj["steps"]:
        lines.append(f"\n【Step {s['step']}】{'⚠️ step_error=True' if s['step_error'] else ''}")

        if s["think"]:
            lines.append(f"<模型思考>:\n{s['think']}")

        tool = s["tool"]
        if tool:
            tname = tool.get("name", "")
            if tname == "crop_and_search":
                lines.append(f"<工具调用>: crop_and_search")
                lines.append(f"  goal: {tool.get('goal','')}")
                lines.append(f"  bbox数量: {tool.get('bbox_count',0)}")
                lines.append(f"  bboxes: {tool.get('bboxes',[])}")
            elif tname == "search":
                lines.append(f"<工具调用>: search")
                lines.append(f"  queries: {tool.get('queries', [])}")
            elif tname == "visit":
                lines.append(f"<工具调用>: visit")
                lines.append(f"  urls: {tool.get('urls', [])}")
            elif tname == "PythonInterpreter":
                lines.append(f"<工具调用>: PythonInterpreter")
            else:
                lines.append(f"<工具调用>: {tname} | {tool.get('args','')}")

        if s["observation"]:
            lines.append(f"<检索结果>:\n{s['observation']}")

        if s["final_answer_in_step"]:
            lines.append(f"<最终答案>: {s['final_answer_in_step']}")

    return "\n".join(lines)

# ────────────────────────────────────────────────────────────────────
# AI 分析函数
# ────────────────────────────────────────────────────────────────────

def ai_analyze_trajectory(traj_text: str, image_parts: list, group_name: str) -> str:
    """
    用 Gemini 分析单条轨迹
    image_parts: [{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}, ...]
    第1张图 = 原图，后续 = 各 crop 区域（按 step 顺序）
    """
    prompt_text = f"""你是一名AI系统评估专家。以下是一个视觉深度研究智能体（Visual Deep Research Agent）处理一道图片问答题的完整轨迹。

工具说明：
- crop_and_search：裁剪图片某个局部区域，对裁剪出的图片做反向图像检索，返回网上匹配结果
- search：文字关键词搜索
- visit：访问网页

实验组：{group_name}
- "0_before"/"21+_before"：专门针对此任务训练的 8B 小模型
- "0"/"21+"：强大商业模型 GPT-5.1
- "0条"图片：该图片在网上反向检索几乎无匹配（困难，模型必须靠视觉理解+推理）
- "21+条"图片：该图片在网上有大量匹配（较容易，检索结果可直接给出线索）

附图说明：第1张=原始图片，后续=crop_and_search 各裁剪区域（按步骤顺序附上）。
请结合图片理解模型的视觉感知和裁剪策略。

{traj_text}

请从以下角度进行分析：
1. **视觉理解质量**（结合原图）：模型对图片的初始感知是否准确？裁剪区域是否对准了关键信息？
2. **检索策略分析**：每次 crop_and_search 的 bbox 区域选择是否合理？检索结果是否返回了有价值的线索？
3. **信息合成与推理**：模型如何利用检索结果？推理链是否清晰？在哪一步出现了关键错误？
4. **错误根源**（如答错）：是视觉误判、检索无效、正确线索被忽略、信息合成错误、还是模型幻觉？
5. **特殊问题**：是否有重复生成、截断、格式错误、过度自信等问题？

请用中文给出结构化分析（约300字），重点突出最关键的问题。"""

    content_parts = [{"type": "text", "text": prompt_text}] + image_parts

    try:
        resp = AI_CLIENT.chat.completions.create(
            model=AI_MODEL,
            messages=[{"role": "user", "content": content_parts}],
            max_tokens=2000,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        # 多模态失败，降级纯文本
        print(f"    [多模态失败，降级纯文本]: {e}")
        try:
            resp = AI_CLIENT.chat.completions.create(
                model=AI_MODEL,
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=2000,
            )
            return resp.choices[0].message.content.strip() + "\n[注：图片传入失败，仅文字分析]"
        except Exception as e2:
            return f"[AI分析失败: {e2}]"

def ai_group_summary(group_name: str, individual_analyses: list, stats: dict) -> str:
    """对一组所有轨迹的分析结果做整体总结"""
    combined = "\n\n".join([f"案例{i+1}:\n{a}" for i, a in enumerate(individual_analyses)])
    prompt = f"""你是一名AI系统评估专家，以下是实验组 "{group_name}" 的所有轨迹逐条分析结果。

实验组说明：
- "0_before"/"21+_before"：8B训练模型
- "0"/"21+"：GPT-5.1
- "0条"：困难题（图片无法被反向检索）
- "21+条"：容易题（图片可被反向检索）

统计数据：
- 总题数: {stats['total']}，正确: {stats['correct']}，准确率: {stats['accuracy']:.1%}
- masked率: {stats['masked_rate']:.1%}（被检测为重复输出而强制终止）
- 平均轮数: {stats['avg_rounds']:.1f}
- 平均 prompt tokens: {stats['avg_prompt_tokens']:.0f}，completion tokens: {stats['avg_completion_tokens']:.0f}
- 工具调用: {stats['tool_counts']}

逐条分析：
{combined}

请总结：
1. **核心共性问题**：这组模型最突出的共性错误是什么？
2. **工具使用规律**：bbox 选择、检索次数等有何系统性问题？
3. **难度适应性**：模型对这类图片（0条/21+条）的适应程度如何？
4. **改进方向**：针对发现的问题，最值得优先改进的方向是什么？

请用结构化中文输出（约350字）。"""

    try:
        resp = AI_CLIENT.chat.completions.create(
            model=AI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[组总结失败: {e}]"

def ai_cross_comparison(group_summaries: dict, all_stats: dict) -> str:
    """跨组横向对比总结"""
    text_parts = []
    for gname, summary in group_summaries.items():
        st = all_stats[gname]
        text_parts.append(
            f"【{gname}】\n"
            f"准确率={st['accuracy']:.1%}, masked率={st['masked_rate']:.1%}, "
            f"平均轮数={st['avg_rounds']:.1f}, 工具={st['tool_counts']}\n"
            f"组总结:\n{summary}"
        )
    combined = "\n\n" + ("─" * 50 + "\n").join(text_parts)

    prompt = f"""你是一名AI系统评估专家，正在对比分析四组实验结果。

实验矩阵：
           | 0条图片（困难）  | 21+条图片（容易）
8b训练模型  | 0_before        | 21+_before
GPT-5.1    | 0               | 21+

各组详情：
{combined}

请进行深入横向对比：
1. **模型能力差异**（8b vs GPT-5.1）：在推理深度、工具使用、幻觉程度、错误类型上有何本质区别？
2. **难度的影响**（0条 vs 21+条）：反向检索是否成功，对两个模型的影响有何不同？
3. **8b模型特有问题**：重复生成、格式错乱、幻觉等相比 GPT-5.1 有多严重？
4. **GPT-5.1 的瓶颈**：即使强模型，在什么情况下仍然失败？为什么？
5. **核心结论与训练启示**：这组实验揭示了什么？对改进8b模型的训练有何具体指导意义？

请输出深度对比报告（约500字，条理清晰）。"""

    try:
        resp = AI_CLIENT.chat.completions.create(
            model=AI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[跨组对比失败: {e}]"

# ────────────────────────────────────────────────────────────────────
# 统计函数
# ────────────────────────────────────────────────────────────────────

def compute_stats(episodes: list) -> dict:
    total = len(episodes)
    correct = sum(1 for ep in episodes if ep.get("is_correct", False))
    masked_count = sum(1 for ep in episodes if ep.get("metrics", {}).get("masked", 0) > 0)
    rounds_list = [ep.get("info", {}).get("rounds", 0) for ep in episodes]
    prompt_toks = [ep.get("info", {}).get("token_usage", {}).get("prompt", 0) for ep in episodes]
    comp_toks   = [ep.get("info", {}).get("token_usage", {}).get("completion", 0) for ep in episodes]

    terminations = defaultdict(int)
    for ep in episodes:
        terminations[ep.get("termination_reason", "unknown")] += 1

    tool_counts = defaultdict(int)
    crop_bbox_counts = []
    for ep in episodes:
        for traj in ep.get("trajectories", []):
            for step in traj.get("steps", []):
                action = step.get("action", {})
                if action.get("type") == "tool_call":
                    try:
                        tc = json.loads(action.get("tool_call", "{}"))
                        name = tc.get("name", "unknown")
                        tool_counts[name] += 1
                        if name == "crop_and_search":
                            bboxes = tc.get("arguments", {}).get("bbox", [])
                            crop_bbox_counts.append(len(bboxes))
                    except Exception:
                        pass

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0,
        "masked": masked_count,
        "masked_rate": masked_count / total if total else 0,
        "avg_rounds": sum(rounds_list) / len(rounds_list) if rounds_list else 0,
        "avg_prompt_tokens": sum(prompt_toks) / len(prompt_toks) if prompt_toks else 0,
        "avg_completion_tokens": sum(comp_toks) / len(comp_toks) if comp_toks else 0,
        "terminations": dict(terminations),
        "tool_counts": dict(tool_counts),
        "avg_bboxes_per_crop": sum(crop_bbox_counts) / len(crop_bbox_counts) if crop_bbox_counts else 0,
        "rounds_list": rounds_list,
        "prompt_tokens_list": prompt_toks,
        "comp_tokens_list": comp_toks,
    }

def detect_special_patterns(episodes: list) -> dict:
    repetition_cases, truncated_cases, hallucination_cases, bad_bbox_cases = [], [], [], []

    for ep in episodes:
        ep_id = ep.get("id", "")[:20]
        question = ep.get("task", {}).get("question", "")
        correct_answer = ep.get("task", {}).get("answer", "")
        prediction = ep.get("info", {}).get("prediction", "")
        mask_reason = ep.get("info", {}).get("mask_reason", "")
        comp_tokens = ep.get("info", {}).get("token_usage", {}).get("completion", 0)
        rounds = ep.get("info", {}).get("rounds", 0)

        if mask_reason == "repetition_detected" or ep.get("metrics", {}).get("masked", 0) > 0:
            repetition_cases.append({"id": ep_id, "question": question[:80], "mask_reason": mask_reason})

        if comp_tokens >= 4000:
            truncated_cases.append({"id": ep_id, "question": question[:80], "comp_tokens": comp_tokens})

        if rounds <= 1 and not ep.get("is_correct", False):
            hallucination_cases.append({
                "id": ep_id, "question": question[:80],
                "prediction": prediction[:120], "correct": correct_answer,
            })

        for traj in ep.get("trajectories", []):
            for step in traj.get("steps", []):
                action = step.get("action", {})
                if action.get("type") == "tool_call":
                    try:
                        tc = json.loads(action.get("tool_call", "{}"))
                        if tc.get("name") == "crop_and_search":
                            for bbox in tc.get("arguments", {}).get("bbox", []):
                                if len(bbox) == 4:
                                    x1, y1, x2, y2 = bbox
                                    if (x2 - x1) < 20 or (y2 - y1) < 20:
                                        bad_bbox_cases.append({
                                            "id": ep_id, "bbox": bbox,
                                            "question": question[:60]
                                        })
                    except Exception:
                        pass

    return {
        "repetition_cases": repetition_cases,
        "truncated_cases": truncated_cases,
        "hallucination_cases": hallucination_cases,
        "bad_bbox_cases": bad_bbox_cases,
    }

# ────────────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("视觉深度研究智能体 — 完整轨迹分析（含图片）")
    print("=" * 70)
    if not PIL_AVAILABLE:
        print("⚠️  PIL 不可用，将跳过图片传入（仅文字分析）")

    all_data, all_stats, all_patterns = {}, {}, {}
    all_group_ai_summaries = {}
    report_lines = []

    report_lines.append("# 视觉深度研究智能体 — 轨迹深度分析报告（含图片）\n")
    report_lines.append("## 实验设计\n")
    report_lines.append("""
|  | 0条图片（困难，网上无匹配） | 21+条图片（容易，网上有匹配） |
|--|--------------------------|---------------------------|
| **8b训练模型** | `0_before` | `21+_before` |
| **GPT-5.1** | `0` | `21+` |

"0条/21+条" = 该图片在互联网上通过反向图像检索能找到的匹配数量，代表题目难度。
分析时对每条轨迹传入**原始图片 + 所有裁剪区域图片**，尽量保留完整轨迹信息。
""")

    # ── Step 1: 加载 & 统计 ─────────────────────────────────────────
    print("\n[Step 1] 加载数据 & 基础统计...")
    report_lines.append("\n## 一、基础统计\n")
    report_lines.append("| 组别 | 总题 | 正确 | 准确率 | masked率 | 均轮数 | 均prompt_tok | 均comp_tok |")
    report_lines.append("|------|------|------|--------|---------|--------|-------------|------------|")

    for group_name, fpath in FILES.items():
        data = json.load(open(fpath, encoding="utf-8"))
        all_data[group_name] = data
        stats = compute_stats(data)
        all_stats[group_name] = stats
        all_patterns[group_name] = detect_special_patterns(data)
        print(f"  [{group_name:12s}] 总={stats['total']}, 正确={stats['correct']}, "
              f"准确率={stats['accuracy']:.1%}, masked={stats['masked']}, "
              f"avg_rounds={stats['avg_rounds']:.1f}")
        report_lines.append(
            f"| `{group_name}` | {stats['total']} | {stats['correct']} | "
            f"{stats['accuracy']:.1%} | {stats['masked_rate']:.1%} | "
            f"{stats['avg_rounds']:.1f} | {stats['avg_prompt_tokens']:.0f} | "
            f"{stats['avg_completion_tokens']:.0f} |"
        )

    # ── Step 2: 工具统计 ────────────────────────────────────────────
    print("\n[Step 2] 工具使用统计...")
    report_lines.append("\n## 二、工具使用统计\n")
    report_lines.append("| 组别 | crop_and_search次 | search次 | visit次 | PythonInterpreter次 | 均bbox数/次crop |")
    report_lines.append("|------|------------------|---------|---------|---------------------|--------------|")
    for group_name, stats in all_stats.items():
        tc = stats["tool_counts"]
        report_lines.append(
            f"| `{group_name}` | {tc.get('crop_and_search',0)} | {tc.get('search',0)} | "
            f"{tc.get('visit',0)} | {tc.get('PythonInterpreter',0)} | "
            f"{stats['avg_bboxes_per_crop']:.1f} |"
        )

    # ── Step 3: 特殊问题检测 ────────────────────────────────────────
    print("\n[Step 3] 特殊问题检测...")
    report_lines.append("\n## 三、特殊问题检测\n")
    for group_name, patterns in all_patterns.items():
        report_lines.append(f"### [{group_name}]\n")
        if patterns["repetition_cases"]:
            report_lines.append(f"**🔴 重复生成（masked）**：{len(patterns['repetition_cases'])} 例")
            for c in patterns["repetition_cases"]:
                report_lines.append(f"  - `{c['id']}` | {c['question']} | 原因: `{c['mask_reason']}`")
        if patterns["truncated_cases"]:
            report_lines.append(f"\n**🟡 输出截断（completion_tokens≥4000）**：{len(patterns['truncated_cases'])} 例")
            for c in patterns["truncated_cases"]:
                report_lines.append(f"  - `{c['id']}` | {c['question']} | comp_tokens={c['comp_tokens']}")
        if patterns["hallucination_cases"]:
            report_lines.append(f"\n**🟠 快速错误（≤1轮即答错，疑似幻觉）**：{len(patterns['hallucination_cases'])} 例")
            for c in patterns["hallucination_cases"]:
                report_lines.append(f"  - 问题: {c['question']}")
                report_lines.append(f"    预测: {c['prediction']}")
                report_lines.append(f"    正确: {c['correct']}")
        if patterns["bad_bbox_cases"]:
            report_lines.append(f"\n**🟡 异常小bbox（宽或高<20px）**：{len(patterns['bad_bbox_cases'])} 例")
            for c in patterns["bad_bbox_cases"][:5]:
                report_lines.append(f"  - {c['question']} | bbox={c['bbox']}")
        if not any(patterns.values()):
            report_lines.append("  ✅ 暂无明显特殊问题\n")
        report_lines.append("")

    # ── Step 4: 逐条 AI 分析（含图片）──────────────────────────────
    print("\n[Step 4] AI 逐条轨迹分析（含原图 + 裁剪图）...")
    report_lines.append("\n## 四、逐条轨迹 AI 分析\n")

    for group_name, episodes in all_data.items():
        print(f"\n  ▶ 组别 [{group_name}] — {len(episodes)} 条轨迹")
        report_lines.append(f"\n### 组别: `{group_name}`\n")
        individual_analyses = []

        for ep_idx, ep in enumerate(episodes):
            traj = build_trajectory_full(ep)
            traj_text = format_trajectory_for_ai(group_name, ep_idx, traj)

            # 加载原图 & 准备裁剪图
            image_parts = []
            if PIL_AVAILABLE:
                img_path = traj["image_paths"][0] if traj["image_paths"] else ""
                pil_img = load_pil_image(img_path)
                image_parts = build_image_content_parts(pil_img, traj["all_bboxes"])
                img_info = f"原图{'✓' if pil_img else '✗'}, 裁剪图{len(image_parts)-1 if image_parts else 0}张"
            else:
                img_info = "PIL不可用"

            print(f"    [{ep_idx+1:2d}/{len(episodes)}] {img_info} | Q: {traj['question'][:55]}")

            analysis = ai_analyze_trajectory(traj_text, image_parts, group_name)
            individual_analyses.append(analysis)

            # 写入报告
            report_lines.append(f"#### 案例 {ep_idx+1}: `{traj['question'][:80]}`")
            report_lines.append(f"- **正确答案**: {traj['correct_answer']}")
            report_lines.append(f"- **模型预测**: {traj['prediction'][:200]}")
            report_lines.append(
                f"- **结果**: {'✓ 正确' if traj['is_correct'] else '✗ 错误'} | "
                f"轮数={traj['rounds']} | termination=`{traj['termination']}` | "
                f"masked={traj['masked']} | {img_info}"
            )
            report_lines.append(f"\n**AI 分析**:\n\n{analysis}\n")
            report_lines.append("---\n")

            time.sleep(0.8)

        all_group_ai_summaries[group_name] = individual_analyses

        # 组总结
        print(f"  [{group_name}] 生成组总结...")
        group_summary = ai_group_summary(group_name, individual_analyses, all_stats[group_name])
        report_lines.append(f"### `{group_name}` 组总结\n")
        report_lines.append(group_summary)
        report_lines.append("\n" + "═" * 60 + "\n")
        time.sleep(1.0)

    # ── Step 5: 跨组对比 ────────────────────────────────────────────
    print("\n[Step 5] 跨组横向对比分析...")
    report_lines.append("\n## 五、跨组横向对比分析\n")
    cross_analysis = ai_cross_comparison(all_group_ai_summaries, all_stats)
    report_lines.append(cross_analysis)

    # ── Step 6: 写报告 ───────────────────────────────────────────────
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n✅ 完成！报告已保存至: {OUTPUT_REPORT}")
    print("\n" + "=" * 70)
    print("快速汇总:")
    for group_name, stats in all_stats.items():
        print(f"  [{group_name:12s}] 准确率={stats['accuracy']:.1%}  "
              f"masked率={stats['masked_rate']:.1%}  "
              f"均轮数={stats['avg_rounds']:.1f}  "
              f"工具={stats['tool_counts']}")
    print("=" * 70)
    print("\n📊 跨组对比:\n")
    print(cross_analysis)

if __name__ == "__main__":
    main()
