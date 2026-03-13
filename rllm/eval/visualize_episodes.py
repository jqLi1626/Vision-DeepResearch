#!/usr/bin/env python3
"""
VDR-Bench Episodes 可视化工具
==============================
用法示例：
  # 可视化指定目录（含图片）
  python visualize_episodes.py --dirs outputs/0 outputs/6-9 outputs/21+

  # 自动发现 outputs/ 下所有子目录
  python visualize_episodes.py

  # 不嵌入图片（文件更小，生成更快）
  python visualize_episodes.py --no-images

  # 控制缩略图尺寸（默认 480px）
  python visualize_episodes.py --img-size 320

  # 指定输出路径 / 标题 / 过滤模式
  python visualize_episodes.py --dirs outputs/0 -o report.html --title "实验A" --filter incorrect
"""

import argparse
import base64
import html as html_module
import json
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except ImportError:
    PIL_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# 图片工具
# ─────────────────────────────────────────────────────────────────────────────

# 颜色池：为每个 bbox 分配不同颜色
BBOX_COLORS = [
    "#FF4C4C", "#4CFF8F", "#4CA8FF", "#FFD54C",
    "#FF4CFF", "#4CFFF0", "#FF944C", "#B44CFF",
]


def _img_to_b64(img: "Image.Image", quality: int = 82) -> str:
    """PIL Image → data URI (JPEG base64)"""
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def load_image_b64(path: str, max_size: int = 480) -> str | None:
    """加载图片并缩放，返回 base64 data URI；失败返回 None"""
    if not PIL_OK:
        return None
    try:
        p = Path(path)
        if not p.exists():
            return None
        img = Image.open(p)
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        return _img_to_b64(img)
    except Exception:
        return None


def load_image_with_bboxes(path: str, bboxes: list, max_size: int = 480) -> str | None:
    """
    加载图片，在上面叠加 bbox 矩形（坐标为 0-999 归一化），返回 base64 data URI。
    每个 bbox 用不同颜色标注，左上角标编号。
    """
    if not PIL_OK or not bboxes:
        return load_image_b64(path, max_size)
    try:
        p = Path(path)
        if not p.exists():
            return None
        img = Image.open(p).convert("RGB")
        orig_w, orig_h = img.size

        img.thumbnail((max_size, max_size), Image.LANCZOS)
        new_w, new_h = img.size
        sx = new_w / orig_w
        sy = new_h / orig_h

        draw = ImageDraw.Draw(img)

        for i, bbox in enumerate(bboxes):
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                continue
            x1, y1, x2, y2 = bbox
            # 归一化坐标 → 像素
            px1 = int(x1 / 999 * orig_w * sx)
            py1 = int(y1 / 999 * orig_h * sy)
            px2 = int(x2 / 999 * orig_w * sx)
            py2 = int(y2 / 999 * orig_h * sy)
            color = BBOX_COLORS[i % len(BBOX_COLORS)]
            # 矩形框（画两遍：黑色描边 + 彩色线）
            draw.rectangle([px1 - 1, py1 - 1, px2 + 1, py2 + 1], outline="black", width=3)
            draw.rectangle([px1, py1, px2, py2], outline=color, width=2)
            # 编号标签
            label = str(i + 1)
            lx, ly = max(px1 + 2, 1), max(py1 + 2, 1)
            draw.rectangle([lx - 1, ly - 1, lx + 10, ly + 12], fill="black")
            draw.text((lx, ly), label, fill=color)

        return _img_to_b64(img)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────────────────

def load_folder(folder: Path) -> dict | None:
    episodes_file = folder / "episodes.json"
    metrics_file  = folder / "metrics.json"
    config_file   = folder / "config.json"

    if not episodes_file.exists():
        return None
    try:
        with open(episodes_file, "r", encoding="utf-8") as f:
            episodes = json.load(f)
    except Exception as ex:
        print(f"[warn] 无法读取 {episodes_file}: {ex}", file=sys.stderr)
        return None

    metrics = {}
    if metrics_file.exists():
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            pass

    config = {}
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            pass

    return {"name": folder.name, "path": str(folder),
            "metrics": metrics, "episodes": episodes, "config": config}


def discover_folders(base_dir: Path) -> list[Path]:
    return sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and (d / "episodes.json").exists()],
        key=lambda p: p.name,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HTML 辅助
# ─────────────────────────────────────────────────────────────────────────────

def e(s) -> str:
    return html_module.escape(str(s))

def truncate(s: str, n: int = 200) -> str:
    s = str(s)
    return s[:n] + "…" if len(s) > n else s


def _parse_tool_call(action: dict | str) -> tuple[str, dict | None]:
    """返回 (action_type, tool_call_dict_or_None)"""
    if isinstance(action, dict):
        atype = action.get("type", "")
        tc = action.get("tool_call", "")
        if isinstance(tc, str):
            try:
                tc = json.loads(tc)
            except Exception:
                return atype, None
        if isinstance(tc, dict):
            return atype, tc
        return atype, None
    return str(action), None


def render_step(step_idx: int, step: dict,
                image_path: str | None, embed_images: bool, img_size: int) -> str:
    model_response = str(step.get("model_response", ""))
    action = step.get("action", {})
    observation = str(step.get("observation", ""))
    reward = step.get("reward", 0)

    action_type, tc_dict = _parse_tool_call(action)

    # 解析思考链
    think_content = response_content = ""
    if "<think>" in model_response:
        s = model_response.find("<think>")
        en = model_response.find("</think>")
        if s != -1 and en != -1:
            think_content    = model_response[s + 7 : en]
            response_content = model_response[en + 8 :].strip()
        else:
            think_content = model_response[s + 7:]
    else:
        response_content = model_response

    badge_cls    = "badge-tool" if action_type == "tool_call" else "badge-answer"
    reward_cls   = "reward-pos" if reward > 0 else ""
    uid          = f"s{step_idx}_{id(step) & 0xFFFFFF}"

    # ── bbox 图片 ──────────────────────────────────────
    bbox_img_html = ""
    if embed_images and image_path and tc_dict:
        tool_name = tc_dict.get("name", "")
        if tool_name == "crop_and_search":
            args = tc_dict.get("arguments", {})
            bboxes = args.get("bbox", [])
            goal   = args.get("goal", "")
            if bboxes:
                b64 = load_image_with_bboxes(image_path, bboxes, img_size)
                if b64:
                    # 颜色图例
                    legend_items = "".join(
                        f'<span class="bbox-legend-item" style="border-color:{BBOX_COLORS[i % len(BBOX_COLORS)]}">'
                        f'<span style="color:{BBOX_COLORS[i % len(BBOX_COLORS)]};font-weight:600">#{i+1}</span> '
                        f'[{bx[0]},{bx[1]},{bx[2]},{bx[3]}]</span>'
                        for i, bx in enumerate(bboxes) if isinstance(bx, (list, tuple)) and len(bx) == 4
                    )
                    bbox_img_html = f"""
    <div class="block bbox-block">
      <div class="block-label">🖼️ 裁剪区域（bbox 可视化）</div>
      {f'<div class="bbox-goal">目标：{e(goal)}</div>' if goal else ''}
      <div class="bbox-legend">{legend_items}</div>
      <img class="step-img" src="{b64}" alt="bbox" loading="lazy">
    </div>"""

    # ── 工具调用 JSON ──────────────────────────────────
    tool_call_str = ""
    if tc_dict:
        tool_call_str = json.dumps(tc_dict, ensure_ascii=False, indent=2)
    elif isinstance(action, dict):
        raw_tc = action.get("tool_call", "")
        if raw_tc:
            tool_call_str = raw_tc if isinstance(raw_tc, str) else json.dumps(raw_tc, ensure_ascii=False, indent=2)

    parts = [f"""
<div class="step-card">
  <div class="step-header" onclick="toggleEl('{uid}',this)">
    <span class="step-num">Step {step_idx + 1}</span>
    <span class="step-badge {badge_cls}">{e(action_type) or 'N/A'}</span>
    <span class="step-reward {reward_cls}">reward: {reward}</span>
    <span class="toggle-icon">▼</span>
  </div>
  <div class="collapsible" id="{uid}">"""]

    if think_content:
        parts.append(f"""
    <div class="block think-block">
      <div class="block-label">🧠 思考过程</div>
      <div class="block-content">{e(think_content)}</div>
    </div>""")

    if response_content:
        parts.append(f"""
    <div class="block response-block">
      <div class="block-label">💬 模型回答</div>
      <div class="block-content">{e(response_content)}</div>
    </div>""")

    if tool_call_str:
        parts.append(f"""
    <div class="block action-block">
      <div class="block-label">🔧 工具调用</div>
      <pre class="block-content code-pre">{e(tool_call_str)}</pre>
    </div>""")

    if bbox_img_html:
        parts.append(bbox_img_html)

    if observation:
        parts.append(f"""
    <div class="block obs-block">
      <div class="block-label">👁️ 观察结果</div>
      <div class="block-content">{e(observation)}</div>
    </div>""")

    parts.append("  </div>\n</div>")
    return "".join(parts)


def render_episode(ep_idx: int, ep: dict, folder_name: str,
                   embed_images: bool, img_size: int) -> str:
    task        = ep.get("task", {})
    question    = task.get("question", "")
    answer      = task.get("answer", "")
    images      = task.get("images", [])
    is_correct  = ep.get("is_correct", False)
    termination = ep.get("termination_reason", "")
    info        = ep.get("info", {})
    prediction  = str(info.get("prediction", ""))
    rounds      = info.get("rounds", 0)
    time_taken  = info.get("time_taken", 0)
    token_usage = info.get("token_usage", {})

    trajectories = ep.get("trajectories", [])
    steps = []
    for traj in trajectories:
        steps.extend(traj.get("steps", []))

    # 提取图片路径（取第一张）
    image_path = images[0] if images else None

    # 提取 image_id / clean question
    image_id = ""
    clean_question = question
    if "image_id:" in question:
        for part in question.split():
            if part.startswith("image_id:"):
                image_id = part.replace("image_id:", "")
                break
        clean_question = question.replace(f"image_id:{image_id}", "").strip()
        if clean_question.lower().startswith("question:"):
            clean_question = clean_question[9:].strip()

    correct_class = "correct" if is_correct else "incorrect"
    correct_icon  = "✅" if is_correct else "❌"
    uid = f"ep_{folder_name}_{ep_idx}"
    pred_class = "pred-correct" if is_correct else "pred-incorrect"

    token_info = (
        f"Prompt: {token_usage.get('prompt','—')} | "
        f"Completion: {token_usage.get('completion','—')} | "
        f"Max: {token_usage.get('max_prompt','—')}"
    )

    # ── 原图 ────────────────────────────────────────────
    orig_img_html = ""
    if embed_images and image_path:
        b64 = load_image_b64(image_path, img_size)
        if b64:
            orig_img_html = f"""
      <div class="meta-item img-item" style="grid-column: 1 / -1">
        <div class="meta-label">🖼️ 原始图片</div>
        <img class="ep-img" src="{b64}" alt="{e(image_id)}" loading="lazy">
      </div>"""

    steps_html = "\n".join(
        render_step(i, s, image_path, embed_images, img_size)
        for i, s in enumerate(steps)
    )

    return f"""
<div class="episode-card {correct_class}" data-correct="{str(is_correct).lower()}">
  <div class="episode-header" onclick="toggleEp('{uid}')">
    <div class="ep-left">
      <span class="ep-num">#{ep_idx + 1}</span>
      <span>{correct_icon}</span>
      <span class="ep-question">{e(truncate(clean_question, 100))}</span>
    </div>
    <div class="ep-right">
      <span class="ep-tag">{e(termination)}</span>
      <span class="ep-meta">{rounds}轮</span>
      <span class="ep-meta">{time_taken:.1f}s</span>
      <span class="ep-toggle" id="icon_{uid}">▶</span>
    </div>
  </div>
  <div class="episode-body" id="{uid}" style="display:none">
    <div class="meta-grid">
      {orig_img_html}
      <div class="meta-item">
        <div class="meta-label">问题</div>
        <div class="meta-value">{e(clean_question)}</div>
      </div>
      <div class="meta-item">
        <div class="meta-label">图片 ID</div>
        <div class="meta-value">{e(image_id) if image_id else '—'}</div>
      </div>
      <div class="meta-item answer-item">
        <div class="meta-label">✅ 正确答案</div>
        <div class="meta-value answer-val">{e(answer)}</div>
      </div>
      <div class="meta-item {pred_class}">
        <div class="meta-label">{correct_icon} 模型预测</div>
        <div class="meta-value">{e(truncate(prediction, 600))}</div>
      </div>
      <div class="meta-item">
        <div class="meta-label">Token 用量</div>
        <div class="meta-value">{token_info}</div>
      </div>
    </div>
    <div class="steps-section">
      <div class="steps-title">🗂️ 推理轨迹（{len(steps)} 步）</div>
      {steps_html}
    </div>
  </div>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# CSS / JS
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", sans-serif;
       background: #0f1117; color: #e2e8f0; min-height: 100vh; }

/* ── Header ── */
.header { background: linear-gradient(135deg,#1a1f2e,#16213e);
          padding: 24px 32px; border-bottom: 1px solid #2d3748; }
.header h1 { font-size: 26px; font-weight: 700; color: #90cdf4; margin-bottom: 4px; }
.header p  { color: #718096; font-size: 13px; }

/* ── Summary ── */
.summary-grid { display: grid; grid-template-columns: repeat(auto-fill,minmax(220px,1fr));
                gap: 16px; padding: 20px 32px; }
.summary-card { background: #1a202c; border: 1px solid #2d3748; border-radius: 10px;
                padding: 18px; position: relative; overflow: hidden; }
.summary-card::before { content:''; position:absolute; top:0; left:0; right:0; height:3px;
                         background: linear-gradient(90deg,#4299e1,#9f7aea); }
.summary-card h3 { font-size: 13px; color: #718096; margin-bottom: 10px;
                   text-transform: uppercase; letter-spacing: .5px; }
.summary-stats { display: flex; gap: 16px; flex-wrap: wrap; }
.stat-item { display: flex; flex-direction: column; gap: 2px; }
.stat-label { font-size: 11px; color: #4a5568; }
.stat-value { font-size: 20px; font-weight: 700; }
.sv-blue { color: #90cdf4; } .sv-green { color: #68d391; } .sv-red { color: #fc8181; }
.acc-bar { margin-top: 10px; background: #2d3748; border-radius: 999px; height: 5px; }
.acc-fill { height: 100%; background: linear-gradient(90deg,#4299e1,#68d391);
             border-radius: 999px; }
.term-info { margin-top: 8px; font-size: 11px; color: #718096; }

/* ── Tabs ── */
.tabs { display: flex; padding: 0 32px; border-bottom: 1px solid #2d3748;
        background: #13172b; overflow-x: auto; }
.tab { padding: 11px 22px; cursor: pointer; font-size: 14px; font-weight: 500;
       color: #718096; border-bottom: 3px solid transparent; white-space: nowrap; transition: all .2s; }
.tab:hover { color: #90cdf4; }
.tab.active { color: #90cdf4; border-bottom-color: #4299e1; background: rgba(66,153,225,.06); }

/* ── Tab Content ── */
.tab-content { display: none; padding: 20px 32px; }
.tab-content.active { display: block; }

/* ── Filter Bar ── */
.filter-bar { display: flex; gap: 10px; margin-bottom: 16px; align-items: center; flex-wrap: wrap; }
.filter-btn { padding: 5px 14px; border-radius: 20px; border: 1px solid #4a5568;
              background: transparent; color: #a0aec0; cursor: pointer; font-size: 13px; transition: all .2s; }
.filter-btn:hover { border-color: #4299e1; color: #90cdf4; }
.filter-btn.active { background: #2b6cb0; border-color: #4299e1; color: #bee3f8; }
.search-box { padding: 7px 12px; background: #1a202c; border: 1px solid #4a5568;
              border-radius: 8px; color: #e2e8f0; font-size: 13px; width: 260px; outline: none; }
.search-box:focus { border-color: #4299e1; }
.count-label { font-size: 13px; color: #718096; margin-left: auto; }

/* ── Episode Card ── */
.episode-card { background: #1a202c; border: 1px solid #2d3748; border-radius: 10px;
                margin-bottom: 10px; overflow: hidden; transition: border-color .2s; }
.episode-card:hover { border-color: #4a5568; }
.episode-card.correct   { border-left: 3px solid #68d391; }
.episode-card.incorrect { border-left: 3px solid #fc8181; }

.episode-header { display: flex; align-items: center; justify-content: space-between;
                  padding: 13px 16px; cursor: pointer; gap: 10px; }
.episode-header:hover { background: rgba(255,255,255,.02); }
.ep-left  { display: flex; align-items: center; gap: 8px; flex: 1; min-width: 0; }
.ep-right { display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
.ep-num      { font-size: 12px; color: #4a5568; min-width: 26px; }
.ep-question { font-size: 14px; color: #cbd5e0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ep-tag  { font-size: 11px; padding: 2px 7px; border-radius: 4px; background: #2d3748; color: #a0aec0; }
.ep-meta { font-size: 12px; color: #718096; }
.ep-toggle { font-size: 12px; color: #718096; }
.episode-body { padding: 0 16px 16px; border-top: 1px solid #2d3748; }

/* ── Meta Grid ── */
.meta-grid { display: grid; grid-template-columns: repeat(auto-fill,minmax(280px,1fr));
             gap: 10px; margin: 14px 0; }
.meta-item { background: #131928; border: 1px solid #2d3748; border-radius: 8px; padding: 11px; }
.meta-item.img-item  { background: #0e131f; border-color: #3a4a6a; text-align: center; }
.meta-item.answer-item  { border-color: #276749; background: #0d2018; }
.meta-item.pred-correct { border-color: #276749; background: #0d2018; }
.meta-item.pred-incorrect { border-color: #744210; background: #1a140a; }
.meta-label { font-size: 11px; color: #718096; margin-bottom: 6px;
              text-transform: uppercase; letter-spacing: .5px; }
.meta-value { font-size: 13px; color: #e2e8f0; line-height: 1.55; }
.answer-val { font-weight: 600; color: #68d391; }

/* ── Images ── */
.ep-img {
  max-width: 100%; max-height: 380px; border-radius: 6px;
  border: 1px solid #2d3748; object-fit: contain;
  cursor: zoom-in; transition: transform .15s;
}
.ep-img:hover { transform: scale(1.01); }
.step-img {
  max-width: 100%; max-height: 340px; border-radius: 6px;
  border: 1px solid #3a4a6a; object-fit: contain; margin-top: 8px;
  cursor: zoom-in;
}

/* ── Lightbox ── */
#lightbox { display:none; position:fixed; inset:0; background:rgba(0,0,0,.88);
            z-index:9999; align-items:center; justify-content:center; cursor:zoom-out; }
#lightbox.open { display:flex; }
#lightbox img { max-width:92vw; max-height:92vh; border-radius:8px;
                box-shadow:0 8px 48px rgba(0,0,0,.6); object-fit:contain; }

/* ── Steps ── */
.steps-section { margin-top: 14px; }
.steps-title { font-size: 13px; color: #718096; margin-bottom: 8px;
               padding-bottom: 7px; border-bottom: 1px solid #2d3748; }
.step-card { background: #131928; border: 1px solid #2d3748; border-radius: 8px;
             margin-bottom: 7px; overflow: hidden; }
.step-header { display: flex; align-items: center; gap: 9px; padding: 9px 13px; cursor: pointer; }
.step-header:hover { background: rgba(255,255,255,.02); }
.step-num   { font-size: 12px; font-weight: 600; color: #90cdf4; min-width: 48px; }
.step-badge { font-size: 11px; padding: 2px 7px; border-radius: 4px; }
.badge-tool   { background: #2c3e6e; color: #90cdf4; }
.badge-answer { background: #1a4a2e; color: #68d391; }
.step-reward  { font-size: 12px; color: #718096; margin-left: auto; }
.reward-pos   { color: #68d391 !important; }
.toggle-icon  { font-size: 11px; color: #4a5568; }
.collapsible  { display: none; padding: 0 13px 13px; }

/* ── Blocks ── */
.block { border-radius: 0 6px 6px 0; padding: 9px 11px; margin: 8px 0; }
.block-label { font-size: 11px; font-weight: 600; color: #718096; margin-bottom: 5px;
               text-transform: uppercase; letter-spacing: .5px; }
.block-content { font-size: 12px; line-height: 1.6; white-space: pre-wrap;
                 max-height: 280px; overflow-y: auto; }
.think-block    { background: #1c1030; border-left: 3px solid #9f7aea; }
.think-block .block-content { color: #b794f4; }
.response-block { background: #0d1f0d; border-left: 3px solid #68d391; }
.response-block .block-content { color: #c6f6d5; }
.action-block   { background: #0d1a2e; border-left: 3px solid #4299e1; }
.action-block .block-content { color: #bee3f8; }
.obs-block      { background: #1a1a0d; border-left: 3px solid #ecc94b; }
.obs-block .block-content { color: #faf089; }
.bbox-block     { background: #0e1a2e; border-left: 3px solid #63b3ed; }
.bbox-goal      { font-size: 12px; color: #90cdf4; margin-bottom: 6px; font-style: italic; }
.bbox-legend    { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
.bbox-legend-item { font-size: 11px; color: #a0aec0; padding: 2px 8px;
                    border: 1px solid; border-radius: 4px; background: rgba(0,0,0,.3); }
.code-pre { font-family: "JetBrains Mono","Fira Code",monospace; font-size: 11.5px; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #1a202c; }
::-webkit-scrollbar-thumb { background: #4a5568; border-radius: 3px; }
"""

JS = """
// ── Lightbox ──────────────────────────────────────────────────────────
const lb = document.getElementById('lightbox');
const lbImg = document.getElementById('lb-img');
document.querySelectorAll('.ep-img, .step-img').forEach(img => {
  img.addEventListener('click', ev => {
    ev.stopPropagation();
    lbImg.src = img.src;
    lb.classList.add('open');
  });
});
lb.addEventListener('click', () => lb.classList.remove('open'));
document.addEventListener('keydown', ev => { if (ev.key === 'Escape') lb.classList.remove('open'); });

// ── Tabs ──────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.getElementById('tab_' + name).classList.add('active');
  document.getElementById('content_' + name).classList.add('active');
}

// ── Expand / Collapse ─────────────────────────────────────────────────
function toggleEp(id) {
  const body = document.getElementById(id);
  const icon = document.getElementById('icon_' + id);
  const open = body.style.display === 'block';
  body.style.display = open ? 'none' : 'block';
  icon.textContent = open ? '▶' : '▼';
}

function toggleEl(id, hdr) {
  const el   = document.getElementById(id);
  const icon = hdr.querySelector('.toggle-icon');
  const open = el.style.display === 'block';
  el.style.display = open ? 'none' : 'block';
  if (icon) icon.textContent = open ? '▼' : '▲';
}

// ── Filter / Search ───────────────────────────────────────────────────
function applyFilters(folder) {
  const filterVal = document.querySelector('[data-folder="' + folder + '"] .filter-btn.active')?.dataset.filter || 'all';
  const searchVal = (document.getElementById('search_' + folder)?.value || '').toLowerCase();
  const container = document.getElementById('episodes_' + folder);
  let count = 0;
  container.querySelectorAll('.episode-card').forEach(card => {
    const ok = (filterVal === 'all' || card.dataset.correct === filterVal)
            && (!searchVal || card.textContent.toLowerCase().includes(searchVal));
    card.style.display = ok ? '' : 'none';
    if (ok) count++;
  });
  document.getElementById('count_' + folder).textContent = count + ' 条';
}

function setFilter(folder, filter, btn) {
  btn.closest('.filter-bar').querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  applyFilters(folder);
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# 完整 HTML 页面
# ─────────────────────────────────────────────────────────────────────────────

def build_html(datasets: list[dict], title: str,
               filter_mode: str, embed_images: bool, img_size: int) -> str:
    parts = []

    parts.append(f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{e(title)}</title>
<style>{CSS}</style>
</head>
<body>
<!-- Lightbox -->
<div id="lightbox"><img id="lb-img" src="" alt=""></div>
""")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    img_note = f" · 图片嵌入：{'✅ 含 bbox 可视化' if embed_images else '❌ 已禁用（--no-images）'}"
    parts.append(f"""
<div class="header">
  <h1>🔬 {e(title)}</h1>
  <p>生成时间：{now} · 共 {len(datasets)} 个实验组{img_note}</p>
</div>
""")

    # ── Summary ──
    parts.append('<div class="summary-grid">')
    for ds in datasets:
        m       = ds["metrics"]
        total   = m.get("total",    len(ds["episodes"]))
        correct = m.get("correct",  sum(1 for ep in ds["episodes"] if ep.get("is_correct")))
        acc     = m.get("accuracy", correct / total if total else 0) * 100
        term    = m.get("termination_distribution", {})
        term_str= "  |  ".join(f"{k}: {v}" for k, v in term.items())
        acc_cls = "sv-green" if acc > 20 else "sv-red"
        parts.append(f"""
  <div class="summary-card">
    <h3>{e(ds['name'])}</h3>
    <div class="summary-stats">
      <div class="stat-item"><span class="stat-label">总数</span>
        <span class="stat-value sv-blue">{total}</span></div>
      <div class="stat-item"><span class="stat-label">正确</span>
        <span class="stat-value sv-green">{correct}</span></div>
      <div class="stat-item"><span class="stat-label">准确率</span>
        <span class="stat-value {acc_cls}">{acc:.1f}%</span></div>
    </div>
    <div class="acc-bar"><div class="acc-fill" style="width:{acc:.1f}%"></div></div>
    <div class="term-info">{term_str}</div>
  </div>""")
    parts.append("</div>")

    # ── Tabs ──
    parts.append('<div class="tabs">')
    for i, ds in enumerate(datasets):
        active = "active" if i == 0 else ""
        parts.append(f'<div class="tab {active}" id="tab_{e(ds["name"])}" '
                     f'onclick="switchTab(\'{e(ds["name"])}\')">'
                     f'{e(ds["name"])} ({len(ds["episodes"])})</div>')
    parts.append("</div>")

    # ── Tab Contents ──
    for i, ds in enumerate(datasets):
        name   = ds["name"]
        active = "active" if i == 0 else ""
        episodes = ds["episodes"]
        if filter_mode == "correct":
            episodes = [ep for ep in episodes if ep.get("is_correct")]
        elif filter_mode == "incorrect":
            episodes = [ep for ep in episodes if not ep.get("is_correct")]

        parts.append(f'<div class="tab-content {active}" id="content_{e(name)}">')
        parts.append(f"""
  <div class="filter-bar" data-folder="{e(name)}">
    <button class="filter-btn active" data-filter="all"
      onclick="setFilter('{e(name)}','all',this)">全部</button>
    <button class="filter-btn" data-filter="true"
      onclick="setFilter('{e(name)}','true',this)">✅ 正确</button>
    <button class="filter-btn" data-filter="false"
      onclick="setFilter('{e(name)}','false',this)">❌ 错误</button>
    <input class="search-box" id="search_{e(name)}" type="text"
      placeholder="搜索问题 / 答案 / 预测..."
      oninput="applyFilters('{e(name)}')">
    <span class="count-label" id="count_{e(name)}">{len(episodes)} 条</span>
  </div>
  <div id="episodes_{e(name)}">""")
        for idx, ep in enumerate(episodes):
            parts.append(render_episode(idx, ep, name, embed_images, img_size))
        parts.append("  </div>\n</div>")

    parts.append(f"""
<script>{JS}</script>
</body>
</html>""")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VDR-Bench Episodes 可视化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dirs", nargs="*", metavar="DIR",
        help="要可视化的输出目录（可多个）。不填则自动发现 --base 下所有子目录。")
    parser.add_argument("--base", default=None, metavar="BASE_DIR",
        help="自动发现模式的根目录，默认为脚本同级的 outputs/ 目录。")
    parser.add_argument("--output", "-o", default=None, metavar="FILE",
        help="输出 HTML 文件路径（默认：<base>/episodes_viz.html）")
    parser.add_argument("--title", default="VDR-Bench Episodes 可视化",
        help="报告标题")
    parser.add_argument("--filter", choices=["all", "correct", "incorrect"], default="all",
        help="只输出正确/错误/全部的 episodes（默认 all）")
    parser.add_argument("--no-images", action="store_true",
        help="不嵌入图片（生成更快、文件更小）")
    parser.add_argument("--img-size", type=int, default=480, metavar="PX",
        help="缩略图最长边像素（默认 480）")
    args = parser.parse_args()

    embed_images = not args.no_images
    if embed_images and not PIL_OK:
        print("[warn] 未找到 Pillow，自动禁用图片嵌入。可执行：pip install Pillow", file=sys.stderr)
        embed_images = False

    script_dir = Path(__file__).parent
    base_dir   = Path(args.base) if args.base else (script_dir.parent.parent / "data" / "eval_outputs")
    if not base_dir.exists():
        print(f"[error] 根目录不存在: {base_dir}", file=sys.stderr)
        sys.exit(1)

    if args.dirs:
        target_folders = [Path(d) for d in args.dirs]
    else:
        target_folders = discover_folders(base_dir)
        if not target_folders:
            print(f"[error] 在 {base_dir} 下未找到任何包含 episodes.json 的子目录。", file=sys.stderr)
            sys.exit(1)

    datasets = []
    for folder in target_folders:
        folder = Path(folder)
        if not folder.is_absolute():
            for base in [Path.cwd(), script_dir]:
                candidate = base / folder
                if candidate.exists():
                    folder = candidate
                    break
        print(f"[load] {folder} ...", end=" ", flush=True)
        ds = load_folder(folder)
        if ds:
            print(f"{len(ds['episodes'])} episodes")
            datasets.append(ds)
        else:
            print("跳过（无有效 episodes.json）")

    if not datasets:
        print("[error] 没有有效数据可以可视化。", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else (script_dir.parent.parent / "data" / "eval_analysis" / "episodes_viz.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[build] 正在生成 HTML（{len(datasets)} 组 | 过滤: {args.filter}"
          f" | 图片: {'嵌入 ' + str(args.img_size) + 'px' if embed_images else '禁用'}）...")
    html_content = build_html(datasets, args.title, args.filter, embed_images, args.img_size)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"[done]  {output_path}  ({size_mb:.1f} MB)")
    print(f"        用浏览器打开即可查看。")


if __name__ == "__main__":
    main()
