#!/usr/bin/env python3
"""
Vision DeepResearch 全面实验分析脚本
比较 0条 vs 21+条 反向图像检索结果 对模型性能的影响

覆盖维度：
  - 准确率 / masked 率
  - 难度分层（base / hard1 / hard）
  - 类别分析（Nature / Architecture / Movie 等）
  - 终止原因（正常答题 / 50轮未果 / 重复检测）
  - Judge vs 精确匹配
  - Token 消耗与效率
  - 工具调用策略
  - 反向图像搜索质量（成功率、错误率）
  - 答案线索命中分析（何时命中、命中但答错）
  - 输出 token 截断分析
  - 幻觉分析（快速但错误）
  - 逐样本明细
"""

import json
import re
import os
import statistics
from collections import defaultdict

# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_image_id(question: str):
    """从 question 字符串解析 image_id，再解析难度和类别"""
    m = re.search(r'image_id:([\w&]+)', question)
    if not m:
        return 'unknown', 'unknown', 'unknown'
    img_id = m.group(1)
    # 格式: base_Nature_880 / hard1_Architecture_872_1 / hard_Object_657_3
    parts = img_id.split('_')
    if parts[0].startswith('hard'):
        difficulty = parts[0]         # hard / hard1
        category   = parts[1] if len(parts) > 1 else 'unknown'
    else:
        difficulty = parts[0]         # base
        category   = parts[1] if len(parts) > 1 else 'unknown'
    return img_id, difficulty, category

def collect_all_steps(ep):
    steps = []
    for traj in ep.get('trajectories', []):
        steps.extend(traj.get('steps', []))
    return steps

def count_tool_calls_by_name(steps):
    """返回各工具调用次数 dict"""
    counts = defaultdict(int)
    for step in steps:
        action = step.get('action', {})
        if action.get('type') == 'tool_call':
            try:
                tc = json.loads(action.get('tool_call', '{}'))
                counts[tc.get('name', 'unknown')] += 1
            except Exception:
                pass
    return dict(counts)

def count_bboxes_per_crop_call(steps):
    """crop_and_search 每次调用的 bbox 数量之和"""
    total_bboxes = 0
    calls = 0
    for step in steps:
        action = step.get('action', {})
        if action.get('type') == 'tool_call':
            try:
                tc = json.loads(action.get('tool_call', '{}'))
                if tc.get('name') == 'crop_and_search':
                    bboxes = tc.get('arguments', {}).get('bbox', [])
                    total_bboxes += len(bboxes)
                    calls += 1
            except Exception:
                pass
    return total_bboxes, calls

def count_search_queries(steps):
    """web search 每次调用的 query 数量之和"""
    total_queries = 0
    calls = 0
    for step in steps:
        action = step.get('action', {})
        if action.get('type') == 'tool_call':
            try:
                tc = json.loads(action.get('tool_call', '{}'))
                if tc.get('name') == 'search':
                    queries = tc.get('arguments', {}).get('query', [])
                    total_queries += len(queries)
                    calls += 1
            except Exception:
                pass
    return total_queries, calls

def extract_obs_stats(steps):
    """从所有 observation 中统计：有效结果数、错误结果数"""
    valid, error = 0, 0
    for step in steps:
        obs = step.get('observation', '') or ''
        valid += len(re.findall(r'The useful information in \[', obs))
        error += len(re.findall(r'\[Error\]', obs))
    return valid, error

def first_round_clue_found(steps, answer):
    """返回第一次出现答案线索的轮次（0-based），-1 表示未出现"""
    answer_lower = answer.strip().lower()
    if not answer_lower:
        return -1
    for i, step in enumerate(steps):
        obs = step.get('observation', '') or ''
        if answer_lower in obs.lower():
            return i
    return -1

def has_output_truncation(steps):
    """completion token == max_prompt 时认为输出被截断（实际看 4096 限制）"""
    for step in steps:
        for cc in step.get('chat_completions', []):
            content = cc.get('content', '') or ''
            # 如果 model_response 长度接近 4096 token（近似以字符数判断）
            if len(content) > 12000:  # ~3 token/char
                return True
    return False

def is_output_token_maxed(ep):
    """completion token 等于上限 4096"""
    tok = ep.get('info', {}).get('token_usage', {}).get('completion', 0)
    return tok >= 4096

def compute_thinking_tokens(steps):
    """统计 <think>...</think> 段落字符数（近似 token 消耗）"""
    total_chars = 0
    for step in steps:
        for cc in step.get('chat_completions', []):
            content = cc.get('content', '') or ''
            thinks = re.findall(r'<think>(.*?)</think>', content, re.DOTALL)
            for t in thinks:
                total_chars += len(t)
    return total_chars

def classify_failure(is_correct, masked, mask_reason, prediction, answer, judge_judgment, had_clue, rounds):
    """
    将每个失败样本归类为一种失败类型，返回 (类型标签, 简短描述)

    分类体系：
    ─ Masked 类（模型没给出有效答案）
      M1: 50轮未找到答案     — mask_reason 含 "50 rounds"
      M2: 重复检测截断       — mask_reason 含 "repetition"
      M3: 连续步骤错误       — mask_reason 含 "consecutive"
      M4: 服务器/系统错误    — mask_reason 含 "error" / prediction 含 "call_server"
    ─ 答错类（给出了答案但是错的）
      W1: 幻觉/实体识别错误  — 识别出来的是完全不同的实体（花/人/地点等）
      W2: 细节错误           — 识别了对的主体但具体细节错（数字/年份/名称变体）
      W3: 命中线索仍答错     — had_clue=True 但给错了
      W4: 快速幻觉           — ≤3轮就给出但错了
    """
    if not is_correct:
        if masked:
            mr = (mask_reason or '').lower()
            pred_lower = (prediction or '').lower()
            if '50 rounds' in mr or '50 rounds' in pred_lower:
                return 'M1: 50轮未找到答案'
            elif 'repetition' in mr:
                return 'M2: 重复检测截断'
            elif 'consecutive' in mr:
                return 'M3: 连续步骤错误'
            elif 'error' in mr or 'call_server' in pred_lower or 'failed' in pred_lower:
                return 'M4: 系统/服务器错误'
            else:
                return 'M5: 其他Masked'
        else:
            # 答错类：按优先级判断
            if rounds <= 3:
                return 'W4: 快速幻觉(≤3轮)'
            elif had_clue:
                return 'W3: 命中线索仍答错'
            else:
                # 尝试从 judge_judgment 判断是否是"识别对象完全不同"
                j = (judge_judgment or '').lower()
                pred_lower = (prediction or '').lower()
                # 细节错误信号：数字/年份/日期类答案（答案里有数字）
                if re.search(r'\d{3,}', answer or ''):
                    return 'W2: 细节/数字错误'
                # judge 提到了不同的名字/实体
                if 'identifies' in j or 'describes' in j or 'different' in j or 'wrong' in j:
                    return 'W1: 实体识别错误'
                return 'W2: 细节/内容错误'
    return 'correct'


def extract_per_episode_data(ep):
    """提取单个 episode 的所有分析指标"""
    info          = ep.get('info', {})
    task          = ep.get('task', {})
    question      = task.get('question', '')
    answer        = task.get('answer', '')
    image_path    = (task.get('images') or [''])[0]

    img_id, difficulty, category = parse_image_id(question)

    steps         = collect_all_steps(ep)
    tool_counts   = count_tool_calls_by_name(steps)
    total_bboxes, crop_calls = count_bboxes_per_crop_call(steps)
    total_queries, search_calls = count_search_queries(steps)
    valid_res, error_res = extract_obs_stats(steps)
    clue_round    = first_round_clue_found(steps, answer)

    reward_meta   = info.get('reward_metadata', {})
    mask_reason   = info.get('mask_reason', '')
    termination   = ep.get('termination_reason', '')
    prediction    = info.get('prediction', '')
    judge_judgment = reward_meta.get('judgment', '')

    tok            = info.get('token_usage', {})
    prompt_tok     = tok.get('prompt', 0)
    completion_tok = tok.get('completion', 0)
    rounds         = info.get('rounds', 0)

    think_chars    = compute_thinking_tokens(steps)

    is_correct     = ep.get('is_correct', False)
    masked         = ep.get('metrics', {}).get('masked', 0) == 1.0
    judge_used     = reward_meta.get('judge_used', False)
    exact_match    = reward_meta.get('exact_match', False)
    judge_decided  = reward_meta.get('judge_decided', False)

    had_clue_but_failed = (clue_round >= 0) and (not is_correct)

    failure_type = classify_failure(
        is_correct, masked, mask_reason, prediction, answer,
        judge_judgment, clue_round >= 0, rounds
    )

    return {
        # 基本信息
        'id':                    ep.get('id', ''),
        'image_id':              img_id,
        'difficulty':            difficulty,
        'category':              category,
        'question':              question[:70],
        'answer':                answer,
        # 结果
        'is_correct':            is_correct,
        'masked':                masked,
        'mask_reason':           mask_reason,
        'termination_reason':    termination,
        'judge_used':            judge_used,
        'exact_match':           exact_match,
        'judge_decided':         judge_decided,
        'failure_type':          failure_type,
        'prediction':            prediction[:120],
        'judge_judgment':        judge_judgment[:200],
        # 资源消耗
        'rounds':                rounds,
        'time_taken':            round(info.get('time_taken', 0), 1),
        'prompt_tokens':         prompt_tok,
        'completion_tokens':     completion_tok,
        'total_tokens':          prompt_tok + completion_tok,
        'tokens_per_round':      round((prompt_tok + completion_tok) / max(rounds, 1), 0),
        'output_tok_maxed':      completion_tok >= 4096,
        'thinking_chars':        think_chars,
        # 工具调用
        'crop_calls':            tool_counts.get('crop_and_search', 0),
        'search_calls':          tool_counts.get('search', 0),
        'visit_calls':           tool_counts.get('visit', 0),
        'total_bboxes':          total_bboxes,
        'total_search_queries':  total_queries,
        # 搜索结果质量
        'valid_results':         valid_res,
        'error_results':         error_res,
        # 仅在实际有搜索结果时计算成功率；从未搜索则为 None（不计入均值）
        'search_success_rate':   round(valid_res / (valid_res + error_res), 3) if (valid_res + error_res) > 0 else None,
        # 答案线索
        'clue_round':            clue_round,   # -1=未命中
        'had_clue':              clue_round >= 0,
        'had_clue_but_failed':   had_clue_but_failed,
        # 幻觉分析：快速给答案（<=3轮）但错误
        'fast_wrong':            (rounds <= 3) and (not is_correct) and (not masked),
    }

# ─────────────────────────────────────────────
# 打印工具
# ─────────────────────────────────────────────

def mean(lst):
    return round(statistics.mean(lst), 2) if lst else 0

def med(lst):
    return round(statistics.median(lst), 2) if lst else 0

def pct(n, total):
    return f"{n}/{total} ({n/total*100:.1f}%)" if total else "0/0"

def print_section(title):
    print(f"\n{'─'*64}")
    print(f"  {title}")
    print(f"{'─'*64}")

def print_kv_table(rows, col1=28, col2=14, col3=14, header=None):
    if header:
        print(f"  {header[0]:<{col1}}  {header[1]:>{col2}}  {header[2]:>{col3}}")
        print(f"  {'─'*col1}  {'─'*col2}  {'─'*col3}")
    for row in rows:
        print(f"  {row[0]:<{col1}}  {str(row[1]):>{col2}}  {str(row[2]):>{col3}}")

# ─────────────────────────────────────────────
# 单组分析
# ─────────────────────────────────────────────

def analyze_group(episodes, group_name):
    data = [extract_per_episode_data(ep) for ep in episodes]
    N = len(data)

    print(f"\n{'═'*64}")
    print(f"  组别: {group_name}   (N={N})")
    print(f"{'═'*64}")

    # ── 1. 准确率总览 ──────────────────────────
    print_section("1. 准确率总览")
    correct   = [d for d in data if d['is_correct']]
    masked    = [d for d in data if d['masked']]
    judge_ok  = [d for d in data if d['is_correct'] and d['judge_used']]
    exact_ok  = [d for d in data if d['is_correct'] and d['exact_match']]
    tok_maxed = [d for d in data if d['output_tok_maxed']]

    print(f"  正确率         : {pct(len(correct), N)}")
    print(f"  Masked 率      : {pct(len(masked), N)}")
    print(f"  其中 Judge 判对: {pct(len(judge_ok), len(correct))}")
    print(f"  其中 精确匹配  : {pct(len(exact_ok), len(correct))}")
    print(f"  输出token截断  : {pct(len(tok_maxed), N)}  (completion≥4096)")

    # ── 2. 终止原因分类 ────────────────────────
    print_section("2. 终止 / Masked 原因")
    reason_counts = defaultdict(int)
    for d in data:
        r = d['mask_reason'] if d['masked'] else ('correct' if d['is_correct'] else 'wrong_answer')
        reason_counts[r] += 1
    for r, c in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {r:<40} : {pct(c, N)}")

    # ── 3. 难度分层 ────────────────────────────
    print_section("3. 难度分层分析")
    diffs = sorted(set(d['difficulty'] for d in data))
    print_kv_table(
        [(diff,
          pct(sum(1 for d in data if d['difficulty']==diff and d['is_correct']),
              sum(1 for d in data if d['difficulty']==diff)),
          f"N={sum(1 for d in data if d['difficulty']==diff)}")
         for diff in diffs],
        header=("难度", "正确率", "样本数")
    )

    # ── 4. 类别分层 ────────────────────────────
    print_section("4. 类别分层分析")
    cats = sorted(set(d['category'] for d in data))
    print_kv_table(
        [(cat,
          pct(sum(1 for d in data if d['category']==cat and d['is_correct']),
              sum(1 for d in data if d['category']==cat)),
          f"N={sum(1 for d in data if d['category']==cat)}")
         for cat in cats],
        header=("类别", "正确率", "样本数")
    )

    # ── 5. 资源消耗 ────────────────────────────
    print_section("5. 资源消耗统计")
    def stat_row(label, vals):
        return f"  {label:<26}  avg={mean(vals):>8}  med={med(vals):>8}  min={min(vals):>8}  max={max(vals):>8}"

    print(stat_row("推理轮次",          [d['rounds'] for d in data]))
    print(stat_row("耗时(秒)",          [d['time_taken'] for d in data]))
    print(stat_row("Prompt Token",      [d['prompt_tokens'] for d in data]))
    print(stat_row("Completion Token",  [d['completion_tokens'] for d in data]))
    print(stat_row("Total Token",       [d['total_tokens'] for d in data]))
    print(stat_row("Token/轮次",        [d['tokens_per_round'] for d in data]))
    print(stat_row("思考字符数(think)", [d['thinking_chars'] for d in data]))

    # 正确 vs 错误的资源消耗对比
    c_data = [d for d in data if d['is_correct'] and not d['masked']]
    w_data = [d for d in data if not d['is_correct'] and not d['masked']]
    if c_data and w_data:
        print(f"\n  ┌─ 正确答题(非masked) vs 错误答题(非masked) ─────────────┐")
        for key, label in [('rounds','推理轮次'), ('total_tokens','总Token')]:
            cv = [d[key] for d in c_data]
            wv = [d[key] for d in w_data]
            print(f"  │  {label:<14}  正确: avg={mean(cv):>7}  错误: avg={mean(wv):>7}  │")
        print(f"  └────────────────────────────────────────────────────────┘")

    # ── 6. 工具调用策略 ────────────────────────
    print_section("6. 工具调用策略")
    print(f"  {'指标':<28}  {'均值':>8}  {'中位数':>8}  {'最大值':>8}")
    print(f"  {'─'*28}  {'─'*8}  {'─'*8}  {'─'*8}")
    for key, label in [
        ('crop_calls',           'crop_and_search 调用次数'),
        ('total_bboxes',         '  └─ 总 bbox 数'),
        ('search_calls',         'web search 调用次数'),
        ('total_search_queries', '  └─ 总 query 数'),
        ('visit_calls',          'visit 调用次数'),
    ]:
        vals = [d[key] for d in data]
        print(f"  {label:<28}  {mean(vals):>8}  {med(vals):>8}  {max(vals):>8}")

    # 工具使用比例
    total_calls = sum(d['crop_calls']+d['search_calls']+d['visit_calls'] for d in data)
    crop_total  = sum(d['crop_calls'] for d in data)
    srch_total  = sum(d['search_calls'] for d in data)
    visit_total = sum(d['visit_calls'] for d in data)
    if total_calls:
        print(f"\n  工具占比: crop_and_search={crop_total/total_calls*100:.1f}%  "
              f"search={srch_total/total_calls*100:.1f}%  visit={visit_total/total_calls*100:.1f}%")

    # ── 7. 反向图像搜索质量 ────────────────────
    print_section("7. 反向图像搜索结果质量")
    vr = [d['valid_results'] for d in data]
    er = [d['error_results'] for d in data]
    sr = [d['search_success_rate'] for d in data if d['search_success_rate'] is not None]
    print(f"  有效结果条数  avg={mean(vr)}  med={med(vr)}  min={min(vr)}  max={max(vr)}")
    print(f"  错误结果条数  avg={mean(er)}  med={med(er)}  min={min(er)}  max={max(er)}")
    sr_note = f"(基于{len(sr)}个有搜索的episode)" if len(sr) < len(data) else ""
    print(f"  搜索成功率    avg={mean(sr):.3f}  med={med(sr):.3f}  {sr_note}")

    # ── 8. 答案线索命中分析 ────────────────────
    print_section("8. 答案线索命中分析（observation 中出现正确答案）")
    had_clue   = [d for d in data if d['had_clue']]
    no_clue    = [d for d in data if not d['had_clue']]
    clue_fail  = [d for d in data if d['had_clue_but_failed']]
    clue_succ  = [d for d in data if d['had_clue'] and d['is_correct']]

    print(f"  曾命中答案线索   : {pct(len(had_clue), N)}")
    print(f"  从未命中答案线索 : {pct(len(no_clue), N)}")
    print(f"  命中线索且答对   : {pct(len(clue_succ), len(had_clue) if had_clue else 1)}")
    print(f"  命中线索但答错   : {pct(len(clue_fail), len(had_clue) if had_clue else 1)}")

    if had_clue:
        clue_rounds = [d['clue_round'] for d in had_clue]
        print(f"\n  首次命中线索在第几轮: avg={mean(clue_rounds)}  med={med(clue_rounds)}  "
              f"min={min(clue_rounds)}  max={max(clue_rounds)}")

    # 命中线索但答错的详情
    if clue_fail:
        print(f"\n  ⚠️  命中线索但最终答错的样本:")
        for d in clue_fail:
            print(f"     [{d['difficulty']}/{d['category']}] 第{d['clue_round']}轮命中  "
                  f"rounds={d['rounds']}  masked={d['masked']}")
            print(f"       Q: {d['question'][:60]}")
            print(f"       A: {d['answer']}")

    # ── 9. 幻觉分析（快速错误） ────────────────
    print_section("9. 幻觉分析（≤3轮即给出答案但错误、且未被masked）")
    fast_wrong = [d for d in data if d['fast_wrong']]
    print(f"  快速错误样本数: {pct(len(fast_wrong), N)}")
    for d in fast_wrong:
        print(f"    [{d['difficulty']}/{d['category']}] rounds={d['rounds']}  "
              f"Q: {d['question'][:55]}")
        print(f"      正确答案: {d['answer']}")

    # ── 10. 逐样本明细 ────────────────────────
    print_section("10. 逐样本完整明细")
    hdr = (f"{'#':>2} {'Diff':<6} {'Cat':<14} {'OK':>5} {'Mask':>5} "
           f"{'Rd':>4} {'Time':>6} {'PTok':>7} {'CTok':>5} "
           f"{'CS':>3} {'WS':>3} {'V':>3} {'VldR':>5} {'ErrR':>4} "
           f"{'Clue':>5} {'Round':>5} {'TMx':>4}")
    print(hdr)
    print("─" * len(hdr))
    for i, d in enumerate(data):
        print(f"{i+1:>2} {d['difficulty']:<6} {d['category']:<14} "
              f"{str(d['is_correct']):>5} {str(d['masked']):>5} "
              f"{d['rounds']:>4} {d['time_taken']:>6.1f} "
              f"{d['prompt_tokens']:>7} {d['completion_tokens']:>5} "
              f"{d['crop_calls']:>3} {d['search_calls']:>3} {d['visit_calls']:>3} "
              f"{d['valid_results']:>5} {d['error_results']:>4} "
              f"{str(d['had_clue']):>5} {d['clue_round']:>5} "
              f"{str(d['output_tok_maxed']):>4}")
        print(f"   Q: {d['question'][:75]}")

    return data

# ─────────────────────────────────────────────
# 两组对比
# ─────────────────────────────────────────────

def compare_groups(d0, d21, d69=None, name0="0条", name21="21+条", name69="6-9条"):

    groups = [(name0, d0), (name21, d21)]
    if d69 is not None:
        groups = [(name0, d0), (name69, d69), (name21, d21)]

    print(f"\n\n{'═'*64}")
    print(f"  ★ 多组横向对比分析  ({' / '.join(n for n,_ in groups)})")
    print(f"{'═'*64}")

    N0, N21 = len(d0), len(d21)
    N69 = len(d69) if d69 is not None else 0

    # ── A. 核心指标对比 ────────────────────────
    print_section("A. 核心准确率 & 质量指标对比")
    def pct_val(lst, key_fn):
        return round(sum(1 for d in lst if key_fn(d)) / len(lst) * 100, 1) if lst else 0

    metric_keys = [
        ("正确率(%)",           lambda d: d['is_correct']),
        ("Masked率(%)",         lambda d: d['masked']),
        ("命中答案线索率(%)",   lambda d: d['had_clue']),
        ("命中线索但答错率(%)", lambda d: d['had_clue_but_failed']),
        ("输出Token截断率(%)",  lambda d: d['output_tok_maxed']),
        ("快速错误率(%)",       lambda d: d['fast_wrong']),
        ("Judge判对率(%)",      lambda d: d['is_correct'] and d['judge_used']),
        ("精确匹配正确率(%)",   lambda d: d['exact_match'] and d['is_correct']),
    ]

    col_names = [n for n, _ in groups]
    hdr = f"  {'指标':<28}" + "".join(f"  {n:>10}" for n in col_names)
    print(hdr)
    print(f"  {'─'*28}" + ("  " + "─"*10) * len(groups))
    for label, fn in metric_keys:
        vals = [pct_val(data, fn) for _, data in groups]
        row = f"  {label:<28}" + "".join(f"  {v:>10}" for v in vals)
        print(row)

    # ── B. 资源效率对比 ────────────────────────
    print_section("B. 资源消耗效率对比")
    res_metrics = [
        ('rounds',              '推理轮次'),
        ('time_taken',          '耗时(秒)'),
        ('prompt_tokens',       'Prompt Token'),
        ('completion_tokens',   'Completion Token'),
        ('total_tokens',        'Total Token'),
        ('tokens_per_round',    'Token/轮次'),
        ('thinking_chars',      '思考字符数'),
        ('crop_calls',          'crop_and_search次数'),
        ('total_bboxes',        '  总bbox数'),
        ('search_calls',        'web search次数'),
        ('total_search_queries','  总query数'),
        ('visit_calls',         'visit次数'),
        ('valid_results',       '有效搜索结果条数'),
        ('error_results',       '失败搜索结果条数'),
        ('search_success_rate', '搜索成功率'),
    ]
    hdr2 = f"  {'指标':<28}" + "".join(f"  {n:>10}" for n in col_names)
    print(hdr2)
    print(f"  {'─'*28}" + ("  " + "─"*10) * len(groups))
    for key, label in res_metrics:
        vals = [mean([d[key] for d in data if d[key] is not None]) for _, data in groups]
        row = f"  {label:<28}" + "".join(f"  {v:>10}" for v in vals)
        print(row)

    # ── C. 难度分层对比 ────────────────────────
    print_section("C. 难度分层对比")
    all_data_flat = sum((data for _, data in groups), [])
    all_diffs = sorted(set(d['difficulty'] for d in all_data_flat))
    hdr3 = f"  {'难度':<8}" + "".join(f"  {n+' N':>8}  {n+' 正确率':>10}" for n in col_names)
    print(hdr3)
    print(f"  {'─'*8}" + ("  " + "─"*8 + "  " + "─"*10) * len(groups))
    for diff in all_diffs:
        row = f"  {diff:<8}"
        for _, data in groups:
            sub = [d for d in data if d['difficulty'] == diff]
            acc = str(pct_val(sub, lambda d: d['is_correct'])) + '%' if sub else 'N/A'
            row += f"  {len(sub):>8}  {acc:>10}"
        print(row)

    # ── D. 类别对比 ────────────────────────────
    print_section("D. 类别对比")
    all_cats = sorted(set(d['category'] for d in all_data_flat))
    hdr4 = f"  {'类别':<14}" + "".join(f"  {n+' 正确率':>12}" for n in col_names)
    print(hdr4)
    print(f"  {'─'*14}" + ("  " + "─"*12) * len(groups))
    for cat in all_cats:
        row = f"  {cat:<14}"
        for _, data in groups:
            sub = [d for d in data if d['category'] == cat]
            acc = str(pct_val(sub, lambda d: d['is_correct'])) + '%' if sub else 'N/A'
            row += f"  {acc:>12}"
        print(row)

    # ── E. Mask 原因对比 ───────────────────────
    print_section("E. Masked 原因对比")
    def mask_reasons(data):
        d = defaultdict(int)
        for ep in data:
            if ep['masked']:
                d[ep['mask_reason'] or 'unknown'] += 1
        return d

    reason_dicts = [(n, mask_reasons(data)) for n, data in groups]
    all_reasons = sorted(set(r for _, rd in reason_dicts for r in rd.keys()))
    hdr5 = f"  {'Mask原因':<38}" + "".join(f"  {n:>8}" for n, _ in reason_dicts)
    print(hdr5)
    print(f"  {'─'*38}" + ("  " + "─"*8) * len(groups))
    for r in all_reasons:
        row = f"  {r:<38}" + "".join(f"  {rd.get(r,0):>8}" for _, rd in reason_dicts)
        print(row)

    # ── F. 综合结论 ────────────────────────────
    print_section("F. 综合结论与研究启示")
    acc0  = pct_val(d0,  lambda d: d['is_correct'])
    acc21 = pct_val(d21, lambda d: d['is_correct'])
    clue0  = pct_val(d0,  lambda d: d['had_clue'])
    clue21 = pct_val(d21, lambda d: d['had_clue'])
    fail0  = pct_val(d0,  lambda d: d['had_clue_but_failed'])
    fail21 = pct_val(d21, lambda d: d['had_clue_but_failed'])

    mid_info = ""
    if d69 is not None:
        acc69  = pct_val(d69, lambda d: d['is_correct'])
        clue69 = pct_val(d69, lambda d: d['had_clue'])
        fail69 = pct_val(d69, lambda d: d['had_clue_but_failed'])
        mid_info = f"\n    - 6-9条组 正确率 {acc69}%  线索命中 {clue69}%  命中但答错 {fail69}%"

    print(f"""
  【多组准确率趋势】
    - 0条组  正确率 {acc0}%{mid_info}
    - 21+条组 正确率 {acc21}%

  【关键发现：线索命中率 vs 转化率】
    - 0条组  搜索命中答案线索: {clue0}%   命中但答错: {fail0}%
    - 21+条组搜索命中答案线索: {clue21}%   命中但答错: {fail21}%
    → 反向检索条数越多，越容易搜到线索，但噪声也更多

  【样本偏差风险】
    - 各组的难度分布（base/hard/hard1）和类别分布可能不同
    - 建议对同难度/同类别的样本进行配对分析

  【反向检索失效模式分析】
    - 0条组大量 crop_and_search 返回 [Error]（平均 {mean([d['error_results'] for d in d0])} 条错误）
    - 但模型通过 web search 补救了部分正确答案
    - 21+条组 Error 明显减少（{mean([d['error_results'] for d in d21])} 条），但有效结果的质量存疑

  【建议后续实验设计】
    1. 在相同图片上对比 0条 / 6-9条 / 21+条，做曲线分析
    2. 分析 21+条组"命中线索但答错"的具体原因（噪声/歧义/模型推理错误）
    3. 考察 masked 样本中是否存在可救的案例
    4. 引入更多指标：首次 crop_and_search 的命中率（第一轮就找到线索的比例）
""")

# ─────────────────────────────────────────────
# HTML 可视化报告生成
# ─────────────────────────────────────────────

def generate_html_report(d0, d21, output_path, d69=None):
    """生成自包含的 HTML 可视化分析报告（支持 2-3 组）"""
    import json as _json

    has_69 = d69 is not None and len(d69) > 0

    def pv(lst, fn):
        return round(sum(1 for d in lst if fn(d)) / len(lst) * 100, 1) if lst else 0

    def avg(lst, key):
        vals = [d[key] for d in lst if d[key] is not None]
        return round(statistics.mean(vals), 2) if vals else 0

    # ── 准备所有图表数据 ──────────────────────
    # 1. 核心准确率
    accuracy_data = {
        'labels': ['正确率', 'Masked率', '命中线索率', '线索命中→答错率', 'Token截断率'],
        'group0': [
            pv(d0,  lambda d: d['is_correct']),
            pv(d0,  lambda d: d['masked']),
            pv(d0,  lambda d: d['had_clue']),
            pv(d0,  lambda d: d['had_clue_but_failed']),
            pv(d0,  lambda d: d['output_tok_maxed']),
        ],
        'group69': [
            pv(d69, lambda d: d['is_correct']) if has_69 else None,
            pv(d69, lambda d: d['masked']) if has_69 else None,
            pv(d69, lambda d: d['had_clue']) if has_69 else None,
            pv(d69, lambda d: d['had_clue_but_failed']) if has_69 else None,
            pv(d69, lambda d: d['output_tok_maxed']) if has_69 else None,
        ],
        'group21': [
            pv(d21, lambda d: d['is_correct']),
            pv(d21, lambda d: d['masked']),
            pv(d21, lambda d: d['had_clue']),
            pv(d21, lambda d: d['had_clue_but_failed']),
            pv(d21, lambda d: d['output_tok_maxed']),
        ],
    }

    # 2. Masked 原因
    def mask_reason_counts(data):
        c = defaultdict(int)
        for d in data:
            if d['masked']:
                c[d['mask_reason'] or 'unknown'] += 1
        return dict(c)
    mr0  = mask_reason_counts(d0)
    mr21 = mask_reason_counts(d21)
    mr69 = mask_reason_counts(d69) if has_69 else {}

    # 3. 难度分层
    diffs = ['base', 'hard1', 'hard']
    diff_data = {
        'labels': diffs,
        'group0':  [pv([x for x in d0  if x['difficulty']==dif], lambda d: d['is_correct']) for dif in diffs],
        'group69': [pv([x for x in d69 if x['difficulty']==dif], lambda d: d['is_correct']) if has_69 else None for dif in diffs],
        'group21': [pv([x for x in d21 if x['difficulty']==dif], lambda d: d['is_correct']) for dif in diffs],
        'n0':      [sum(1 for x in d0  if x['difficulty']==dif) for dif in diffs],
        'n69':     [sum(1 for x in d69 if x['difficulty']==dif) if has_69 else 0 for dif in diffs],
        'n21':     [sum(1 for x in d21 if x['difficulty']==dif) for dif in diffs],
    }

    # 4. 类别分析
    all_src = d0 + (d69 if has_69 else []) + d21
    all_cats = sorted(set(d['category'] for d in all_src))
    cat_data = {
        'labels':  all_cats,
        'group0':  [pv([x for x in d0  if x['category']==c], lambda d: d['is_correct']) for c in all_cats],
        'group69': [pv([x for x in d69 if x['category']==c], lambda d: d['is_correct']) if has_69 else None for c in all_cats],
        'group21': [pv([x for x in d21 if x['category']==c], lambda d: d['is_correct']) for c in all_cats],
    }

    # 5. 资源消耗
    resource_data = {
        'labels': ['推理轮次', 'Prompt Token(K)', 'Compl Token(K)', '耗时(秒)', 'crop调用数', 'search调用数', 'visit调用数'],
        'group0': [
            avg(d0, 'rounds'),
            round(avg(d0, 'prompt_tokens')/1000, 1),
            round(avg(d0, 'completion_tokens')/1000, 1),
            avg(d0, 'time_taken'),
            avg(d0, 'crop_calls'),
            avg(d0, 'search_calls'),
            avg(d0, 'visit_calls'),
        ],
        'group69': [
            avg(d69, 'rounds') if has_69 else None,
            round(avg(d69, 'prompt_tokens')/1000, 1) if has_69 else None,
            round(avg(d69, 'completion_tokens')/1000, 1) if has_69 else None,
            avg(d69, 'time_taken') if has_69 else None,
            avg(d69, 'crop_calls') if has_69 else None,
            avg(d69, 'search_calls') if has_69 else None,
            avg(d69, 'visit_calls') if has_69 else None,
        ],
        'group21': [
            avg(d21, 'rounds'),
            round(avg(d21, 'prompt_tokens')/1000, 1),
            round(avg(d21, 'completion_tokens')/1000, 1),
            avg(d21, 'time_taken'),
            avg(d21, 'crop_calls'),
            avg(d21, 'search_calls'),
            avg(d21, 'visit_calls'),
        ],
    }

    # 6. 工具调用占比
    def tool_totals(data):
        return {
            'crop':  sum(d['crop_calls'] for d in data),
            'search':sum(d['search_calls'] for d in data),
            'visit': sum(d['visit_calls'] for d in data),
        }
    tt0  = tool_totals(d0)
    tt21 = tool_totals(d21)
    tt69 = tool_totals(d69) if has_69 else {'crop':0,'search':0,'visit':0}

    # 7. 搜索质量
    search_quality = {
        'labels': ['有效结果条数(均值)', '失败结果条数(均值)', '搜索成功率(%)'],
        'group0':  [avg(d0, 'valid_results'), avg(d0, 'error_results'), round(avg(d0, 'search_success_rate')*100, 1)],
        'group69': [avg(d69,'valid_results') if has_69 else None, avg(d69,'error_results') if has_69 else None, round(avg(d69,'search_success_rate')*100,1) if has_69 else None],
        'group21': [avg(d21,'valid_results'), avg(d21,'error_results'), round(avg(d21,'search_success_rate')*100,1)],
    }

    # 8. 散点图：轮次 vs 总Token
    scatter0  = [{'x': d['rounds'], 'y': round(d['total_tokens']/1000, 1),
                  'correct': d['is_correct'], 'masked': d['masked'],
                  'label': d['image_id']} for d in d0]
    scatter21 = [{'x': d['rounds'], 'y': round(d['total_tokens']/1000, 1),
                  'correct': d['is_correct'], 'masked': d['masked'],
                  'label': d['image_id']} for d in d21]
    scatter69 = [{'x': d['rounds'], 'y': round(d['total_tokens']/1000, 1),
                  'correct': d['is_correct'], 'masked': d['masked'],
                  'label': d['image_id']} for d in (d69 if has_69 else [])]

    # 9. Token 堆叠
    token_stack = {
        'labels0':  [d['image_id'].replace('_', '_\n') for d in d0],
        'labels69': [d['image_id'].replace('_', '_\n') for d in (d69 if has_69 else [])],
        'labels21': [d['image_id'].replace('_', '_\n') for d in d21],
        'prompt0':  [round(d['prompt_tokens']/1000, 1) for d in d0],
        'compl0':   [round(d['completion_tokens']/1000, 1) for d in d0],
        'prompt69': [round(d['prompt_tokens']/1000, 1) for d in (d69 if has_69 else [])],
        'compl69':  [round(d['completion_tokens']/1000, 1) for d in (d69 if has_69 else [])],
        'prompt21': [round(d['prompt_tokens']/1000, 1) for d in d21],
        'compl21':  [round(d['completion_tokens']/1000, 1) for d in d21],
        'correct0':  [d['is_correct'] for d in d0],
        'correct69': [d['is_correct'] for d in (d69 if has_69 else [])],
        'correct21': [d['is_correct'] for d in d21],
    }

    # KPI 直接计算值
    n0, n21 = len(d0), len(d21)
    n69 = len(d69) if has_69 else 0
    acc0  = pv(d0, lambda d: d['is_correct'])
    acc21 = pv(d21, lambda d: d['is_correct'])
    acc69 = pv(d69, lambda d: d['is_correct']) if has_69 else None
    mask0  = pv(d0, lambda d: d['masked'])
    mask21 = pv(d21, lambda d: d['masked'])
    mask69 = pv(d69, lambda d: d['masked']) if has_69 else None
    clue0  = pv(d0, lambda d: d['had_clue'])
    clue21 = pv(d21, lambda d: d['had_clue'])
    clue69 = pv(d69, lambda d: d['had_clue']) if has_69 else None
    # 线索命中→答对转化率
    def clue_to_correct(data):
        clued = [d for d in data if d['had_clue']]
        return round(sum(1 for d in clued if d['is_correct']) / len(clued) * 100, 1) if clued else 0
    conv0  = clue_to_correct(d0)
    conv21 = clue_to_correct(d21)
    conv69 = clue_to_correct(d69) if has_69 else None
    rounds0  = avg(d0, 'rounds')
    rounds21 = avg(d21, 'rounds')
    rounds69 = avg(d69, 'rounds') if has_69 else None
    tok0  = round(avg(d0, 'total_tokens')/1000, 1)
    tok21 = round(avg(d21, 'total_tokens')/1000, 1)
    tok69 = round(avg(d69, 'total_tokens')/1000, 1) if has_69 else None
    sr0  = round(avg(d0, 'search_success_rate')*100, 1)
    sr21 = round(avg(d21, 'search_success_rate')*100, 1)
    sr69 = round(avg(d69, 'search_success_rate')*100, 1) if has_69 else None
    err0  = avg(d0, 'error_results')
    err21 = avg(d21, 'error_results')
    err69 = avg(d69, 'error_results') if has_69 else None

    def kpi_69_str(val, label, fmt=lambda v: str(v)):
        if not has_69 or val is None:
            return ''
        return f'<div><div class="kpi-val v69">{fmt(val)}</div><div class="kpi-sub">{label}</div></div>'

    # 10. 逐样本明细表格数据
    def table_rows(data, group_name):
        rows = []
        for d in data:
            rows.append({
                'group': group_name,
                'image_id': d['image_id'],
                'difficulty': d['difficulty'],
                'category': d['category'],
                'is_correct': d['is_correct'],
                'masked': d['masked'],
                'mask_reason': d['mask_reason'],
                'rounds': d['rounds'],
                'time': d['time_taken'],
                'prompt_tok': d['prompt_tokens'],
                'compl_tok': d['completion_tokens'],
                'crop': d['crop_calls'],
                'search': d['search_calls'],
                'visit': d['visit_calls'],
                'valid_r': d['valid_results'],
                'error_r': d['error_results'],
                'had_clue': d['had_clue'],
                'clue_round': d['clue_round'],
                'tok_maxed': d['output_tok_maxed'],
                'question': d['question'],
                'answer': d['answer'],
                'failure_type': d['failure_type'],
                'prediction': d['prediction'],
                'judge_judgment': d['judge_judgment'],
            })
        return rows
    all_rows = table_rows(d0, '0条') + (table_rows(d69, '6-9条') if has_69 else []) + table_rows(d21, '21+条')

    # 11. 失败原因分类数据
    FAILURE_TYPES = ['M1: 50轮未找到答案', 'M2: 重复检测截断', 'M3: 连续步骤错误',
                     'M4: 系统/服务器错误', 'M5: 其他Masked',
                     'W1: 实体识别错误', 'W2: 细节/内容错误', 'W2: 细节/数字错误',
                     'W3: 命中线索仍答错', 'W4: 快速幻觉(≤3轮)']
    def failure_counts(data):
        c = defaultdict(int)
        for d in data:
            if not d['is_correct']:
                c[d['failure_type']] += 1
        return dict(c)

    fc0  = failure_counts(d0)
    fc21 = failure_counts(d21)
    fc69 = failure_counts(d69) if has_69 else {}

    all_fail_types = sorted(set(list(fc0.keys()) + list(fc21.keys()) + list(fc69.keys())))
    failure_data = {
        'labels': all_fail_types,
        'group0':  [fc0.get(t, 0) for t in all_fail_types],
        'group69': [fc69.get(t, 0) for t in all_fail_types] if has_69 else None,
        'group21': [fc21.get(t, 0) for t in all_fail_types],
    }

    # 失败样本明细（仅失败的）
    fail_rows = [r for r in all_rows if not r['is_correct']]

    # ── 构建 HTML ─────────────────────────────
    data_js = f"""
const HAS_69 = {'true' if has_69 else 'false'};
const accuracyData = {_json.dumps(accuracy_data)};
const maskReasons0 = {_json.dumps(mr0)};
const maskReasons69 = {_json.dumps(mr69)};
const maskReasons21 = {_json.dumps(mr21)};
const diffData = {_json.dumps(diff_data)};
const catData = {_json.dumps(cat_data)};
const resourceData = {_json.dumps(resource_data)};
const toolTotals0 = {_json.dumps(tt0)};
const toolTotals69 = {_json.dumps(tt69)};
const toolTotals21 = {_json.dumps(tt21)};
const searchQuality = {_json.dumps(search_quality)};
const scatter0 = {_json.dumps(scatter0)};
const scatter69 = {_json.dumps(scatter69)};
const scatter21 = {_json.dumps(scatter21)};
const tokenStack = {_json.dumps(token_stack)};
const tableRows = {_json.dumps(all_rows)};
const failureData = {_json.dumps(failure_data)};
const failRows = {_json.dumps(fail_rows)};
"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Vision DeepResearch — 反向检索条数影响分析</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2.2.0/dist/chartjs-plugin-datalabels.min.js"></script>
<style>
:root {{
  --c0: #6366f1;   /* 0条 indigo */
  --c69: #22d3ee;  /* 6-9条 cyan */
  --c21: #f59e0b;  /* 21+条 amber */
  --cok: #10b981;
  --cerr: #ef4444;
  --cmask: #8b5cf6;
  --bg: #0f172a;
  --card: #1e293b;
  --border: #334155;
  --text: #e2e8f0;
  --muted: #94a3b8;
  --radius: 14px;
}}
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  background: var(--bg);
  color: var(--text);
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
  font-size: 14px;
  line-height: 1.6;
}}
header {{
  background: linear-gradient(135deg, #1e1b4b 0%, #172554 50%, #0c1445 100%);
  border-bottom: 1px solid var(--border);
  padding: 28px 40px 24px;
}}
header h1 {{ font-size: 24px; font-weight: 700; letter-spacing: -0.5px; }}
header h1 span {{ color: var(--c0); }}
header h1 span.s21 {{ color: var(--c21); }}
header p {{ color: var(--muted); margin-top: 6px; font-size: 13px; }}

.legend-bar {{
  display: flex; gap: 24px; margin-top: 14px; flex-wrap: wrap;
}}
.legend-item {{
  display: flex; align-items: center; gap: 8px; font-size: 13px;
}}
.legend-dot {{
  width: 12px; height: 12px; border-radius: 50%;
}}

main {{ padding: 32px 40px; max-width: 1600px; margin: 0 auto; }}

/* KPI 卡片 */
.kpi-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 16px;
  margin-bottom: 32px;
}}
.kpi-card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px 22px;
  position: relative;
  overflow: hidden;
}}
.kpi-card::before {{
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
}}
.kpi-card.c0::before  {{ background: var(--c0); }}
.kpi-card.c69::before {{ background: var(--c69); }}
.kpi-card.c21::before {{ background: var(--c21); }}
.kpi-card.cok::before  {{ background: var(--cok); }}
.kpi-label {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px; }}
.kpi-vals {{ display: flex; gap: 18px; align-items: flex-end; }}
.kpi-val {{ font-size: 28px; font-weight: 700; line-height: 1; }}
.kpi-val.v0  {{ color: var(--c0); }}
.kpi-val.v69 {{ color: var(--c69); }}
.kpi-val.v21 {{ color: var(--c21); }}
.kpi-sub {{ font-size: 11px; color: var(--muted); margin-top: 4px; }}

/* 图表网格 */
.chart-grid {{
  display: grid;
  gap: 20px;
  margin-bottom: 28px;
}}
.grid-2 {{ grid-template-columns: 1fr 1fr; }}
.grid-3 {{ grid-template-columns: 1fr 1fr 1fr; }}
.grid-1 {{ grid-template-columns: 1fr; }}
@media (max-width: 900px) {{
  .grid-2, .grid-3 {{ grid-template-columns: 1fr; }}
}}

.chart-card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 22px 24px;
}}
.chart-card h3 {{
  font-size: 13px;
  font-weight: 600;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.6px;
  margin-bottom: 18px;
  display: flex;
  align-items: center;
  gap: 8px;
}}
.chart-card h3 .badge {{
  font-size: 10px;
  background: #334155;
  color: #94a3b8;
  padding: 2px 8px;
  border-radius: 99px;
  text-transform: none;
  letter-spacing: 0;
}}
.chart-wrap {{
  position: relative;
  width: 100%;
}}

/* 分割线标题 */
.section-title {{
  font-size: 16px;
  font-weight: 700;
  color: var(--text);
  margin: 36px 0 16px;
  display: flex;
  align-items: center;
  gap: 12px;
}}
.section-title::after {{
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}}

/* 表格 */
.table-wrap {{
  overflow-x: auto;
  border-radius: var(--radius);
  border: 1px solid var(--border);
}}
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 12.5px;
}}
thead tr {{
  background: #253047;
}}
th {{
  padding: 10px 12px;
  text-align: left;
  color: var(--muted);
  font-weight: 600;
  white-space: nowrap;
  border-bottom: 1px solid var(--border);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}}
td {{
  padding: 9px 12px;
  border-bottom: 1px solid #1e2d40;
  vertical-align: top;
  color: var(--text);
}}
tr:last-child td {{ border-bottom: none; }}
tr:hover td {{ background: rgba(99,102,241,0.06); }}
.tag {{
  display: inline-block;
  padding: 2px 8px;
  border-radius: 99px;
  font-size: 11px;
  font-weight: 600;
}}
.tag-ok    {{ background: rgba(16,185,129,0.18); color: #34d399; }}
.tag-err   {{ background: rgba(239,68,68,0.18);  color: #f87171; }}
.tag-mask  {{ background: rgba(139,92,246,0.18); color: #a78bfa; }}
.tag-c0    {{ background: rgba(99,102,241,0.18); color: #818cf8; }}
.tag-c69   {{ background: rgba(34,211,238,0.18); color: #22d3ee; }}
.tag-c21   {{ background: rgba(245,158,11,0.18); color: #fbbf24; }}
.tag-diff-base  {{ background: rgba(6,182,212,0.15);  color: #22d3ee; }}
.tag-diff-hard1 {{ background: rgba(251,191,36,0.15); color: #fcd34d; }}
.tag-diff-hard  {{ background: rgba(239,68,68,0.15);  color: #fca5a5; }}

/* 搜索 */
.table-controls {{
  display: flex; gap: 12px; margin-bottom: 14px; flex-wrap: wrap; align-items: center;
}}
.table-controls input, .table-controls select {{
  background: var(--card);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 8px 14px;
  border-radius: 8px;
  font-size: 13px;
  outline: none;
}}
.table-controls input:focus, .table-controls select:focus {{
  border-color: var(--c0);
}}
.table-controls label {{ color: var(--muted); font-size: 13px; }}

/* 洞察框 */
.insight-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 16px;
  margin-bottom: 32px;
}}
.insight-card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-left: 4px solid;
  border-radius: var(--radius);
  padding: 18px 20px;
}}
.insight-card.warn {{ border-left-color: #f59e0b; }}
.insight-card.info {{ border-left-color: #6366f1; }}
.insight-card.good {{ border-left-color: #10b981; }}
.insight-card.danger {{ border-left-color: #ef4444; }}
.insight-card h4 {{ font-size: 13px; font-weight: 700; margin-bottom: 8px; }}
.insight-card p  {{ font-size: 12.5px; color: var(--muted); line-height: 1.7; }}
.insight-card strong {{ color: var(--text); }}

footer {{
  text-align: center;
  padding: 24px;
  color: var(--muted);
  font-size: 12px;
  border-top: 1px solid var(--border);
  margin-top: 40px;
}}
</style>
</head>
<body>
<header>
  <h1>🔬 Vision DeepResearch &nbsp;|&nbsp; 反向检索条数影响分析</h1>
  <p>比较图片在网络上的传播程度（反向检索条数）对模型答题准确率的影响</p>
  <div class="legend-bar">
    <div class="legend-item"><div class="legend-dot" style="background:var(--c0)"></div> 0条 反向检索结果组（N={n0}）</div>
    {'<div class="legend-item"><div class="legend-dot" style="background:var(--c69)"></div> 6-9条 反向检索结果组（N=' + str(n69) + '）</div>' if has_69 else ''}
    <div class="legend-item"><div class="legend-dot" style="background:var(--c21)"></div> 21+条 反向检索结果组（N={n21}）</div>
    <div class="legend-item"><div class="legend-dot" style="background:var(--cok)"></div> 答题正确</div>
    <div class="legend-item"><div class="legend-dot" style="background:var(--cerr)"></div> 答题错误</div>
    <div class="legend-item"><div class="legend-dot" style="background:var(--cmask)"></div> Masked（未完成）</div>
  </div>
</header>

<main>

<!-- ══════════════════ KPI 卡片 ══════════════════ -->
<div class="section-title">📊 核心指标一览</div>
<div class="kpi-grid">
  <div class="kpi-card c0">
    <div class="kpi-label">正确率</div>
    <div class="kpi-vals">
      <div><div class="kpi-val v0">{acc0}%</div><div class="kpi-sub">0条组 N={n0}</div></div>
      {kpi_69_str(acc69, f'6-9条 N={n69}', lambda v: f'{v}%')}
      <div><div class="kpi-val v21">{acc21}%</div><div class="kpi-sub">21+条 N={n21}</div></div>
    </div>
  </div>
  <div class="kpi-card cok">
    <div class="kpi-label">Masked率</div>
    <div class="kpi-vals">
      <div><div class="kpi-val v0">{mask0}%</div><div class="kpi-sub">0条</div></div>
      {kpi_69_str(mask69, '6-9条', lambda v: f'{v}%')}
      <div><div class="kpi-val v21">{mask21}%</div><div class="kpi-sub">21+条</div></div>
    </div>
  </div>
  <div class="kpi-card c0">
    <div class="kpi-label">命中答案线索率</div>
    <div class="kpi-vals">
      <div><div class="kpi-val v0">{clue0}%</div><div class="kpi-sub">0条</div></div>
      {kpi_69_str(clue69, '6-9条 ↑', lambda v: f'{v}%')}
      <div><div class="kpi-val v21">{clue21}%</div><div class="kpi-sub">21+条 ↑</div></div>
    </div>
  </div>
  <div class="kpi-card c21">
    <div class="kpi-label">线索命中→答对转化</div>
    <div class="kpi-vals">
      <div><div class="kpi-val v0">{conv0}%</div><div class="kpi-sub">0条</div></div>
      {kpi_69_str(conv69, '6-9条', lambda v: f'{v}%')}
      <div><div class="kpi-val v21">{conv21}%</div><div class="kpi-sub">21+条</div></div>
    </div>
  </div>
  <div class="kpi-card c0">
    <div class="kpi-label">平均推理轮次</div>
    <div class="kpi-vals">
      <div><div class="kpi-val v0">{rounds0}</div><div class="kpi-sub">0条</div></div>
      {kpi_69_str(rounds69, '6-9条')}
      <div><div class="kpi-val v21">{rounds21}</div><div class="kpi-sub">21+条</div></div>
    </div>
  </div>
  <div class="kpi-card c0">
    <div class="kpi-label">平均总Token(K)</div>
    <div class="kpi-vals">
      <div><div class="kpi-val v0">{tok0}</div><div class="kpi-sub">0条</div></div>
      {kpi_69_str(tok69, '6-9条')}
      <div><div class="kpi-val v21">{tok21}</div><div class="kpi-sub">21+条</div></div>
    </div>
  </div>
  <div class="kpi-card c21">
    <div class="kpi-label">搜索成功率</div>
    <div class="kpi-vals">
      <div><div class="kpi-val v0">{sr0}%</div><div class="kpi-sub">0条</div></div>
      {kpi_69_str(sr69, '6-9条', lambda v: f'{v}%')}
      <div><div class="kpi-val v21">{sr21}%</div><div class="kpi-sub">21+条 ↑</div></div>
    </div>
  </div>
  <div class="kpi-card c21">
    <div class="kpi-label">搜索失败条数(均值)</div>
    <div class="kpi-vals">
      <div><div class="kpi-val v0">{err0}</div><div class="kpi-sub">0条</div></div>
      {kpi_69_str(err69, '6-9条')}
      <div><div class="kpi-val v21">{err21}</div><div class="kpi-sub">21+条 ↓</div></div>
    </div>
  </div>
</div>

<!-- ══════════════════ 研究洞察 ══════════════════ -->
<div class="section-title">💡 关键发现与研究洞察</div>
<div class="insight-grid">
  <div class="insight-card warn">
    <h4>⚠️ 准确率趋势</h4>
    <p>0条组正确率 <strong>{acc0}%</strong>{'，6-9条组 <strong>' + str(acc69) + '%</strong>' if has_69 else ''}，21+条组 <strong>{acc21}%</strong>——不同组别正确率差异可能受难度/类别分布影响，存在混淆变量。</p>
  </div>
  <div class="insight-card danger">
    <h4>🔴 Repetition Detected 是最大杀手</h4>
    <p>0条组大量样本因重复检测被强制终止——反向检索全部失败→模型陷入死循环→被截断，而非真正"答错"。</p>
  </div>
  <div class="insight-card info">
    <h4>🎯 线索命中率随检索数增加</h4>
    <p>0条组线索命中率 <strong>{clue0}%</strong>{'，6-9条组 <strong>' + str(clue69) + '%</strong>' if has_69 else ''}，21+条组 <strong>{clue21}%</strong>。更多检索结果有助于找到正确线索。</p>
  </div>
  <div class="insight-card good">
    <h4>✅ 线索→答对转化率</h4>
    <p>0条组转化率 <strong>{conv0}%</strong>{'，6-9条组 <strong>' + str(conv69) + '%</strong>' if has_69 else ''}，21+条组 <strong>{conv21}%</strong>。检索越多噪声也越大，需关注转化质量。</p>
  </div>
  <div class="insight-card warn">
    <h4>📌 样本偏差风险</h4>
    <p>各组类别和难度分布可能不同，不是纯控制变量实验。<strong>建议设计配对实验</strong>：相同图片分别在不同检索条数条件下对比。</p>
  </div>
  <div class="insight-card info">
    <h4>🧠 检索条数越多回答越快</h4>
    <p>0条组平均推理 <strong>{rounds0} 轮</strong>{'，6-9条组 <strong>' + str(rounds69) + ' 轮</strong>' if has_69 else ''}，21+条组 <strong>{rounds21} 轮</strong>。有效搜索线索帮助模型更快收敛。</p>
  </div>
</div>

<!-- ══════════════════ 图表区 ══════════════════ -->
<div class="section-title">📈 多维度可视化对比</div>

<div class="chart-grid grid-2">
  <div class="chart-card">
    <h3>核心指标对比 <span class="badge">百分比</span></h3>
    <div class="chart-wrap" style="height:280px"><canvas id="chartAccuracy"></canvas></div>
  </div>
  <div class="chart-card">
    <h3>终止 / Masked 原因分布 <span class="badge">两组对比</span></h3>
    <div class="chart-wrap" style="height:280px"><canvas id="chartMaskReasons"></canvas></div>
  </div>
</div>

<div class="chart-grid grid-3">
  <div class="chart-card">
    <h3>难度分层正确率 <span class="badge">base/hard1/hard</span></h3>
    <div class="chart-wrap" style="height:240px"><canvas id="chartDiff"></canvas></div>
  </div>
  <div class="chart-card">
    <h3>类别正确率对比</h3>
    <div class="chart-wrap" style="height:240px"><canvas id="chartCat"></canvas></div>
  </div>
  <div class="chart-card">
    <h3>工具调用占比</h3>
    <div class="chart-wrap" style="height:240px"><canvas id="chartTools"></canvas></div>
  </div>
</div>

<div class="chart-grid grid-2">
  <div class="chart-card">
    <h3>资源消耗对比 <span class="badge">雷达图</span></h3>
    <div class="chart-wrap" style="height:320px"><canvas id="chartRadar"></canvas></div>
  </div>
  <div class="chart-card">
    <h3>反向图像搜索质量</h3>
    <div class="chart-wrap" style="height:320px"><canvas id="chartSearch"></canvas></div>
  </div>
</div>

<div class="chart-grid grid-1">
  <div class="chart-card">
    <h3>推理轮次 vs 总Token消耗 散点图 <span class="badge">每个点=一个样本</span></h3>
    <div class="chart-wrap" style="height:340px"><canvas id="chartScatter"></canvas></div>
  </div>
</div>

<div class="chart-grid {'grid-3' if has_69 else 'grid-2'}">
  <div class="chart-card">
    <h3>0条组 — 逐样本Token消耗（Prompt + Completion）</h3>
    <div class="chart-wrap" style="height:300px"><canvas id="chartToken0"></canvas></div>
  </div>
  {'<div class="chart-card"><h3>6-9条组 — 逐样本Token消耗（Prompt + Completion）</h3><div class="chart-wrap" style="height:300px"><canvas id="chartToken69"></canvas></div></div>' if has_69 else ''}
  <div class="chart-card">
    <h3>21+条组 — 逐样本Token消耗（Prompt + Completion）</h3>
    <div class="chart-wrap" style="height:300px"><canvas id="chartToken21"></canvas></div>
  </div>
</div>

<!-- ══════════════════ 失败原因分析 ══════════════════ -->
<div class="section-title">🔍 失败原因深度分析</div>

<div class="chart-grid grid-2">
  <div class="chart-card">
    <h3>失败类型分布 <span class="badge">各组对比（失败样本数）</span></h3>
    <div class="chart-wrap" style="height:320px"><canvas id="chartFailType"></canvas></div>
  </div>
  <div class="chart-card">
    <h3>Masked 原因详细 <span class="badge">各组</span></h3>
    <div class="chart-wrap" style="height:320px"><canvas id="chartMaskedDetail"></canvas></div>
  </div>
</div>

<div class="chart-grid grid-2">
  <div class="chart-card">
    <h3>失败类型分类说明</h3>
    <div style="padding:8px 0; font-size:12.5px; line-height:2;">
      <div><span class="tag" style="background:rgba(139,92,246,.2);color:#a78bfa;min-width:180px;display:inline-block">M1: 50轮未找到答案</span> 穷尽所有推理轮次仍无结论</div>
      <div><span class="tag" style="background:rgba(139,92,246,.2);color:#a78bfa;min-width:180px;display:inline-block">M2: 重复检测截断</span> 模型陷入重复循环被系统强制终止</div>
      <div><span class="tag" style="background:rgba(139,92,246,.2);color:#a78bfa;min-width:180px;display:inline-block">M3: 连续步骤错误</span> 连续发生工具调用错误</div>
      <div><span class="tag" style="background:rgba(139,92,246,.2);color:#a78bfa;min-width:180px;display:inline-block">M4: 系统/服务器错误</span> call_server failed 等底层错误</div>
      <div style="margin-top:8px"><span class="tag" style="background:rgba(239,68,68,.2);color:#f87171;min-width:180px;display:inline-block">W1: 实体识别错误</span> 识别出完全不同的实体（幻觉）</div>
      <div><span class="tag" style="background:rgba(239,68,68,.2);color:#f87171;min-width:180px;display:inline-block">W2: 细节/内容错误</span> 识别了对的主体但具体细节/数字错</div>
      <div><span class="tag" style="background:rgba(239,68,68,.2);color:#f87171;min-width:180px;display:inline-block">W3: 命中线索仍答错</span> 搜索到了正确答案线索但最终仍答错</div>
      <div><span class="tag" style="background:rgba(239,68,68,.2);color:#f87171;min-width:180px;display:inline-block">W4: 快速幻觉(≤3轮)</span> 前3轮即快速给出错误答案</div>
    </div>
  </div>
  <div class="chart-card">
    <h3>答错 vs Masked 比例 <span class="badge">各组</span></h3>
    <div class="chart-wrap" style="height:240px"><canvas id="chartFailRatio"></canvas></div>
  </div>
</div>

<!-- 失败样本明细表 -->
<div class="section-title">❌ 失败样本逐条明细</div>
<div class="table-controls">
  <label>筛选组别：</label>
  <select id="filterFailGroup">
    <option value="">全部</option>
    <option value="0条">0条组</option>
    {'<option value="6-9条">6-9条组</option>' if has_69 else ''}
    <option value="21+条">21+条组</option>
  </select>
  <label>失败类型：</label>
  <select id="filterFailType">
    <option value="">全部</option>
    <option value="M">仅 Masked 类</option>
    <option value="W">仅 答错 类</option>
  </select>
  <input type="text" id="filterFailSearch" placeholder="🔍 搜索 image_id / 答案..." style="flex:1;min-width:200px">
</div>
<div class="table-wrap">
  <table id="failTable">
    <thead>
      <tr>
        <th>组别</th><th>难度</th><th>类别</th><th>Image ID</th>
        <th>失败类型</th><th>推理轮次</th>
        <th>正确答案</th><th>模型预测</th>
        <th>Judge 判断摘要</th>
        <th>线索命中</th><th>Token截断</th>
      </tr>
    </thead>
    <tbody id="failTableBody"></tbody>
  </table>
</div>

<!-- ══════════════════ 样本明细表 ══════════════════ -->
<div class="section-title">📋 逐样本明细表（全部）</div>
<div class="table-controls">
  <label>筛选组别：</label>
  <select id="filterGroup">
    <option value="">全部</option>
    <option value="0条">0条组</option>
    {'<option value="6-9条">6-9条组</option>' if has_69 else ''}
    <option value="21+条">21+条组</option>
  </select>
  <label>筛选难度：</label>
  <select id="filterDiff">
    <option value="">全部</option>
    <option value="base">base</option>
    <option value="hard1">hard1</option>
    <option value="hard">hard</option>
  </select>
  <label>筛选结果：</label>
  <select id="filterResult">
    <option value="">全部</option>
    <option value="correct">正确</option>
    <option value="wrong">错误</option>
    <option value="masked">Masked</option>
  </select>
  <input type="text" id="filterSearch" placeholder="🔍 搜索 image_id / 问题..." style="flex:1;min-width:200px">
</div>
<div class="table-wrap">
  <table id="detailTable">
    <thead>
      <tr>
        <th>组别</th><th>难度</th><th>类别</th><th>Image ID</th>
        <th>结果</th><th>Masked原因</th>
        <th>轮次</th><th>耗时(s)</th><th>Prompt(K)</th><th>Compl(K)</th>
        <th>Crop</th><th>Search</th><th>Visit</th>
        <th>有效R</th><th>失败R</th>
        <th>命中线索</th><th>线索轮次</th><th>Token截断</th>
        <th>问题</th>
      </tr>
    </thead>
    <tbody id="tableBody"></tbody>
  </table>
</div>

</main>
<footer>Vision DeepResearch 实验分析报告 · 自动生成 · {__import__('datetime').date.today()}</footer>

<script>
{data_js}

// ── 全局样式 ─────────────────────────────────
Chart.register(ChartDataLabels);
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = '#1e2d40';
Chart.defaults.font.family = "'Inter', 'Segoe UI', sans-serif";
Chart.defaults.plugins.datalabels.display = false;

const C0 = '#6366f1', C69 = '#22d3ee', C21 = '#f59e0b', COK = '#10b981', CERR = '#ef4444', CMASK = '#8b5cf6';
const C0a = 'rgba(99,102,241,0.15)', C69a = 'rgba(34,211,238,0.15)', C21a = 'rgba(245,158,11,0.15)';

// ── 1. 核心指标对比 ──────────────────────────
{{
  const datasets = [
    {{ label: '0条组', data: accuracyData.group0, backgroundColor: C0+'cc', borderColor: C0, borderWidth: 1.5, borderRadius: 5 }},
  ];
  if (HAS_69) datasets.push({{ label: '6-9条组', data: accuracyData.group69, backgroundColor: C69+'cc', borderColor: C69, borderWidth: 1.5, borderRadius: 5 }});
  datasets.push({{ label: '21+条组', data: accuracyData.group21, backgroundColor: C21+'cc', borderColor: C21, borderWidth: 1.5, borderRadius: 5 }});
  new Chart(document.getElementById('chartAccuracy'), {{
    type: 'bar',
    data: {{ labels: accuracyData.labels, datasets }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ position: 'top' }},
        datalabels: {{
          display: true,
          color: '#e2e8f0',
          font: {{ size: 11, weight: '600' }},
          formatter: v => v != null ? v + '%' : '',
          anchor: 'end', align: 'top',
        }}
      }},
      scales: {{
        y: {{ max: 110, ticks: {{ callback: v => v + '%' }} }},
        x: {{ ticks: {{ maxRotation: 20 }} }}
      }}
    }}
  }});
}}

// ── 2. Masked 原因 ───────────────────────────
{{
  const allMrSrc = [maskReasons0, ...(HAS_69 ? [maskReasons69] : []), maskReasons21];
  const mrLabels = [...new Set(allMrSrc.flatMap(o => Object.keys(o)))];
  const mrDatasets = [
    {{ label: '0条组', data: mrLabels.map(k => maskReasons0[k]||0), backgroundColor: C0+'cc', borderColor: C0, borderWidth: 1.5, borderRadius: 5 }},
  ];
  if (HAS_69) mrDatasets.push({{ label: '6-9条组', data: mrLabels.map(k => maskReasons69[k]||0), backgroundColor: C69+'cc', borderColor: C69, borderWidth: 1.5, borderRadius: 5 }});
  mrDatasets.push({{ label: '21+条组', data: mrLabels.map(k => maskReasons21[k]||0), backgroundColor: C21+'cc', borderColor: C21, borderWidth: 1.5, borderRadius: 5 }});
  new Chart(document.getElementById('chartMaskReasons'), {{
    type: 'bar',
    data: {{ labels: mrLabels, datasets: mrDatasets }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ position: 'top' }},
        datalabels: {{ display: true, color:'#e2e8f0', font:{{size:12,weight:'700'}}, anchor:'end', align:'top' }}
      }},
      scales: {{ y: {{ ticks: {{ stepSize: 1 }} }} }}
    }}
  }});
}}

// ── 3. 难度分层 ──────────────────────────────
{{
  const diffLabels = diffData.labels.map((l,i) => {{
    let s = l + ' (N0=' + diffData.n0[i];
    if (HAS_69) s += ',N69=' + diffData.n69[i];
    s += ',N21=' + diffData.n21[i] + ')';
    return s;
  }});
  const diffDs = [
    {{ label: '0条组正确率', data: diffData.group0, backgroundColor: C0+'cc', borderColor: C0, borderWidth: 1.5, borderRadius: 5 }},
  ];
  if (HAS_69) diffDs.push({{ label: '6-9条正确率', data: diffData.group69, backgroundColor: C69+'cc', borderColor: C69, borderWidth: 1.5, borderRadius: 5 }});
  diffDs.push({{ label: '21+条正确率', data: diffData.group21, backgroundColor: C21+'cc', borderColor: C21, borderWidth: 1.5, borderRadius: 5 }});
  new Chart(document.getElementById('chartDiff'), {{
    type: 'bar',
    data: {{ labels: diffLabels, datasets: diffDs }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ position: 'top' }},
        datalabels: {{ display: true, color:'#e2e8f0', font:{{size:11,weight:'600'}}, formatter: v=>v!=null?v+'%':'', anchor:'end',align:'top' }}
      }},
      scales: {{ y: {{ max: 120, ticks: {{ callback: v => v + '%' }} }} }}
    }}
  }});
}}

// ── 4. 类别正确率 ────────────────────────────
{{
  const catDs = [
    {{ label: '0条组', data: catData.group0, backgroundColor: C0+'cc', borderColor: C0, borderWidth: 1.5, borderRadius: 4 }},
  ];
  if (HAS_69) catDs.push({{ label: '6-9条组', data: catData.group69, backgroundColor: C69+'cc', borderColor: C69, borderWidth: 1.5, borderRadius: 4 }});
  catDs.push({{ label: '21+条组', data: catData.group21, backgroundColor: C21+'cc', borderColor: C21, borderWidth: 1.5, borderRadius: 4 }});
  new Chart(document.getElementById('chartCat'), {{
    type: 'bar',
    data: {{ labels: catData.labels, datasets: catDs }},
    options: {{
      indexAxis: 'y',
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ position: 'top' }},
        datalabels: {{ display: true, color:'#e2e8f0', font:{{size:10}}, formatter: v=>v!=null?v+'%':'', anchor:'end',align:'right' }}
      }},
      scales: {{ x: {{ max: 130, ticks: {{ callback: v => v + '%' }} }} }}
    }}
  }});
}}

// ── 5. 工具调用占比（分组柱状图，展示三组对比）────
{{
  const toolLabels = ['crop_and_search', 'search (web)', 'visit'];
  const toolDs = [
    {{
      label: '0条组',
      data: [toolTotals0.crop, toolTotals0.search, toolTotals0.visit],
      backgroundColor: C0 + 'cc', borderColor: C0, borderWidth: 1.5, borderRadius: 5,
    }},
  ];
  if (HAS_69) toolDs.push({{
    label: '6-9条组',
    data: [toolTotals69.crop, toolTotals69.search, toolTotals69.visit],
    backgroundColor: C69 + 'cc', borderColor: C69, borderWidth: 1.5, borderRadius: 5,
  }});
  toolDs.push({{
    label: '21+条组',
    data: [toolTotals21.crop, toolTotals21.search, toolTotals21.visit],
    backgroundColor: C21 + 'cc', borderColor: C21, borderWidth: 1.5, borderRadius: 5,
  }});
  new Chart(document.getElementById('chartTools'), {{
    type: 'bar',
    data: {{ labels: toolLabels, datasets: toolDs }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ position: 'top' }},
        datalabels: {{
          display: true,
          color: '#e2e8f0',
          font: {{ size: 11, weight: '600' }},
          anchor: 'end', align: 'top',
        }},
      }},
      scales: {{
        y: {{ title: {{ display: true, text: '调用次数(总)', color:'#94a3b8' }}, beginAtZero: true }},
      }},
    }}
  }});
}}

// ── 6. 资源雷达 ──────────────────────────────
{{
  const radarDs = [
    {{ label: '0条组', data: resourceData.group0, fill: true, backgroundColor: C0a, borderColor: C0, pointBackgroundColor: C0, borderWidth: 2 }},
  ];
  if (HAS_69) radarDs.push({{ label: '6-9条组', data: resourceData.group69, fill: true, backgroundColor: C69a, borderColor: C69, pointBackgroundColor: C69, borderWidth: 2 }});
  radarDs.push({{ label: '21+条组', data: resourceData.group21, fill: true, backgroundColor: C21a, borderColor: C21, pointBackgroundColor: C21, borderWidth: 2 }});
  new Chart(document.getElementById('chartRadar'), {{
    type: 'radar',
    data: {{ labels: resourceData.labels, datasets: radarDs }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ position: 'top' }}, datalabels: {{ display: false }} }},
      scales: {{
        r: {{
          grid: {{ color: '#1e2d40' }},
          ticks: {{ display: false }},
          pointLabels: {{ font: {{ size: 11 }}, color: '#94a3b8' }}
        }}
      }}
    }}
  }});
}}

// ── 7. 搜索质量 ──────────────────────────────
{{
  const sqDs = [
    {{ label: '0条组', data: searchQuality.group0, backgroundColor: C0+'cc', borderColor: C0, borderWidth: 1.5, borderRadius: 5 }},
  ];
  if (HAS_69) sqDs.push({{ label: '6-9条组', data: searchQuality.group69, backgroundColor: C69+'cc', borderColor: C69, borderWidth: 1.5, borderRadius: 5 }});
  sqDs.push({{ label: '21+条组', data: searchQuality.group21, backgroundColor: C21+'cc', borderColor: C21, borderWidth: 1.5, borderRadius: 5 }});
  new Chart(document.getElementById('chartSearch'), {{
    type: 'bar',
    data: {{ labels: searchQuality.labels, datasets: sqDs }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ position: 'top' }},
        datalabels: {{ display: true, color:'#e2e8f0', font:{{size:11,weight:'600'}}, anchor:'end',align:'top' }}
      }},
    }}
  }});
}}

// ── 8. 散点图 ────────────────────────────────
function scatterPoint(d, group) {{
  let bc, label;
  if (d.correct) {{ bc = COK; label = '✅'; }}
  else if (d.masked) {{ bc = CMASK; label = '🔲'; }}
  else {{ bc = CERR; label = '❌'; }}
  return {{ x: d.x, y: d.y, label: d.label, group, status: label, backgroundColor: bc + 'cc', borderColor: bc }};
}}
const sp0  = scatter0.map(d => scatterPoint(d, '0条'));
const sp69 = HAS_69 ? scatter69.map(d => scatterPoint(d, '6-9条')) : [];
const sp21 = scatter21.map(d => scatterPoint(d, '21+条'));

const makeScatterDS = (pts, color, label) => ({{
  label, data: pts,
  backgroundColor: pts.map(p => p.backgroundColor),
  borderColor: pts.map(p => p.borderColor),
  pointRadius: 10, pointHoverRadius: 13, borderWidth: 2,
}});

{{
  const scatterDs = [makeScatterDS(sp0, C0, '0条组')];
  if (HAS_69) scatterDs.push(makeScatterDS(sp69, C69, '6-9条组'));
  scatterDs.push(makeScatterDS(sp21, C21, '21+条组'));
  new Chart(document.getElementById('chartScatter'), {{
    type: 'scatter',
    data: {{ datasets: scatterDs }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        datalabels: {{
          display: true,
          color: '#e2e8f0',
          font: {{ size: 9.5 }},
          formatter: (_, ctx) => {{
            const pt = ctx.dataset.data[ctx.dataIndex];
            const sym = pt.group === '0条' ? '◆' : (pt.group === '6-9条' ? '▲' : '●');
            return sym + ' ' + pt.label.replace(/_(base|hard1|hard)_/, '_').slice(-12);
          }},
          anchor: 'end', align: 'top', offset: 4,
        }},
        tooltip: {{
          callbacks: {{
            label: (ctx) => {{
              const pt = ctx.raw;
              return [`${{pt.group}} · ${{pt.label}}`, `轮次: ${{pt.x}}  总Token: ${{pt.y}}K  ${{pt.status}}`];
            }}
          }}
        }}
      }},
      scales: {{
        x: {{ title: {{ display: true, text: '推理轮次', color:'#94a3b8' }}, grid: {{ color:'#1e2d4066' }} }},
        y: {{ title: {{ display: true, text: '总Token (K)', color:'#94a3b8' }}, grid: {{ color:'#1e2d4066' }} }},
      }}
    }}
  }});
}}

// ── 9. Token 堆叠 (0条) ──────────────────────
function makeTokenChart(canvasId, labels, promptVals, complVals, correctArr) {{
  new Chart(document.getElementById(canvasId), {{
    type: 'bar',
    data: {{
      labels: labels,
      datasets: [
        {{
          label: 'Prompt Token(K)',
          data: promptVals,
          backgroundColor: labels.map((_,i) => correctArr[i] ? COK+'aa' : CERR+'55'),
          borderColor: labels.map((_,i) => correctArr[i] ? COK : CERR),
          borderWidth: 1.5, borderRadius: {{ topLeft:0,topRight:0,bottomLeft:4,bottomRight:4 }},
          stack: 'tok',
        }},
        {{
          label: 'Completion Token(K)',
          data: complVals,
          backgroundColor: labels.map((_,i) => correctArr[i] ? '#34d399aa' : '#f87171aa'),
          borderColor: labels.map((_,i) => correctArr[i] ? '#34d399' : '#f87171'),
          borderWidth: 1.5, borderRadius: {{ topLeft:4,topRight:4,bottomLeft:0,bottomRight:0 }},
          stack: 'tok',
        }},
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ position: 'top' }},
        datalabels: {{ display: false }},
        tooltip: {{
          callbacks: {{
            title: (items) => labels[items[0].dataIndex],
            footer: (items) => {{
              const total = items.reduce((s,it)=>s+it.raw,0);
              const ok = correctArr[items[0].dataIndex];
              return `Total: ${{total.toFixed(1)}}K  ${{ok ? '✅ 正确' : '❌ 错误/masked'}}`;
            }}
          }}
        }}
      }},
      scales: {{
        x: {{ ticks: {{ maxRotation: 45, font: {{ size: 9.5 }} }} }},
        y: {{ stacked: true, title: {{ display: true, text: 'Token (K)', color:'#94a3b8' }} }},
      }}
    }}
  }});
}}

makeTokenChart('chartToken0',
  tokenStack.labels0, tokenStack.prompt0, tokenStack.compl0, tokenStack.correct0);
if (HAS_69 && document.getElementById('chartToken69')) {{
  makeTokenChart('chartToken69',
    tokenStack.labels69, tokenStack.prompt69, tokenStack.compl69, tokenStack.correct69);
}}
makeTokenChart('chartToken21',
  tokenStack.labels21, tokenStack.prompt21, tokenStack.compl21, tokenStack.correct21);

// ── 表格渲染 ─────────────────────────────────
function diffTag(d) {{
  const cls = {{base:'tag-diff-base', hard1:'tag-diff-hard1', hard:'tag-diff-hard'}}[d] || '';
  return `<span class="tag ${{cls}}">${{d}}</span>`;
}}
function resultTag(row) {{
  if (row.is_correct) return '<span class="tag tag-ok">✅ 正确</span>';
  if (row.masked)     return '<span class="tag tag-mask">🔲 Masked</span>';
  return '<span class="tag tag-err">❌ 错误</span>';
}}
function groupTag(g) {{
  if (g === '0条')  return '<span class="tag tag-c0">0条</span>';
  if (g === '6-9条') return '<span class="tag tag-c69">6-9条</span>';
  return '<span class="tag tag-c21">21+条</span>';
}}

function renderTable(rows) {{
  const tbody = document.getElementById('tableBody');
  tbody.innerHTML = rows.map(r => `
    <tr>
      <td>${{groupTag(r.group)}}</td>
      <td>${{diffTag(r.difficulty)}}</td>
      <td>${{r.category}}</td>
      <td style="font-family:monospace;font-size:11px;white-space:nowrap">${{r.image_id}}</td>
      <td>${{resultTag(r)}}</td>
      <td style="color:var(--muted);font-size:11px">${{r.mask_reason || '—'}}</td>
      <td style="text-align:center">${{r.rounds}}</td>
      <td style="text-align:right">${{r.time}}</td>
      <td style="text-align:right">${{(r.prompt_tok/1000).toFixed(1)}}</td>
      <td style="text-align:right;${{r.compl_tok>=4096?'color:#f59e0b':''}}">${{(r.compl_tok/1000).toFixed(2)}}</td>
      <td style="text-align:center">${{r.crop}}</td>
      <td style="text-align:center">${{r.search}}</td>
      <td style="text-align:center">${{r.visit}}</td>
      <td style="text-align:center">${{r.valid_r}}</td>
      <td style="text-align:center;${{r.error_r>0?'color:#f87171':''}}">${{r.error_r}}</td>
      <td style="text-align:center">${{r.had_clue ? '<span style="color:#34d399">✓</span>' : '<span style="color:#475569">—</span>'}}</td>
      <td style="text-align:center">${{r.clue_round >= 0 ? r.clue_round : '—'}}</td>
      <td style="text-align:center">${{r.tok_maxed ? '<span style="color:#f59e0b">⚠️</span>' : '—'}}</td>
      <td style="font-size:11.5px;color:var(--muted);max-width:260px;word-break:break-word">${{r.question.replace(/image_id:\S+\s*/,'').slice(0,80)}}</td>
    </tr>`).join('');
}}

renderTable(tableRows);

// 筛选
function applyFilter() {{
  const g = document.getElementById('filterGroup').value;
  const d = document.getElementById('filterDiff').value;
  const r = document.getElementById('filterResult').value;
  const s = document.getElementById('filterSearch').value.toLowerCase();
  renderTable(tableRows.filter(row => {{
    if (g && row.group !== g) return false;
    if (d && row.difficulty !== d) return false;
    if (r === 'correct' && !row.is_correct) return false;
    if (r === 'wrong'   && (row.is_correct || row.masked)) return false;
    if (r === 'masked'  && !row.masked) return false;
    if (s && !row.image_id.toLowerCase().includes(s) && !row.question.toLowerCase().includes(s)) return false;
    return true;
  }}));
}}
['filterGroup','filterDiff','filterResult'].forEach(id =>
  document.getElementById(id).addEventListener('change', applyFilter));
document.getElementById('filterSearch').addEventListener('input', applyFilter);

// ══════════════════════════════════════════════
// 失败原因分析图表
// ══════════════════════════════════════════════

// ── A. 失败类型柱状图 ─────────────────────────
{{
  const ftDs = [
    {{ label: '0条组', data: failureData.group0, backgroundColor: C0+'cc', borderColor: C0, borderWidth: 1.5, borderRadius: 5 }},
  ];
  if (HAS_69 && failureData.group69) ftDs.push({{ label: '6-9条组', data: failureData.group69, backgroundColor: C69+'cc', borderColor: C69, borderWidth: 1.5, borderRadius: 5 }});
  ftDs.push({{ label: '21+条组', data: failureData.group21, backgroundColor: C21+'cc', borderColor: C21, borderWidth: 1.5, borderRadius: 5 }});
  new Chart(document.getElementById('chartFailType'), {{
    type: 'bar',
    data: {{ labels: failureData.labels, datasets: ftDs }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ position: 'top' }},
        datalabels: {{ display: true, color:'#e2e8f0', font:{{size:12,weight:'700'}}, anchor:'end', align:'top' }},
      }},
      scales: {{
        y: {{ beginAtZero: true, ticks: {{ stepSize: 1 }}, title: {{ display: true, text: '失败样本数', color:'#94a3b8' }} }},
        x: {{ ticks: {{ maxRotation: 30, font: {{ size: 11 }} }} }},
      }},
    }}
  }});
}}

// ── B. Masked 原因详细图 ──────────────────────
{{
  const mdSrc = [maskReasons0, ...(HAS_69 ? [maskReasons69] : []), maskReasons21];
  const mdLabels = [...new Set(mdSrc.flatMap(o => Object.keys(o)))];
  const mdDs = [
    {{ label: '0条组', data: mdLabels.map(k => maskReasons0[k]||0), backgroundColor: C0+'cc', borderColor: C0, borderWidth: 1.5, borderRadius: 5 }},
  ];
  if (HAS_69) mdDs.push({{ label: '6-9条组', data: mdLabels.map(k => maskReasons69[k]||0), backgroundColor: C69+'cc', borderColor: C69, borderWidth: 1.5, borderRadius: 5 }});
  mdDs.push({{ label: '21+条组', data: mdLabels.map(k => maskReasons21[k]||0), backgroundColor: C21+'cc', borderColor: C21, borderWidth: 1.5, borderRadius: 5 }});
  new Chart(document.getElementById('chartMaskedDetail'), {{
    type: 'bar',
    data: {{ labels: mdLabels, datasets: mdDs }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ position: 'top' }},
        datalabels: {{ display: true, color:'#e2e8f0', font:{{size:12,weight:'700'}}, anchor:'end', align:'top' }},
      }},
      scales: {{
        y: {{ beginAtZero: true, ticks: {{ stepSize: 1 }} }},
        x: {{ ticks: {{ maxRotation: 20 }} }},
      }},
    }}
  }});
}}

// ── C. 答错 vs Masked 堆叠比例图 ─────────────
{{
  function countFailKind(fc, kind) {{
    return Object.entries(fc).filter(([k]) => k.startsWith(kind)).reduce((s,[,v])=>s+v, 0);
  }}
  const grpNames = ['0条组', ...(HAS_69 ? ['6-9条组'] : []), '21+条组'];
  const fcAll = [failureData.group0, ...(HAS_69 && failureData.group69 ? [failureData.group69] : []), failureData.group21];
  const flabels = failureData.labels;
  const maskedCounts = fcAll.map(arr => flabels.reduce((s,l,i) => l.startsWith('M') ? s+arr[i] : s, 0));
  const wrongCounts  = fcAll.map(arr => flabels.reduce((s,l,i) => l.startsWith('W') ? s+arr[i] : s, 0));
  new Chart(document.getElementById('chartFailRatio'), {{
    type: 'bar',
    data: {{
      labels: grpNames,
      datasets: [
        {{ label: 'Masked类（M）', data: maskedCounts, backgroundColor: CMASK+'cc', borderColor: CMASK, borderWidth: 1.5, borderRadius: {{topLeft:0,topRight:0,bottomLeft:5,bottomRight:5}}, stack:'fail' }},
        {{ label: '答错类（W）',   data: wrongCounts,  backgroundColor: CERR+'cc',  borderColor: CERR,  borderWidth: 1.5, borderRadius: {{topLeft:5,topRight:5,bottomLeft:0,bottomRight:0}}, stack:'fail' }},
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ position: 'top' }},
        datalabels: {{ display: true, color:'#fff', font:{{size:13,weight:'700'}}, anchor:'center', align:'center',
          formatter: v => v > 0 ? v : '' }},
      }},
      scales: {{ y: {{ stacked: true, beginAtZero: true, ticks: {{ stepSize: 1 }} }} }},
    }}
  }});
}}

// ── D. 失败样本明细表 ─────────────────────────
const FAIL_TYPE_COLOR = {{
  'M1: 50轮未找到答案':    'tag-mask',
  'M2: 重复检测截断':      'tag-mask',
  'M3: 连续步骤错误':      'tag-mask',
  'M4: 系统/服务器错误':   'tag-mask',
  'M5: 其他Masked':        'tag-mask',
  'W1: 实体识别错误':      'tag-err',
  'W2: 细节/内容错误':     'tag-err',
  'W2: 细节/数字错误':     'tag-err',
  'W3: 命中线索仍答错':    'tag-err',
  'W4: 快速幻觉(≤3轮)':   'tag-err',
}};

function renderFailTable(rows) {{
  const tbody = document.getElementById('failTableBody');
  tbody.innerHTML = rows.map(r => {{
    const ftCls = FAIL_TYPE_COLOR[r.failure_type] || 'tag-err';
    const judgeShort = r.judge_judgment ? r.judge_judgment.replace(/^correct: no\s*\\nreasoning:\s*/i, '').slice(0, 100) + (r.judge_judgment.length > 100 ? '…' : '') : '';
    return `
    <tr>
      <td>${{groupTag(r.group)}}</td>
      <td>${{diffTag(r.difficulty)}}</td>
      <td>${{r.category}}</td>
      <td style="font-size:11px;word-break:break-all">${{r.image_id}}</td>
      <td><span class="tag ${{ftCls}}" style="white-space:nowrap;font-size:10px">${{r.failure_type}}</span></td>
      <td style="text-align:center">${{r.rounds}}</td>
      <td style="color:#34d399;max-width:140px">${{r.answer}}</td>
      <td style="color:#f87171;max-width:150px;font-size:11px">${{r.prediction}}</td>
      <td style="color:#94a3b8;max-width:220px;font-size:11px">${{judgeShort}}</td>
      <td style="text-align:center">${{r.had_clue ? '✅' : '—'}}</td>
      <td style="text-align:center">${{r.tok_maxed ? '⚠️' : '—'}}</td>
    </tr>`;
  }}).join('');
}}

renderFailTable(failRows);

function applyFailFilter() {{
  const g = document.getElementById('filterFailGroup').value;
  const t = document.getElementById('filterFailType').value;
  const s = document.getElementById('filterFailSearch').value.toLowerCase();
  renderFailTable(failRows.filter(r => {{
    if (g && r.group !== g) return false;
    if (t && !r.failure_type.startsWith(t)) return false;
    if (s && !r.image_id.toLowerCase().includes(s) && !(r.answer||'').toLowerCase().includes(s)) return false;
    return true;
  }}));
}}
['filterFailGroup','filterFailType'].forEach(id =>
  document.getElementById(id).addEventListener('change', applyFailFilter));
document.getElementById('filterFailSearch').addEventListener('input', applyFailFilter);
</script>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n✅ HTML 报告已生成: {output_path}")


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────

def main():
    base_dir = "/home/yangyajie/ljq/Vision-DeepResearch/data/eval_outputs"
    path_0   = os.path.join(base_dir, "0/episodes.json")
    path_69  = os.path.join(base_dir, "6-9/episodes.json")
    path_21  = os.path.join(base_dir, "21+/episodes.json")

    print("正在加载数据...")
    eps_0  = load_json(path_0)
    eps_21 = load_json(path_21)

    data_0  = analyze_group(eps_0,  "0条反向检索结果")
    data_21 = analyze_group(eps_21, "21+条反向检索结果")

    data_69 = None
    if os.path.exists(path_69):
        eps_69  = load_json(path_69)
        data_69 = analyze_group(eps_69, "6-9条反向检索结果")

    compare_groups(data_0, data_21, data_69)

    # 生成 HTML 可视化报告
    report_path = "/home/yangyajie/ljq/Vision-DeepResearch/data/eval_analysis/analysis_report.html"
    generate_html_report(data_0, data_21, report_path, data_69)

if __name__ == '__main__':
    main()
