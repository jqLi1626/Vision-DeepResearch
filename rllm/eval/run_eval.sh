SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLLM_DIR="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$RLLM_DIR:$PYTHONPATH"

# 加载 .env 文件中的环境变量
if [ -f "$RLLM_DIR/.env" ]; then
  set -a
  source "$RLLM_DIR/.env"
  set +a
fi

# ── SSH 反向隧道代理 ──────────────────────────────────────────
# worker 节点不能直接联网，运行此脚本前需先在【本地电脑】建立反向隧道。
#
#   pip install pproxy
#   pproxy -l socks5://:11080 &
#   ssh -R 7891:127.0.0.1:11080 ws-b25f492fce479505-worker-blmfc.yangyajie.ailab-ma4agismall.pod@h.pjlab.org.cn

# ─────────────────────────────────────────────────────────────
PROXY_PORT=7891
if ss -tnlp 2>/dev/null | grep -q ":${PROXY_PORT}" || \
   netstat -tnlp 2>/dev/null | grep -q ":${PROXY_PORT}"; then
  echo "[proxy] 检测到 SOCKS5 代理已在 :${PROXY_PORT} 运行，启用代理..."
  # 只对 HTTPS 流量走代理（OSS/Serper 等外网服务均为 HTTPS）
  # 不设置 HTTP_PROXY/http_proxy，避免 localhost:8002 (judge/extract) 被错误路由到 SOCKS
  export HTTPS_PROXY="socks5h://127.0.0.1:${PROXY_PORT}"
  export https_proxy="socks5h://127.0.0.1:${PROXY_PORT}"
  export TOOL_HTTPS_PROXY="socks5h://127.0.0.1:${PROXY_PORT}"
  # 本地/内网地址不走代理
  export no_proxy="127.0.0.1,localhost,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,100.96.0.0/12"
  export NO_PROXY="${no_proxy}"
else
  echo "[proxy] 警告：未检测到代理 (port ${PROXY_PORT})，Serper/Jina Reader/OSS 等外网服务无法使用。"
  echo "[proxy] 请先按脚本注释中的方式建立反向隧道。"
  # 无代理时让 OSS 尝试直连（规避 pjlab Squid 的 403）
  export no_proxy="${no_proxy},aliyuncs.com,.aliyuncs.com"
  export NO_PROXY="${NO_PROXY},aliyuncs.com,.aliyuncs.com"
fi
# ─────────────────────────────────────────────────────────────

mkdir -p "$SCRIPT_DIR/logs"

# =============================================================================
# 模式一：标准模式（单一模型，兼容原有训练好的模型）
# =============================================================================
# /data/miniconda3/envs/vision-deepresearch/bin/python -u "$SCRIPT_DIR/eval_runner.py" \
#   --parquet /home/yangyajie/ljq/Vision-DeepResearch/vdr_testmini_0_matches.parquet \
#   --base-url https://aiberm.com/v1 \
#   --model google/gemini-3.1-pro \
#   --api-key sk-btCGkgdyFPhB2aowxniX7K3zXChEZzqsdUT82GHdMkCnDtkC \
#   --parallel-tasks 3 \
#   2>&1 | tee "$SCRIPT_DIR/logs/log.txt"

# =============================================================================
# 模式二：Planner-Executor 双模型模式
#
# 架构：
#   Planner（大模型）  → 只负责 <think> 推理，描述想做什么
#       ↓  thinking 内容（含 → ACTION 指令）
#   Executor（小模型） → 看到 thinking + 图像，生成精确的 tool_call / bbox / answer
#       ↓  tool_call 或 answer
#   工具执行（crop_and_search / search / visit）
#       ↓  tool_response
#   回到 Planner 继续推理...
#
# 优势：
#   - 大模型不再需要输出精确 bbox，只需描述"裁剪图像左上角的 logo"
#   - 小模型专门训练 bbox 精度和 tool_call 格式规范性
#   - 大模型的格式错误大幅减少（stop token 在 </think> 截断）
# =============================================================================
/data/miniconda3/envs/vision-deepresearch/bin/python -u "$SCRIPT_DIR/eval_runner.py" \
  --parquet /home/yangyajie/ljq/Vision-DeepResearch/vdr_testmini_0_matches.parquet \
  --agent-mode planner-executor \
  \
  `# ── Planner：大模型，负责 think 推理 ──────────────────────` \
  --base-url https://aiberm.com/v1 \
  --model google/gemini-3.1-pro \
  --api-key sk-btCGkgdyFPhB2aowxniX7K3zXChEZzqsdUT82GHdMkCnDtkC \
  \
  `# ── Executor：训练好的小模型，负责 tool_call / bbox ───────` \
  --executor-base-url http://localhost:8001/v1 \
  --executor-model Vision-DeepResearch-8B \
  --executor-api-key EMPTY \
  --executor-temperature 0.0 \
  \
  `# ── 其他参数 ─────────────────────────────────────────────` \
  --parallel-tasks 3 \
  --max-crops-per-call 1 \
  2>&1 | tee "$SCRIPT_DIR/logs/log.txt"
