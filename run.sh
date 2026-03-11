#!/usr/bin/env bash
# set是什么意思
# -e: 当脚本中的任何命令返回非零状态时，立即退出
# -u: 当脚本中使用未定义的变量时，立即退出
# -o pipefail: 如果管道中的任何命令失败，整个管道返回
set -euo pipefail

# 进入脚本所在目录
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

PY_FILE="demo_code.py"

# 第1个参数可指定日志文件路径；未传则使用默认路径,并添加时间戳
LOG_FILE="${1:-$PROJECT_DIR/logs/demo_code_$(date +"%m%d_%H%M").log}"
# LOG_FILE="${1:-$PROJECT_DIR/logs/demo_code.log}"
PID_FILE="$PROJECT_DIR/run.pid"

# 可选：如需激活 conda 环境，取消下面两行注释并替换环境名
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env_name

mkdir -p "$(dirname "$LOG_FILE")"

# 后台运行并记录 PID
nohup python -u "$PY_FILE" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "Started: $PY_FILE"
echo "PID: $(cat "$PID_FILE")"
echo "Log: $LOG_FILE"