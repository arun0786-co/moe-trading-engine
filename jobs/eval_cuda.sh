#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -C 2080ti
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH -J moe-eval
#SBATCH -o logs/%x-%j.out
#
# full walk forward eval on nifty 50 for one seed:  sbatch --export=ALL,SEED=0 jobs/eval_cuda.sh
set -uo pipefail
SCR=""
for base in /scratch /ssd_scratch /tmp; do
    if [ -n "$(df -P $base 2>/dev/null | awk 'NR == 2 && $4 > 20 * 1024 * 1024')" ]; then SCR=$base/$USER/moe; break; fi
done
[ -n "$SCR" ] || { echo "no local disk"; exit 3; }
mkdir -p $SCR
export UV_CACHE_DIR=$SCR/cache/uv UV_PYTHON_INSTALL_DIR=$SCR/python
[ -x $SCR/tools/bin/uv ] || { python3 -m venv $SCR/tools && $SCR/tools/bin/pip install -q uv; }
[ -x $SCR/env/bin/python ] || $SCR/tools/bin/uv venv -q --python 3.12 $SCR/env
PY=$SCR/env/bin/python
$SCR/tools/bin/uv pip install -q --python $PY --torch-backend=cu126 torch numpy pandas scipy yfinance ta pyyaml torchdiffeq pytest
nvidia-smi --query-gpu=name --format=csv,noheader
$PY -m pytest -q tests | tail -2
SEED=${SEED:-0}
$PY scripts/evaluate.py --device cuda --seed $SEED --out results/eval_nifty_s$SEED.json 2>&1 | grep -v -i warn
