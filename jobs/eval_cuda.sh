#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -C 2080ti
#SBATCH --mem=24G
#SBATCH -t 10:00:00
#SBATCH -J moe-eval
#SBATCH -o logs/%x-%j.out
#
# full walk forward eval on nifty 50 for one seed:  sbatch --export=ALL,SEED=0 jobs/eval_cuda.sh
set -uo pipefail
source jobs/env.sh
nvidia-smi --query-gpu=name --format=csv,noheader
$PY -m pytest -q tests | tail -2
SEED=${SEED:-0}
$PY -u scripts/evaluate.py --device cuda --seed $SEED --reuse ${REUSE:-0} ${EVAL_ARGS:-} --out results/eval_nifty_s$SEED.json 2>&1 | grep --line-buffered -v -i warn
