#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -C 2080ti
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -J moe-diag
#SBATCH -o logs/%x-%j.out
#
# routing and per expert accuracy for the saved eval checkpoints:  sbatch jobs/diagnose_cuda.sh
set -uo pipefail
source jobs/env.sh
$PY -u scripts/diagnose.py --device cuda --end-date 2026-11-22 --out results/diagnose.json 2>&1 | grep --line-buffered -v -i warn
