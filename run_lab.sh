#!/usr/bin/env bash
# The one command for the lab:  bash run_lab.sh
# Extra arguments are passed to train.py, e.g.  bash run_lab.sh --max_steps 500 --n_numbers 4
set -euo pipefail
source .venv/bin/activate
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
# export HF_HUB_OFFLINE=1   # uncomment if the lab has no internet (models are already cached)

OUT=outputs/grpo-countdown

python check_env.py
python eval.py --name baseline --n 200                       # 1) before training
python train.py --output_dir "$OUT" "$@"                     # 2) GRPO training
python eval.py --name trained --adapter "$OUT/final" --n 200 # 3) after training
python plot.py --run "$OUT" --out reward_curve.png           # 4) the tweet chart
echo
echo "Done. See results/baseline.json, results/trained.json, reward_curve.png"
