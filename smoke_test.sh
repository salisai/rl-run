#!/usr/bin/env bash
# Verifies the WHOLE pipeline (train -> save -> eval -> plot) with a tiny 0.5B model.
# Takes a few minutes on a GPU. Run it tonight and again at the lab before the real run.
set -euo pipefail
source .venv/bin/activate
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"

python check_env.py
python train.py --smoke
python eval.py --smoke --adapter outputs/smoke/final
python plot.py --run outputs/smoke --out smoke_curve.png
echo
echo "SMOKE TEST PASSED. The real run will use the same code path."
