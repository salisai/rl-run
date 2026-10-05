#!/usr/bin/env bash
# One-time setup. Run on your own machine tonight:  bash setup.sh
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# Keep model downloads inside the project so you can copy the whole folder to the lab.
export HF_HOME="$PWD/.hf_cache"

python test_countdown.py
python check_env.py

python - <<'EOF'
from huggingface_hub import snapshot_download
for repo in ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]:
    print("downloading", repo)
    snapshot_download(repo)
print("models cached in", __import__("os").environ["HF_HOME"])
EOF

pip freeze > requirements.lock
echo
echo "Setup done. Next: bash smoke_test.sh"
