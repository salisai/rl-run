# GRPO Countdown: teach a small model to reason with RL

Train a small Qwen model with GRPO on **Countdown** (given some numbers and a
target, write an equation that hits the target using each number once).
The answer is checked by code, so no reward model or dataset download is needed.

**Design choices (kept simple on purpose):**
one task, one model, TRL + LoRA, no YAML, no vLLM required.
Everything lives in six small files.

| File | Purpose |
|---|---|
| `countdown.py` | Problem generator, safe verifier, reward functions (pure Python) |
| `train.py` | GRPO training (TRL + LoRA) |
| `eval.py` | Accuracy on held-out problems, before and after training |
| `plot.py` | Reward and length curves for your tweet |
| `check_env.py` | Prints versions and GPU info |
| `test_countdown.py` | Unit tests for the task code |

---

## Tonight (on your own machine)

```bash
bash setup.sh          # venv + dependencies + tests + downloads models into ./.hf_cache
bash smoke_test.sh     # 3-step run with a 0.5B model: train -> save -> eval -> plot
```

If the smoke test passes, the real run will work. If you have no GPU at home,
the smoke test still runs on CPU (slowly); at minimum run `python test_countdown.py`.

**Pack for the lab:** copy the whole folder, including `.venv` only if the lab machine
has the same OS and Python (otherwise just re-run `bash setup.sh` there, or copy
`.hf_cache` and `requirements.lock` so nothing big needs downloading).

## At the lab

```bash
bash setup.sh          # skip if .venv already works there
bash smoke_test.sh     # 5 minutes, catches version or driver problems early
bash run_lab.sh        # baseline eval -> training -> post eval -> plot
```

Run it inside `tmux` or `screen` so a dropped connection does not kill training.
Training saves a checkpoint every 50 steps; resume with
`python train.py --resume outputs/grpo-countdown/checkpoint-100`.

## What to watch while it trains

Healthy signs:
- `correctness_reward` trends upward (the noisy curve is normal; `plot.py` smooths it).
- Printed sample completions get more structured: `<think>` reasoning, then a valid `<answer>`.
- Completion length stays stable or grows slowly.

Warning signs:
- **Reward stuck at ~0.1 for 50+ steps:** the task is too hard. Use `--n_numbers 3 --max_num 30`.
- **Reward jumps to ~1.0 immediately:** too easy; raise `--n_numbers` or `--max_num`.
- **Length collapses to a few tokens or explodes to the max:** the model is gaming the reward. Read the samples.
- **Format reward high, correctness at 0:** the model learned the tags but not the math. Be patient, or lower difficulty.

## Knobs you may want

| Flag | Default | Notes |
|---|---|---|
| `--model` | Qwen2.5-1.5B-Instruct | Swap for another causal LM; same flag in `eval.py` |
| `--n_numbers` | 3 | Difficulty. 4 and 5 are much harder |
| `--max_steps` | 300 | Time the first ~5 steps, then decide |
| `--num_generations` | 8 | Group size. `batch_size * grad_accum` must be divisible by it |
| `--batch_size`, `--grad_accum` | 8, 4 | Use `4, 8` if you run out of memory |
| `--use_vllm` | off | Much faster generation if `vllm` is installed and compatible |
| `--report_to wandb` | none | Needs `pip install wandb` and `wandb login` |

## Troubleshooting

- **CUDA out of memory:** `--batch_size 4 --grad_accum 8`, add `--grad_ckpt`, or lower `--max_completion_length 384`.
- **"must be divisible by num_generations":** adjust the flags so `batch_size * grad_accum` is a multiple of `num_generations`.
- **Unknown config key warning:** your TRL version does not support that option. It is ignored automatically, which is usually fine.
- **Slow generation:** the default uses plain Hugging Face `generate`. Try `--use_vllm`.
- **`eval.py` accuracy looks too good or too bad:** it uses greedy decoding on held-out problems with no overlap with the training set.

## Ideas for after the first run (the tweet-worthy parts)

1. **Size ladder:** run the same recipe on 0.5B, 1.5B, 3B and plot final accuracy against size.
2. **Difficulty curriculum:** train on 3 numbers, then test on 4 (does reasoning transfer?).
3. **Reward ablation:** remove the partial-credit `0.1` and see what changes.
4. **Show samples:** pick a "before" and "after" completion for the same problem.
5. **Swap the task** for another verifiable one (Reasoning Gym, Sudoku) by replacing `countdown.py`.

## Honest notes

- Defaults (learning rate 2e-5, LoRA rank 32, group size 8) are sensible starting points, not tuned values.
- Library APIs change between TRL versions. `requirements.lock` is written by `setup.sh` so you can reproduce the exact environment later.
