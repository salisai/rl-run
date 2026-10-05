"""Evaluate a model (optionally with a LoRA adapter) on held-out Countdown problems.

  python eval.py --name baseline
  python eval.py --name trained --adapter outputs/grpo-countdown/final
  python eval.py --smoke            # quick check with the 0.5B model, 8 problems
"""
import argparse
import json
import os
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from countdown import (EVAL_SEED, TRAIN_SEED, build_problems, has_format,
                       problem_key, score_one)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", default=None, help="LoRA adapter dir (e.g. outputs/grpo-countdown/final)")
    p.add_argument("--name", default="eval", help="used for the results filename")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--n_numbers", type=int, default=3)
    p.add_argument("--max_num", type=int, default=50)
    p.add_argument("--train_size", type=int, default=2000, help="must match train.py to keep splits disjoint")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.model, args.n, args.batch_size, args.max_new_tokens = "Qwen/Qwen2.5-0.5B-Instruct", 8, 4, 128
        args.name = "smoke"

    # Held-out problems: different seed, and none overlap the training set.
    train = build_problems(args.train_size, TRAIN_SEED, args.n_numbers, args.max_num)
    problems = build_problems(args.n, EVAL_SEED, args.n_numbers, args.max_num,
                              exclude={problem_key(p) for p in train})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.to(device).eval()

    records, statuses, n_format, total_tokens = [], Counter(), 0, 0
    for i in range(0, len(problems), args.batch_size):
        batch = problems[i:i + args.batch_size]
        texts = [tok.apply_chat_template(p["prompt"], tokenize=False, add_generation_prompt=True)
                 for p in batch]
        enc = tok(texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        for p, ids in zip(batch, new_tokens):
            completion = tok.decode(ids, skip_special_tokens=True)
            reward, status = score_one(completion, p["nums"], p["target"])
            statuses[status] += 1
            n_format += has_format(completion)
            total_tokens += int((ids != tok.pad_token_id).sum())
            records.append({"nums": p["nums"], "target": p["target"], "status": status,
                            "completion": completion})
        print(f"  {min(i + args.batch_size, len(problems))}/{len(problems)}", flush=True)

    n = len(problems)
    metrics = {
        "name": args.name, "model": args.model, "adapter": args.adapter, "n": n,
        "accuracy": statuses["correct"] / n,
        "format_rate": n_format / n,
        "avg_completion_tokens": total_tokens / n,
        "status_counts": dict(statuses),
    }
    os.makedirs("results", exist_ok=True)
    path = f"results/{args.name}.json"
    with open(path, "w") as f:
        json.dump({"metrics": metrics, "samples": records}, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
