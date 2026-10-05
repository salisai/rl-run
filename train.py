"""GRPO on the Countdown task with TRL + LoRA.

Examples:
  python train.py --smoke                      # 3-step pipeline check with a 0.5B model
  python train.py                              # real run (defaults below)
  python train.py --n_numbers 4 --max_steps 500
  python train.py --batch_size 4 --grad_accum 8   # for smaller GPUs
"""
import argparse
import inspect
import json
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from countdown import (TRAIN_SEED, build_problems, correctness_reward,
                       format_reward)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output_dir", default="outputs/grpo-countdown")
    p.add_argument("--n_numbers", type=int, default=3, help="difficulty: how many numbers")
    p.add_argument("--max_num", type=int, default=50)
    p.add_argument("--train_size", type=int, default=2000)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=8, help="per-device completions per micro-batch")
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--num_generations", type=int, default=8, help="group size G")
    p.add_argument("--max_completion_length", type=int, default=512)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report_to", default="none", help="none | wandb | tensorboard")
    p.add_argument("--use_vllm", action="store_true", help="faster generation; needs vllm installed")
    p.add_argument("--grad_ckpt", action="store_true", help="save memory, slower")
    p.add_argument("--resume", default=None, help="path to a checkpoint dir to resume from")
    p.add_argument("--smoke", action="store_true", help="tiny 3-step run to verify the pipeline")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.model = SMOKE_MODEL
        args.max_steps, args.train_size = 3, 32
        args.batch_size, args.grad_accum, args.num_generations = 4, 1, 4
        args.max_completion_length = 128
        args.output_dir = "outputs/smoke"

    # TRL needs (batch_size * grad_accum * n_gpus) divisible by num_generations.
    if (args.batch_size * args.grad_accum) % args.num_generations != 0:
        raise SystemExit(
            f"batch_size*grad_accum = {args.batch_size * args.grad_accum} must be "
            f"divisible by num_generations = {args.num_generations}")

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"model={args.model} cuda={use_cuda} bf16={use_bf16}")

    # ---- data: procedural, so nothing to download -------------------------------
    problems = build_problems(args.train_size, TRAIN_SEED, args.n_numbers, args.max_num)
    train_ds = Dataset.from_list(problems)

    # ---- model + LoRA -----------------------------------------------------------
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    except TypeError:  # older transformers
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )

    # ---- GRPO config (unknown keys for your TRL version are dropped with a warning) ----
    cfg = dict(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=1.0,
        beta=0.0,                       # no KL penalty: simpler, saves memory
        bf16=use_bf16,
        gradient_checkpointing=args.grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_steps=5,
        lr_scheduler_type="constant_with_warmup",
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        report_to=args.report_to,
        seed=args.seed,
        log_completions=True,           # prints sample completions while training
        num_completions_to_print=2,
        use_vllm=args.use_vllm,
    )
    valid = set(inspect.signature(GRPOConfig.__init__).parameters)
    dropped = sorted(k for k in cfg if k not in valid)
    if dropped:
        print(f"[warn] this TRL version does not know: {dropped} (ignored)")
    grpo_args = GRPOConfig(**{k: v for k, v in cfg.items() if k in valid})

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[correctness_reward, format_reward],
        args=grpo_args,
        train_dataset=train_ds,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=args.resume)

    # ---- save adapter + logs ---------------------------------------------------
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    with open(os.path.join(args.output_dir, "log_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f)
    with open(os.path.join(args.output_dir, "run_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Done. Adapter saved to {final_dir}")


if __name__ == "__main__":
    main()
