"""Countdown task for GRPO: problem generation, prompt, safe verification, rewards.

Pure Python (no torch), so it can be tested anywhere:  python test_countdown.py

Task: given a few numbers and a target, write an arithmetic expression that uses
each number exactly once and evaluates to the target. The answer can be checked
automatically, which is exactly what GRPO needs.
"""
import ast
import operator
import random
import re
from fractions import Fraction

TRAIN_SEED = 0
EVAL_SEED = 12345

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

PROMPT_TEMPLATE = (
    "Using the numbers {nums}, create an equation that equals {target}. "
    "You may use + - * / and parentheses, and each number must be used exactly once. "
    "Think step by step inside <think> </think> tags, then write only the final "
    "equation inside <answer> </answer> tags, for example <answer>(1 + 2) * 3</answer>."
)

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ALLOWED_CHARS = re.compile(r"[\d+\-*/()\s]+")


# ----------------------------------------------------------------------------
# Safe evaluation (no eval(): only ints, + - * / and parentheses, exact math)
# ----------------------------------------------------------------------------
def _eval_node(node):
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return Fraction(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    raise ValueError("unsupported expression")


def safe_eval(expr):
    """Return the exact value (Fraction) of expr, or None if it is not valid."""
    try:
        tree = ast.parse(expr.strip(), mode="eval")
        return _eval_node(tree.body)
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError,
            MemoryError, OverflowError):
        return None


# ----------------------------------------------------------------------------
# Parsing and scoring a model completion
# ----------------------------------------------------------------------------
def extract_answer(text):
    """Return the expression inside the LAST <answer> tag, or None."""
    found = _ANSWER_RE.findall(text)
    if not found:
        return None
    expr = found[-1].strip()
    if "=" in expr:  # tolerate "(1+2)*3 = 9"
        expr = expr.split("=")[0].strip()
    return expr


def has_format(text):
    return bool(_THINK_RE.search(text)) and bool(_ANSWER_RE.search(text))


def score_one(text, nums, target):
    """Return (reward, status). Status is useful for eval and debugging."""
    expr = extract_answer(text)
    if expr is None:
        return 0.0, "no_answer"
    if len(expr) > 200 or not _ALLOWED_CHARS.fullmatch(expr):
        return 0.0, "invalid"
    used = sorted(int(x) for x in re.findall(r"\d+", expr))
    if used != sorted(nums):
        return 0.0, "bad_numbers"
    value = safe_eval(expr)
    if value is None:
        return 0.0, "invalid"
    if value == target:
        return 1.0, "correct"
    return 0.1, "wrong_value"  # valid equation, right numbers, wrong result


# ----------------------------------------------------------------------------
# Reward functions for TRL (signature: completions, **dataset_columns)
# ----------------------------------------------------------------------------
def _text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:  # chat format
        return completion[-1].get("content", "")
    return str(completion)


def format_reward(completions, **kwargs):
    """0.1 if the completion has both <think> and <answer> blocks."""
    return [0.1 if has_format(_text(c)) else 0.0 for c in completions]


def correctness_reward(completions, nums, target, **kwargs):
    """1.0 correct, 0.1 valid-but-wrong value, 0.0 otherwise."""
    return [score_one(_text(c), n, t)[0] for c, n, t in zip(completions, nums, target)]


# ----------------------------------------------------------------------------
# Problem generation (procedural: unlimited data, a solution always exists)
# ----------------------------------------------------------------------------
def _random_problem(rng, n_numbers, max_num):
    nums = [rng.randint(1, max_num) for _ in range(n_numbers)]
    order = nums[:]
    rng.shuffle(order)
    expr, value = str(order[0]), Fraction(order[0])
    for x in order[1:]:
        op = rng.choice("+-*/")
        if op == "+":
            value = value + x
        elif op == "-":
            value = value - x
        elif op == "*":
            value = value * x
        else:
            value = value / x
        expr = f"({expr} {op} {x})"
        if value.denominator != 1 or value <= 0:
            return None  # keep every intermediate value a positive integer
    target = int(value)
    if target > 999 or target in nums:
        return None
    return {
        "nums": nums,
        "target": target,
        "solution": expr,
        "prompt": [{
            "role": "user",
            "content": PROMPT_TEMPLATE.format(nums=nums, target=target),
        }],
    }


def problem_key(p):
    return (tuple(sorted(p["nums"])), p["target"])


def build_problems(n, seed, n_numbers=3, max_num=50, exclude=None):
    """Generate n unique problems. Pass exclude=set of keys to keep splits disjoint."""
    rng = random.Random(seed)
    seen = set(exclude or [])
    out, tries = [], 0
    while len(out) < n:
        tries += 1
        if tries > n * 2000 + 10000:
            raise RuntimeError("Could not generate enough unique problems; "
                               "raise --max_num or --n_numbers.")
        p = _random_problem(rng, n_numbers, max_num)
        if p is None:
            continue
        key = problem_key(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out
