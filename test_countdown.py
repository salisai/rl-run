"""Plain-assert tests. Run:  python test_countdown.py   (also works with pytest)."""
from fractions import Fraction

from countdown import (build_problems, correctness_reward, extract_answer,
                       format_reward, problem_key, safe_eval, score_one)


def test_safe_eval():
    assert safe_eval("(1 + 2) * 3") == Fraction(9)
    assert safe_eval("8 / 4 / 2") == Fraction(1)
    assert safe_eval("1 / 3 * 3") == Fraction(1)      # exact, no float error
    assert safe_eval("1 / 0") is None
    assert safe_eval("__import__('os')") is None       # no code execution
    assert safe_eval("2 ** 3") is None                 # power not allowed
    assert safe_eval("-3 + 5") is None                 # unary minus not allowed
    assert safe_eval("1.5 + 1") is None                # floats not allowed
    assert safe_eval("") is None


def test_extract_answer():
    assert extract_answer("<answer>(1+2)*3</answer>") == "(1+2)*3"
    assert extract_answer("<answer> 1+2 = 3 </answer>") == "1+2"
    assert extract_answer("<answer>1+1</answer> ... <answer>2+2</answer>") == "2+2"
    assert extract_answer("no tags here") is None


def test_score_one():
    nums, target = [1, 2, 3], 9
    assert score_one("<answer>(1 + 2) * 3</answer>", nums, target) == (1.0, "correct")
    assert score_one("<answer>1 + 2 + 3</answer>", nums, target) == (0.1, "wrong_value")
    assert score_one("<answer>(1 + 2) * 3 * 1</answer>", nums, target)[1] == "bad_numbers"
    assert score_one("<answer>9</answer>", nums, target)[1] == "bad_numbers"
    assert score_one("<answer>1 + two</answer>", nums, target)[1] == "invalid"
    assert score_one("(1+2)*3", nums, target)[1] == "no_answer"
    assert score_one("<answer>" + "9" * 5000 + "</answer>", nums, target)[1] == "invalid"


def test_rewards_batch_and_chat_format():
    comps = [
        [{"role": "assistant", "content": "<think>ok</think><answer>(1+2)*3</answer>"}],
        "no answer at all",
    ]
    assert correctness_reward(comps, nums=[[1, 2, 3], [1, 2, 3]], target=[9, 9]) == [1.0, 0.0]
    assert format_reward(comps) == [0.1, 0.0]


def test_generator_solutions_are_correct_and_reproducible():
    probs = build_problems(200, seed=0, n_numbers=3, max_num=50)
    assert len(probs) == 200
    assert len({problem_key(p) for p in probs}) == 200  # unique
    for p in probs:
        text = f"<think>x</think><answer>{p['solution']}</answer>"
        assert score_one(text, p["nums"], p["target"]) == (1.0, "correct")
    assert probs == build_problems(200, seed=0, n_numbers=3, max_num=50)


def test_splits_are_disjoint():
    train = build_problems(300, seed=0)
    ev = build_problems(100, seed=12345, exclude={problem_key(p) for p in train})
    assert not ({problem_key(p) for p in train} & {problem_key(p) for p in ev})


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
    print("All tests passed.")
