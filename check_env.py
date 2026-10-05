"""Print what this machine has. Run first, everywhere:  python check_env.py"""
import importlib
import platform

print("python      ", platform.python_version())
for pkg in ["torch", "transformers", "trl", "peft", "accelerate", "datasets", "vllm", "wandb"]:
    try:
        m = importlib.import_module(pkg)
        print(f"{pkg:<12}", getattr(m, "__version__", "?"))
    except Exception:
        print(f"{pkg:<12} not installed")

try:
    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB, "
                  f"bf16={torch.cuda.is_bf16_supported()}")
    else:
        print("No CUDA GPU visible (smoke tests will run on CPU, slowly).")
except Exception as e:
    print("torch check failed:", e)
