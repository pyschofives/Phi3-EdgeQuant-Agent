"""
quant_8bit.py — updated
Produce an 8-bit quantized Phi-3 model.

Strategy:
1) Try GPTQModel (preferred) using its quantize API (quantize bits=8).
2) If GPTQModel call fails for any reason, fallback to bitsandbytes-style
   8-bit using transformers + BitsAndBytesConfig (load_in_8bit -> then save).

This update fixes a few issues you hit earlier:
- Use POSIX-style local paths when running in WSL (convert D:\ -> /mnt/d/).
- Use local_files_only=True when loading from a local folder to avoid HF repo parsing.
- Try multiple GPTQModel signatures and use safer save paths.

Notes:
- Run inside WSL for best bitsandbytes/GPTQ CUDA extension support.
- If you still hit native binary errors for bitsandbytes, you likely need to
  install/compile the bitsandbytes binary matching your CUDA version.
"""
import os
import sys
import traceback
from pathlib import Path

# --- User-editable paths (put local Windows path or POSIX WSL path) ---
RAW_MODEL_PATH = r"D:/Phi3-EdgeQuant-Agent/models/microsoft/Phi-3-mini-4k-instruct"
RAW_OUTPUT_PATH = r"D:/Phi3-EdgeQuant-Agent/models/quantized/Phi-3-mini-4k-instruct-8bit-gptq"

# --- Helpers ------------------------------------------------------------

def running_under_wsl() -> bool:
    """Detect WSL environment."""
    try:
        with open("/proc/version", "r", encoding="utf-8") as f:
            return "microsoft" in f.read().lower() or "wsl" in f.read().lower()
    except Exception:
        return False


def to_wsl_path(win_path: str) -> str:
    """Convert a Windows path like `D:\\...` or `D:/...` to `/mnt/d/...` for WSL.
    If the path already looks POSIX, return unchanged.
    """
    p = win_path.replace("\\", "/")
    # Already looks POSIX
    if p.startswith("/"):
        return p
    # Drive letter like C:/ or C:\
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        rest = p[2:]
        if rest.startswith("/"):
            rest = rest[1:]
        return f"/mnt/{drive}/{rest}"
    return p


# Normalize model / output paths based on environment
if running_under_wsl():
    MODEL_LOCAL = Path(to_wsl_path(RAW_MODEL_PATH))
    OUTPUT_DIR = Path(to_wsl_path(RAW_OUTPUT_PATH))
else:
    # running on Windows/native Python
    MODEL_LOCAL = Path(RAW_MODEL_PATH)
    OUTPUT_DIR = Path(RAW_OUTPUT_PATH)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Useful debug: print versions
def print_env():
    try:
        import torch, transformers
        print(f"python: {sys.version.splitlines()[0]}")
        print(f"torch: {torch.__version__} cuda: {getattr(torch.version, 'cuda', None)}")
        print(f"transformers: {transformers.__version__}")
    except Exception as e:
        print("Failed printing environment:", e)


print("=== Environment ===")
print_env()
print("MODEL_LOCAL:", MODEL_LOCAL)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("===================")


# Attempt 1: GPTQModel quantization (preferred for benchmark parity)
try:
    print("\n>>> Attempting GPTQModel-based 8-bit quantization (preferred).")
    # Try to import GPTQModel from site-packages first, then local clone if present
    GPTQ_IMPORTED = False
    try:
        from gptqmodel import GPTQModel
        GPTQ_IMPORTED = True
        print("Imported GPTQModel from site-packages.")
    except Exception as e_site:
        # try local clone path (common convention: repo placed beside this project)
        local = Path(__file__).resolve().parents[1] / "GPTQModel"
        if local.exists():
            sys.path.insert(0, str(local))
            try:
                from gptqmodel import GPTQModel
                GPTQ_IMPORTED = True
                print("Imported GPTQModel from local GPTQModel clone.")
            except Exception:
                # fall-through
                pass
        if not GPTQ_IMPORTED:
            raise RuntimeError("Could not import GPTQModel (site-packages or local).")

    # Verify model dir exists
    if not MODEL_LOCAL.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_LOCAL}")

    # Try multiple GPTQModel call signatures
    model = None
    tried_signatures = []

    # signature A
    try:
        tried_signatures.append("from_pretrained(model_id, device='cuda', quantize='8bit')")
        model = GPTQModel.from_pretrained(str(MODEL_LOCAL), device="cuda", quantize="8bit", use_triton=False)
        print("GPTQModel: used signature A")
    except Exception as e:
        print("signature A failed:", e)

    # signature B
    if model is None:
        try:
            tried_signatures.append("from_pretrained(model_id, quantize_config={'bits':8}, device='cuda')")
            qc = {"bits": 8, "group_size": 128}
            model = GPTQModel.from_pretrained(str(MODEL_LOCAL), quantize_config=qc, device="cuda")
            print("GPTQModel: used signature B")
        except Exception as e:
            print("signature B failed:", e)

    # signature C: load then quantize
    if model is None:
        try:
            tried_signatures.append("load then model.quantize(bits=8, ...)")
            if hasattr(GPTQModel, "load_from_pretrained"):
                base = GPTQModel.load_from_pretrained(str(MODEL_LOCAL))
            else:
                base = None
            if base is None:
                raise RuntimeError("no load_from_pretrained available on GPTQModel")
            base.quantize(bits=8, group_size=128)
            model = base
            print("GPTQModel: used signature C")
        except Exception as e:
            print("signature C failed:", e)

    if model is None:
        raise RuntimeError(f"GPTQModel 8-bit attempt failed (tried signatures: {tried_signatures}).")

    # Save quantized model to OUTPUT_DIR
    try:
        if hasattr(model, "save_quantized"):
            model.save_quantized(str(OUTPUT_DIR))
            print("Saved quantized model using model.save_quantized()")
        else:
            # some GPTQModel variants integrate with HF; try save_pretrained
            model.save_pretrained(str(OUTPUT_DIR))
            print("Saved quantized model using save_pretrained()")
        print("✅ GPTQModel 8-bit quantization finished and saved to:", OUTPUT_DIR)
        sys.exit(0)
    except Exception as e:
        print("Error while saving GPTQModel quantized model:", e)
        # continue to bitsandbytes fallback

except Exception:
    print("GPTQModel path failed. Error (traceback follows):")
    traceback.print_exc()


# Attempt 2: Fallback — bitsandbytes 8-bit (transformers + BitsAndBytesConfig)
try:
    print("\n>>> Falling back to bitsandbytes-style 8-bit (transformers + BitsAndBytesConfig).")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from transformers import BitsAndBytesConfig
    except Exception:
        # Provide a very small shim for older HF versions
        class BitsAndBytesConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0, llm_int8_enable_fp32_cpu_offload=True)
    print("BitsAndBytesConfig:", getattr(bnb_cfg, "__dict__", str(bnb_cfg)))

    print("Loading tokenizer (local files only)...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_LOCAL), trust_remote_code=True, local_files_only=True)

    print("Loading model with 8-bit quantization (local files only)...")
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_LOCAL),
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    print("Saving 8-bit model to disk...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("✅ bitsandbytes 8-bit model saved to:", OUTPUT_DIR)
    sys.exit(0)

except Exception:
    print("bitsandbytes fallback failed. Error (traceback follows):")
    traceback.print_exc()


# If we reach here, everything failed
print("\n❌ All attempts failed. Summary:")
print("- GPTQModel path error (see above).")
print("- bitsandbytes fallback error (see above).")
print("\nRecommendations:")
print("1) Run this in WSL/Linux where building/using bitsandbytes & GPTQ CUDA extensions is more reliable.")
print("2) Ensure CUDA and PyTorch versions match expected prebuilt bitsandbytes/GPTQ binaries.")
print("3) If you want, paste the above tracebacks back here and I will parse the exact failure lines.")
