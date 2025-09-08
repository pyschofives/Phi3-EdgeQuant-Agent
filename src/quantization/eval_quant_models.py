# src/quantization/eval_quant_models.py
"""
Enhanced evaluation script for quantized models with optional Weights & Biases (wandb)
- Prints model existence and sizes
- Attempts multiple load strategies (GPTQModel, transformers+bitsandbytes, plain transformers)
- Runs a short generation and records timings / GPU peak memory
- Logs benchmark metrics and artifacts to wandb (if installed and env configured)
- Writes human-readable benchmark.md and per-model output files to ./outputs/

Environment variables (optional, used when wandb is available):
WANDB_PROJECT=Phi3-EdgeQuant-Agent
WANDB_ENTITY=stifler
WANDB_DIR=./wandb_logs
WANDB_MODE=online

Save location for outputs: ./outputs/
"""

import os
import time
import shutil
import json
from pathlib import Path
from datetime import datetime

# --- Utilities -----------------------------------------------------------------

def human_size(path: Path):
    try:
        total = 0
        if path.is_file():
            total = path.stat().st_size
        else:
            for p in path.rglob("*"):
                if p.is_file():
                    total += p.stat().st_size
        for unit in ("B","KB","MB","GB","TB"):
            if total < 1024.0:
                return f"{total:3.2f}{unit}"
            total /= 1024.0
        return f"{total:.2f}PB"
    except Exception:
        return "N/A"


def ensure_outputs_dir(base: Path):
    out = base / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


# --- Configuration -------------------------------------------------------------
print("Found quantized models in D:/Phi3-EdgeQuant-Agent/models/quantized\n")
BASE = Path(r"D:/Phi3-EdgeQuant-Agent/models/quantized")
models = [
    BASE / "Phi-3-mini-4k-instruct-1bit",
    BASE / "Phi-3-mini-4k-instruct-4bit-gptq",
    BASE / "Phi-3-mini-4k-instruct-8bit-gptq",
]

# Prompt to use
PROMPT = "Explain knowledge distillation in AI in simple terms."

# Outputs/benchmarks base
OUT_BASE = Path.cwd()
OUT_DIR = ensure_outputs_dir(OUT_BASE)
BENCHMARK_MD = OUT_DIR / "benchmark.md"
BENCH_JSON = OUT_DIR / "benchmark.json"

# Prepare benchmark outputs
bench_records = []

# --- WandB setup (optional) ---------------------------------------------------
try:
    import wandb
    WANDB_AVAILABLE = True
    # Respect environment variables if set; else allow user defaults through explicit kwargs
    wandb_project = os.environ.get("WANDB_PROJECT", "Phi3-EdgeQuant-Agent")
    wandb_entity = os.environ.get("WANDB_ENTITY", None)
    # set wandb dir if provided
    if os.environ.get("WANDB_DIR"):
        os.environ.setdefault("WANDB_DIR", os.environ.get("WANDB_DIR"))
    print("wandb available; will attempt to log runs if wandb.init succeeds.")
except Exception:
    wandb = None
    WANDB_AVAILABLE = False
    print("wandb not available — continuing without external experiment logging.")


# --- Core: try load and generate -----------------------------------------------

def try_load_and_generate(model_path: Path):
    record = {
        "model_name": model_path.name,
        "path": str(model_path),
        "exists": model_path.exists(),
        "size": human_size(model_path) if model_path.exists() else "N/A",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "success": False,
        "loader": None,
        "load_time_s": None,
        "gen_time_s": None,
        "gpu_peak_bytes": None,
        "output_snippet": None,
        "error": None,
    }

    print(f"\n--- {model_path.name} ---")
    if not model_path.exists():
        print("Path not found, skipping.")
        return record

    # Optionally initialize a wandb run per-model
    run = None
    if WANDB_AVAILABLE:
        try:
            run = wandb.init(project=wandb_project, entity=wandb_entity, name=model_path.name, reinit=True)
            run.log({"model/size": record["size"]})
        except Exception as e:
            print("wandb.init failed, continuing locally:", e)
            run = None

    # Try GPTQModel (if available)
    try:
        from gptqmodel import GPTQModel
        print("Trying to load with GPTQModel.from_pretrained()...")
        start = time.time()
        # try common signatures
        try:
            m = GPTQModel.from_pretrained(str(model_path), device="cuda", quantize="8bit")
            record["loader"] = "GPTQModel.from_pretrained(signature A)"
        except Exception:
            try:
                qc = {"bits": 8}
                m = GPTQModel.from_pretrained(str(model_path), quantize_config=qc, device="cuda")
                record["loader"] = "GPTQModel.from_pretrained(signature B)"
            except Exception:
                if hasattr(GPTQModel, "load_from_pretrained"):
                    m = GPTQModel.load_from_pretrained(str(model_path))
                    record["loader"] = "GPTQModel.load_from_pretrained"
                else:
                    raise RuntimeError("GPTQModel present but no usable loader signature succeeded.")
        load_elapsed = time.time() - start
        record["load_time_s"] = load_elapsed
        # tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        # generate
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        start = time.time()
        out = m.generate(PROMPT, max_new_tokens=120, temperature=0.7, top_p=0.95)
        gen_elapsed = time.time() - start
        record["gen_time_s"] = gen_elapsed
        # decode
        if isinstance(out, str):
            text = out
        else:
            try:
                text = tokenizer.decode(out[0], skip_special_tokens=True)
            except Exception:
                text = str(out)
        record["output_snippet"] = text[:1000]
        import torch
        record["gpu_peak_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        record["success"] = True
        print(f"Loaded via GPTQModel. Time: {gen_elapsed:.3f}s, GPU peak bytes: {record['gpu_peak_bytes']}")

        # save outputs
        save_model_run_outputs(model_path, text, record, run)
        if run:
            run.log(record)
            run.finish()
        return record
    except Exception as e:
        print("GPTQModel load/generate failed:", e)
        record["error"] = str(e)

    # Fallback: transformers + bitsandbytes (8-bit or 4-bit) or plain transformers
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        try:
            from transformers import BitsAndBytesConfig
        except Exception:
            class BitsAndBytesConfig:
                def __init__(self, **kw):
                    self.__dict__.update(kw)

        print("Trying transformers AutoModelForCausalLM.from_pretrained (with bitsandbytes config where possible)...")
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0, llm_int8_enable_fp32_cpu_offload=True)
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        start = time.time()
        model = AutoModelForCausalLM.from_pretrained(str(model_path),
                                                     quantization_config=bnb_cfg,
                                                     device_map="auto",
                                                     trust_remote_code=True)
        load_elapsed = time.time() - start
        record["load_time_s"] = load_elapsed
        record["loader"] = "transformers+bitsandbytes"

        inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        start = time.time()
        outputs = model.generate(**inputs, max_new_tokens=120, temperature=0.7, top_p=0.95)
        gen_elapsed = time.time() - start
        record["gen_time_s"] = gen_elapsed
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        record["output_snippet"] = decoded[:1000]
        record["gpu_peak_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        record["success"] = True
        print(f"transformers+bitsandbytes generation done. Time: {gen_elapsed:.3f}s, GPU peak bytes: {record['gpu_peak_bytes']}")

        save_model_run_outputs(model_path, decoded, record, run)
        if run:
            run.log(record)
            run.finish()
        return record
    except Exception as e:
        print("transformers + bitsandbytes load failed:", e)
        record.setdefault("errors", []).append(str(e))

    # Final fallback: plain transformers (float32/16) load
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("Trying plain AutoModelForCausalLM.from_pretrained (no quant libs)...")
        start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(str(model_path), device_map="auto", trust_remote_code=True)
        load_elapsed = time.time() - start
        record["load_time_s"] = load_elapsed
        record["loader"] = "transformers_plain"

        inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        start = time.time()
        outputs = model.generate(**inputs, max_new_tokens=120)
        gen_elapsed = time.time() - start
        record["gen_time_s"] = gen_elapsed
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        record["output_snippet"] = decoded[:1000]
        record["gpu_peak_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        record["success"] = True
        print(f"Plain transformers generation done. Time: {gen_elapsed:.3f}s, GPU peak bytes: {record['gpu_peak_bytes']}")

        save_model_run_outputs(model_path, decoded, record, run)
        if run:
            run.log(record)
            run.finish()
        return record
    except Exception as e:
        print("Plain transformers load failed too:", e)
        record.setdefault("errors", []).append(str(e))

    # If we reach here, nothing worked
    if run:
        try:
            run.log(record)
        except Exception:
            pass
        try:
            run.finish()
        except Exception:
            pass
    return record


# --- Helpers for saving artifacts ------------------------------------------------

def save_model_run_outputs(model_path: Path, text: str, record: dict, run):
    # writes plain output and updates benchmark.md. Also pushes small artifact to wandb if available
    out_txt = OUT_DIR / f"{model_path.name}_output.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"# Model: {model_path.name}\n# Path: {model_path}\n# Timestamp: {record.get('timestamp')}\n\n")
        f.write(text)

    # append to benchmark.md
    with open(BENCHMARK_MD, "a", encoding="utf-8") as f:
        f.write(f"## {model_path.name}\n")
        f.write(f"- path: {model_path}\n")
        f.write(f"- exists: {record.get('exists')}\n")
        f.write(f"- size: {record.get('size')}\n")
        f.write(f"- loader: {record.get('loader')}\n")
        f.write(f"- load_time_s: {record.get('load_time_s')}\n")
        f.write(f"- gen_time_s: {record.get('gen_time_s')}\n")
        f.write(f"- gpu_peak_bytes: {record.get('gpu_peak_bytes')}\n")
        f.write(f"- success: {record.get('success')}\n")
        f.write("\n```")
        # write a short snippet for quick reading
        snippet = (record.get('output_snippet') or "")[:800]
        f.write(snippet)
        f.write("\n``""\n\n")

    # push artifact to wandb (small text artifact)
    if WANDB_AVAILABLE and run is not None:
        try:
            artifact = wandb.Artifact(f"{model_path.name}-output", type="output")
            artifact.add_file(str(out_txt))
            run.log_artifact(artifact)
        except Exception as e:
            print("wandb artifact upload failed:", e)


# --- Execute across all models -------------------------------------------------

for mp in models:
    rec = try_load_and_generate(mp)
    bench_records.append(rec)

# write full JSON summary
try:
    with open(BENCH_JSON, "w", encoding="utf-8") as f:
        json.dump(bench_records, f, indent=2)
    print(f"Wrote benchmark summary to: {BENCH_JSON}")
except Exception as e:
    print("Failed to write JSON benchmark:", e)

print("\nDone. If any model failed to load, paste the printed stacktrace here and I'll parse the exact fix.")
