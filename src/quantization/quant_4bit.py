# D:/Phi3-EdgeQuant-Agent/src/quantization/quant_4bit.py
import sys
import types
import importlib.util
import os

# -----------------------------
# ROBUST MOCK FOR TENSORFLOW
# -----------------------------
tf_mock = types.ModuleType("tensorflow")
tf_mock.__spec__ = importlib.util.spec_from_loader("tensorflow", loader=None)
sys.modules["tensorflow"] = tf_mock
sys.modules["tensorflow.python"] = types.ModuleType("tensorflow.python")
sys.modules["tensorflow.python"].__spec__ = importlib.util.spec_from_loader("tensorflow.python", loader=None)

# -----------------------------
# ACTUAL IMPORTS
# -----------------------------
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

# Paths
BASE_MODEL_PATH = "D:/Phi3-EdgeQuant-Agent/models/microsoft/Phi-3-mini-4k-instruct"
OUTPUT_PATH = "D:/Phi3-EdgeQuant-Agent/models/quantized/Phi-3-mini-4k-instruct-4bit-gptq"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load tokenizer
print(f"🔄 Loading tokenizer from {BASE_MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

# Load calibration dataset (raw text)
print("📚 Loading calibration dataset (wikitext-2, first 128 samples)...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:128]")
calib_texts = dataset["text"][:128]  # list of raw strings

# GPTQ quantization config
quant_config = GPTQConfig(
    bits=4,
    group_size=128,
    desc_act=False,
    dataset=calib_texts,  # raw strings for GPTQ
    tokenizer=tokenizer
)

# Load model with GPTQ quantization
print("⚙️ Applying GPTQ quantization...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True,
)

# Save quantized model
print(f"💾 Saving quantized model to {OUTPUT_PATH}")
model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print("✅ 4-bit GPTQ quantization completed successfully.")
