import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Base model path
base_model_path = "D:/Phi3-EdgeQuant-Agent/models/microsoft/Phi-3-mini-4k-instruct"

# Save path
quantized_model_path = "D:/Phi3-EdgeQuant-Agent/models/quantized/Phi-3-mini-4k-instruct-1bit"
os.makedirs(quantized_model_path, exist_ok=True)

# ⚠ Mock: using float16 to simulate 1-bit
# Real 1-bit needs custom CUDA kernels (not supported in HF yet)
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype="float16",   # Load lightweight but valid dtype
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto"
)

print(f"✅ Saving 'mock' 1-bit quantized model (actually float16) -> {quantized_model_path}")
model.save_pretrained(quantized_model_path)
tokenizer.save_pretrained(quantized_model_path)
