import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Local model path (original full precision)
base_model_path = "D:/Phi3-EdgeQuant-Agent/models/microsoft/Phi-3-mini-4k-instruct"

# Quantized model save path
quantized_model_path = "D:/Phi3-EdgeQuant-Agent/models/quantized/Phi-3-mini-4k-instruct-8bit"
os.makedirs(quantized_model_path, exist_ok=True)

# 8-bit quantization config using bitsandbytes
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=None,
    llm_int8_enable_fp32_cpu_offload=True
)

# Load tokenizer (no quant here)
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    trust_remote_code=True
)

# Load 8-bit quantized model
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Save model and tokenizer to quantized path
print(f"✅ Saving quantized model to: {quantized_model_path}")
model.save_pretrained(quantized_model_path)
tokenizer.save_pretrained(quantized_model_path)

print("🚀 Phi-3-mini quantized to 8-bit and saved successfully.")
