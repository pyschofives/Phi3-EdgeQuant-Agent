import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Local model path (full precision)
base_model_path = "D:/Phi3-EdgeQuant-Agent/models/microsoft/Phi-3-mini-4k-instruct"

# Quantized model save path (4-bit)
quantized_model_path = "D:/Phi3-EdgeQuant-Agent/models/quantized/Phi-3-mini-4k-instruct-4bit"
os.makedirs(quantized_model_path, exist_ok=True)

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load tokenizer (not quantized)
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    trust_remote_code=True
)

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Save quantized model
print(f"✅ Saving 4-bit quantized model to: {quantized_model_path}")
model.save_pretrained(quantized_model_path)
tokenizer.save_pretrained(quantized_model_path)

print("🚀 Phi-3-mini quantized to 4-bit and saved successfully.")
