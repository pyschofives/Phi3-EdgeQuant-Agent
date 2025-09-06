from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from dotenv import load_dotenv
load_dotenv()


print("🔍 Loading 4-bit Quantized Model...")

model = AutoModelForCausalLM.from_pretrained(
    "D:/Phi3-EdgeQuant-Agent/models/quantized/Phi-3-mini-4k-instruct-4bit",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    ),
    device_map="auto",
    trust_remote_code=True,   # This is key to load modeling_phi3.py from HF
    revision="main",          # Optional, ensure it fetches latest
)

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",  # Keep tokenizer ref to remote repo
    trust_remote_code=True
)

prompt = "Explain the benefits of 4-bit quantization for edge LLMs."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
