import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import logging

logging.set_verbosity_error()

print("🔍 Loading 4-bit Quantized Model...")

# Use your local quantized model path
model_path = "models/quantized/Phi-3-mini-4k-instruct-4bit"

# Define 4-bit quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=quant_config,
    trust_remote_code=True
)

# Prompt for testing
prompt = "Summarize the following edge sensor logs: Temperature 32.4°C, Humidity 45%, CO2 levels stable."

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(model.device)
attention_mask = inputs["attention_mask"].to(model.device)

# Avoid size mismatch in generation
if input_ids.shape[1] % 2 != 0:
    pad_token = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids = torch.cat([input_ids, torch.tensor([[pad_token]]).to(model.device)], dim=1)
    attention_mask = torch.cat([attention_mask, torch.tensor([[0]]).to(model.device)], dim=1)

# Run inference
outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=80,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=40
)

# Print result
print("\n📋 Response:\n" + tokenizer.decode(outputs[0], skip_special_tokens=True))
