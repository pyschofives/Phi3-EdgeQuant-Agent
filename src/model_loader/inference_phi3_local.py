import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Local path
model_path = "D:/Phi3-EdgeQuant-Agent/models/microsoft/Phi-3-mini-4k-instruct"

# Load locally
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
model.eval()

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Sample prompt
prompt = "Explain why the sky is blue in a simple way for a 10-year-old."

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
    )

# Decode
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n🧠 Model Response:\n", response)
