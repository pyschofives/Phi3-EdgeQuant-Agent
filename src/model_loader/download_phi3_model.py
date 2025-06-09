import os
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM

# Temp cache directory
hf_cache_dir = "D:/Phi3-EdgeQuant-Agent/models"

# Clean final save path
clean_model_dir = os.path.join(hf_cache_dir, "microsoft", "Phi-3-mini-4k-instruct")
os.makedirs(clean_model_dir, exist_ok=True)

# Download using cache
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    trust_remote_code=True,
    cache_dir=hf_cache_dir
)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    trust_remote_code=True,
    cache_dir=hf_cache_dir
)

# Find actual downloaded path
import huggingface_hub
repo_path = huggingface_hub.snapshot_download(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    cache_dir=hf_cache_dir,
    local_dir=clean_model_dir,
    local_dir_use_symlinks=False
)

print(f"Model saved cleanly at: {clean_model_dir}")
