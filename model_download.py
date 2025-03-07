from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch
import os
from tqdm import tqdm

# Set environment variables for better CPU performance
os.environ["OMP_NUM_THREADS"] = str(max(1, os.cpu_count() // 2))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_name = "mistralai/Mistral-7B-v0.1"

# 1. Load tokenizer separately (usually works fine)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Tokenizer loaded successfully!")

# 2. Try loading a much smaller model for testing
print("For testing, let's use a much smaller model...")
small_model_name = "facebook/opt-125m"  # ~125M parameters vs 7B

print(f"Loading small model: {small_model_name}")
small_model = AutoModelForCausalLM.from_pretrained(
    small_model_name,
    device_map="cpu",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)
print("Small model loaded successfully!")

# 3. If you absolutely need Mistral-7B on CPU, try this with caution:
print("\nIf you still want to try loading Mistral-7B, here's how:")
print("1. Make sure you have at least 32GB of RAM")
print("2. Set up disk offloading:")
print("""
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    device_map="cpu",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    offload_folder="offload_folder"
)
""")
print("3. Be prepared to wait - this can take 30+ minutes on some systems")