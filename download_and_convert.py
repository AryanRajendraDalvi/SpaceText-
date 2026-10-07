"""
Script to directly download Param-1-Instruct and convert it to Q4_K_M GGUF.
Relies on Unsloth's built-in llama.cpp wrappers to avoid manual C++ compilation on Windows.
"""

from unsloth import FastLanguageModel
import os

model_name = "bharatgenai/Param-1-2.9B-Instruct"
output_dir = "./models/param-1-2.9b-instruct"

print(f"[*] Downloading raw weights for {model_name}...")

# Load the raw model from HuggingFace
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False
)

print("[*] Model downloaded successfully.")
print("[*] Starting llama.cpp GGUF conversion to q4_k_m... (This might take a few minutes)")

# Automatically export to GGUF using Unsloth's wrapper
model.save_pretrained_gguf(
    output_dir, 
    tokenizer, 
    quantization_method="q4_k_m"
)

print(f"[*] Done! Your GGUF file is ready in {output_dir}")
