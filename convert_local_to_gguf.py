import os
from unsloth import FastLanguageModel

# Point directly to the folder we just downloaded on the D drive
model_dir = "./models/param-raw"
output_dir = "./models/param-1-2.9b-instruct"

print(f"[*] Loading raw weights from local D drive folder: {model_dir}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_dir,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False
)

print("[*] Starting local conversion to q4_k_m GGUF...")

# Export to GGUF
model.save_pretrained_gguf(
    output_dir, 
    tokenizer, 
    quantization_method="q4_k_m"
)

print(f"[*] Done! Your model is ready for summarize_param.py")
