"""
SpaceText SLM Fine-Tuning — QLoRA with Unsloth
================================================
Fine-tunes Qwen2.5-3B-Instruct on the SpaceText aerospace instruction dataset
using 4-bit QLoRA quantization via Unsloth (2x speed, 60% less VRAM).

This script runs Phase 2 of the SpaceText pipeline:
  Phase 1: Data Acquisition (acquisition/pipeline.py)  ← DONE ✓
  Phase 2: SLM Fine-Tuning  (training/finetune.py)     ← THIS FILE
  Phase 3: Quantization     (training/quantize.py)
  Phase 4: Multi-Agent API  (api/)

Model Choice: Qwen2.5-3B-Instruct
  - Best performance/size trade-off for aerospace technical text
  - Fits in 8GB VRAM with 4-bit QLoRA
  - Strong multi-lingual support (handles ISRO/DRDO Hindi technical terms)
  - Native ChatML format matches our Alpaca ChatML-style dataset

Hardware Requirements:
  - GPU: NVIDIA with ≥8GB VRAM (RTX 3070 / T4 / A10G)
  - RAM: ≥16GB system RAM
  - Storage: ~15GB (model weights + adapter)

Usage:
    # Full fine-tuning (recommended)
    python -m spacetext.training.finetune

    # Quick smoke-test with 100 examples
    python -m spacetext.training.finetune --max-steps 50 --batch-size 1

    # Resume from checkpoint
    python -m spacetext.training.finetune --resume-from ./checkpoints/checkpoint-500
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("spacetext.training.finetune")

# ── Constants ────────────────────────────────────────────────────────────────
BASE_MODEL        = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_OUT_DIR   = "./checkpoints/spacetext-qwen2.5-3b-qlora"
MERGED_OUT_DIR    = "./checkpoints/spacetext-qwen2.5-3b-merged"
TRAIN_DATA        = "./training_data/train.jsonl"
VAL_DATA          = "./training_data/val.jsonl"
MAX_SEQ_LEN       = 2048   # max token length per example
LORA_RANK         = 16     # LoRA rank — 16 is sweet spot for 3B model
LORA_ALPHA        = 32     # LoRA scaling (2× rank is standard)
LORA_DROPOUT      = 0.05
TARGET_MODULES    = [      # Qwen2.5 attention + FFN projection layers
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
TRAIN_EPOCHS      = 3
BATCH_SIZE        = 2      # per-device — increase if you have >8GB VRAM
GRAD_ACCUM        = 4      # effective batch = BATCH_SIZE × GRAD_ACCUM = 8
LEARNING_RATE     = 2e-4
WARMUP_RATIO      = 0.03
LR_SCHEDULER      = "cosine"
WEIGHT_DECAY      = 0.01
LOGGING_STEPS     = 25
SAVE_STEPS        = 250
EVAL_STEPS        = 250


def _check_dependencies():
    """Verify all ML dependencies are installed before starting."""
    missing = []
    for pkg in ["unsloth", "transformers", "peft", "trl", "datasets", "torch"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        logger.error(
            f"Missing dependencies: {missing}\n"
            f"Please uncomment the Training section in requirements.txt and run:\n"
            f"  pip install -r requirements.txt\n"
            f"Or install individually:\n"
            f"  pip install unsloth transformers peft trl datasets torch accelerate bitsandbytes"
        )
        sys.exit(1)
    logger.info("All dependencies present ✓")


def _load_jsonl(path: str):
    """Load a JSONL file as a list of dicts."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def prepare_model_and_tokenizer(max_seq_len: int, load_in_4bit: bool = True):
    """Load Qwen2.5-3B with 4-bit QLoRA via Unsloth."""
    from unsloth import FastLanguageModel

    logger.info(f"Loading base model: {BASE_MODEL} (4-bit={load_in_4bit})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name      = BASE_MODEL,
        max_seq_length  = max_seq_len,
        dtype           = None,           # auto-detect (bfloat16 on Ampere+)
        load_in_4bit    = load_in_4bit,
    )

    logger.info(f"Applying LoRA adapters (rank={LORA_RANK}, α={LORA_ALPHA})")
    model = FastLanguageModel.get_peft_model(
        model,
        r                   = LORA_RANK,
        target_modules      = TARGET_MODULES,
        lora_alpha          = LORA_ALPHA,
        lora_dropout        = LORA_DROPOUT,
        bias                = "none",
        use_gradient_checkpointing = "unsloth",  # saves 30% VRAM
        random_state        = 42,
        use_rslora          = True,   # Rank-Stabilised LoRA — better convergence
        loftq_config        = None,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model, tokenizer


def build_dataset(tokenizer, train_path: str, val_path: str, max_seq_len: int):
    """Convert JSONL splits to HuggingFace Dataset with tokenized 'text' field."""
    from datasets import Dataset

    def _to_hf(path):
        records = _load_jsonl(path)
        # The 'text' field was pre-built by prepare_dataset.py in ChatML format
        return Dataset.from_list([{"text": r["text"]} for r in records])

    logger.info(f"Loading train split: {train_path}")
    train_ds = _to_hf(train_path)
    logger.info(f"Loading val split:   {val_path}")
    val_ds   = _to_hf(val_path)

    logger.info(f"Train: {len(train_ds):,} examples | Val: {len(val_ds):,} examples")
    return train_ds, val_ds


def train(
    model,
    tokenizer,
    train_ds,
    val_ds,
    output_dir: str,
    max_seq_len: int,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    max_steps: int = -1,
    resume_from: str = None,
):
    """Configure and run SFTTrainer."""
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainingArguments

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir                  = output_dir,
        num_train_epochs            = epochs,
        max_steps                   = max_steps,             # -1 = use epochs
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        gradient_accumulation_steps = grad_accum,
        warmup_ratio                = WARMUP_RATIO,
        learning_rate               = LEARNING_RATE,
        lr_scheduler_type           = LR_SCHEDULER,
        weight_decay                = WEIGHT_DECAY,
        fp16                        = False,
        bf16                        = True,                  # use bfloat16 (Ampere+)
        optim                       = "adamw_8bit",          # 8-bit Adam saves VRAM
        logging_steps               = LOGGING_STEPS,
        save_steps                  = SAVE_STEPS,
        evaluation_strategy         = "steps",
        eval_steps                  = EVAL_STEPS,
        save_total_limit            = 3,                     # keep last 3 checkpoints
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        report_to                   = "none",                # set "tensorboard" if needed
        seed                        = 42,
        # SFT-specific
        dataset_text_field          = "text",
        max_seq_length              = max_seq_len,
        packing                     = True,                  # pack short examples for efficiency
    )

    trainer = SFTTrainer(
        model           = model,
        tokenizer       = tokenizer,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        args            = training_args,
    )

    logger.info("=" * 60)
    logger.info("Starting SpaceText QLoRA fine-tuning")
    logger.info(f"  Base model  : {BASE_MODEL}")
    logger.info(f"  Output dir  : {output_dir}")
    logger.info(f"  Epochs      : {epochs}")
    logger.info(f"  Batch size  : {batch_size} × {grad_accum} grad_accum = {batch_size * grad_accum} effective")
    logger.info(f"  Max seq len : {max_seq_len}")
    logger.info(f"  LoRA rank   : {LORA_RANK}")
    logger.info("=" * 60)

    trainer.train(resume_from_checkpoint=resume_from)

    logger.info("Training complete. Saving LoRA adapter...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"LoRA adapter saved → {output_dir}")

    return trainer


def merge_and_save(model, tokenizer, merged_dir: str):
    """
    Merge LoRA weights back into the base model and save as a standalone
    HuggingFace model (no adapter dependencies needed at inference time).
    """
    from unsloth import FastLanguageModel

    logger.info("Merging LoRA adapter into base model weights...")
    model.save_pretrained_merged(
        merged_dir,
        tokenizer,
        save_method="merged_16bit",   # merge to fp16 for inference
    )
    logger.info(f"Merged model saved → {merged_dir}")

    # Also save a 4-bit GGUF for local inference with llama.cpp / Ollama
    gguf_dir = merged_dir + "_gguf"
    logger.info("Saving GGUF (Q4_K_M) for Ollama/llama.cpp...")
    model.save_pretrained_gguf(
        gguf_dir,
        tokenizer,
        quantization_method="q4_k_m",
    )
    logger.info(f"GGUF model saved → {gguf_dir}")


def main():
    parser = argparse.ArgumentParser(description="SpaceText SLM Fine-Tuning (QLoRA)")
    parser.add_argument("--base-model",    default=BASE_MODEL,       help="HuggingFace model ID")
    parser.add_argument("--train-data",    default=TRAIN_DATA,       help="Path to train.jsonl")
    parser.add_argument("--val-data",      default=VAL_DATA,         help="Path to val.jsonl")
    parser.add_argument("--output-dir",    default=ADAPTER_OUT_DIR,  help="LoRA adapter output directory")
    parser.add_argument("--merged-dir",    default=MERGED_OUT_DIR,   help="Merged model output directory")
    parser.add_argument("--epochs",        type=int,   default=TRAIN_EPOCHS, help="Training epochs")
    parser.add_argument("--batch-size",    type=int,   default=BATCH_SIZE,   help="Per-device batch size")
    parser.add_argument("--grad-accum",    type=int,   default=GRAD_ACCUM,   help="Gradient accumulation steps")
    parser.add_argument("--max-steps",     type=int,   default=-1,           help="Max training steps (-1 = use epochs)")
    parser.add_argument("--max-seq-len",   type=int,   default=MAX_SEQ_LEN,  help="Max sequence length")
    parser.add_argument("--no-4bit",       action="store_true",              help="Disable 4-bit QLoRA (use full precision)")
    parser.add_argument("--skip-merge",    action="store_true",              help="Skip merging LoRA into base model")
    parser.add_argument("--resume-from",   default=None,                     help="Resume from checkpoint path")
    args = parser.parse_args()

    # Pre-flight checks
    _check_dependencies()
    if not Path(args.train_data).exists():
        logger.error(f"Training data not found: {args.train_data}")
        logger.error("Run first: python -m spacetext.training.prepare_dataset")
        sys.exit(1)

    # Load model
    model, tokenizer = prepare_model_and_tokenizer(
        max_seq_len=args.max_seq_len,
        load_in_4bit=not args.no_4bit,
    )

    # Build dataset
    train_ds, val_ds = build_dataset(tokenizer, args.train_data, args.val_data, args.max_seq_len)

    # Train
    trainer = train(
        model       = model,
        tokenizer   = tokenizer,
        train_ds    = train_ds,
        val_ds      = val_ds,
        output_dir  = args.output_dir,
        max_seq_len = args.max_seq_len,
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        grad_accum  = args.grad_accum,
        max_steps   = args.max_steps,
        resume_from = args.resume_from,
    )

    # Merge + export
    if not args.skip_merge:
        merge_and_save(model, tokenizer, args.merged_dir)

    logger.info("Phase 2 complete! Next step: python -m spacetext.training.evaluate")


if __name__ == "__main__":
    main()
