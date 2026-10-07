"""
SpaceText SLM Evaluation
=========================
Evaluates the fine-tuned SpaceText model against the held-out test split
using three complementary metrics:

  1. ROUGE-L   — measures output overlap (good for summarization tasks)
  2. BERTScore — semantic similarity using bert-base embeddings
  3. Taxonomy Accuracy — exact-match on domain classification output

Usage:
    # Evaluate the merged model (recommended)
    python -m spacetext.training.evaluate

    # Evaluate a specific checkpoint
    python -m spacetext.training.evaluate --model-path ./checkpoints/checkpoint-500

    # Quick eval on 200 examples
    python -m spacetext.training.evaluate --max-samples 200
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("spacetext.training.evaluate")

DEFAULT_MODEL_PATH = "./checkpoints/spacetext-qwen2.5-3b-merged"
TEST_DATA_PATH     = "./training_data/test.jsonl"
RESULTS_PATH       = "./training_data/eval_results.json"
MAX_NEW_TOKENS     = 256
BATCH_SIZE         = 8


def _check_deps():
    missing = []
    for pkg in ["transformers", "torch", "rouge_score", "bert_score", "evaluate"]:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            f"Missing packages: {missing}. Run: pip install {' '.join(missing)}"
        )


def load_test_data(path: str, max_samples: int = -1) -> List[Dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if max_samples > 0:
        records = records[:max_samples]
    logger.info(f"Loaded {len(records):,} test examples")
    return records


def load_model(model_path: str):
    """Load the fine-tuned model for inference."""
    try:
        from unsloth import FastLanguageModel
        logger.info(f"Loading with Unsloth: {model_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path, max_seq_length=2048, load_in_4bit=True
        )
        FastLanguageModel.for_inference(model)
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        logger.info(f"Loading with Transformers: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
        model.eval()
    return model, tokenizer


def generate_predictions(
    model, tokenizer, examples: List[Dict], batch_size: int = 8
) -> Tuple[List[str], List[str]]:
    """Run inference on all test examples and return (predictions, references)."""
    import torch

    predictions, references = [], []

    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        prompts = [ex["text"].split("<|assistant|>")[0] + "<|assistant|>\n" for ex in batch]
        refs    = [ex["output"] for ex in batch]

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=1792
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens   = MAX_NEW_TOKENS,
                do_sample        = False,
                temperature      = 1.0,
                repetition_penalty = 1.1,
                pad_token_id     = tokenizer.eos_token_id,
            )

        # Decode only the generated tokens (strip the prompt)
        for j, out in enumerate(outputs):
            prompt_len = inputs["input_ids"].shape[1]
            gen_tokens = out[prompt_len:]
            pred = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            predictions.append(pred)
            references.append(refs[j])

        if (i // batch_size) % 5 == 0:
            logger.info(f"Evaluated {min(i + batch_size, len(examples)):,}/{len(examples):,} examples")

    return predictions, references


def compute_metrics(predictions: List[str], references: List[str]) -> Dict:
    """Compute ROUGE-L, BERTScore, and taxonomy accuracy."""
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score_fn

    results = {}

    # ── ROUGE-L ───────────────────────────────────────────────────────────────
    logger.info("Computing ROUGE-L...")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [scorer.score(ref, pred)["rougeL"].fmeasure
                    for pred, ref in zip(predictions, references)]
    results["rougeL"] = {
        "mean": round(sum(rouge_scores) / len(rouge_scores), 4),
        "min":  round(min(rouge_scores), 4),
        "max":  round(max(rouge_scores), 4),
    }
    logger.info(f"  ROUGE-L (mean): {results['rougeL']['mean']:.4f}")

    # ── BERTScore ─────────────────────────────────────────────────────────────
    logger.info("Computing BERTScore (this may take a minute)...")
    P, R, F1 = bert_score_fn(
        predictions, references, lang="en", verbose=False, batch_size=32
    )
    results["bertscore"] = {
        "precision": round(P.mean().item(), 4),
        "recall":    round(R.mean().item(), 4),
        "f1":        round(F1.mean().item(), 4),
    }
    logger.info(f"  BERTScore F1:   {results['bertscore']['f1']:.4f}")

    # ── Taxonomy Accuracy (for classification tasks) ───────────────────────────
    logger.info("Computing taxonomy accuracy...")
    correct = 0
    for pred, ref in zip(predictions, references):
        # Check if the correct taxonomy label appears anywhere in the prediction
        if ref.split("\n")[0].strip().lower() in pred.lower():
            correct += 1
    results["taxonomy_accuracy"] = round(correct / len(predictions), 4)
    logger.info(f"  Taxonomy Acc:   {results['taxonomy_accuracy']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate SpaceText fine-tuned model")
    parser.add_argument("--model-path",  default=DEFAULT_MODEL_PATH, help="Path to merged model or checkpoint")
    parser.add_argument("--test-data",   default=TEST_DATA_PATH,     help="Path to test.jsonl")
    parser.add_argument("--results-out", default=RESULTS_PATH,       help="Path to save results JSON")
    parser.add_argument("--max-samples", type=int, default=-1,       help="Limit number of test examples (-1 = all)")
    parser.add_argument("--batch-size",  type=int, default=BATCH_SIZE, help="Inference batch size")
    args = parser.parse_args()

    _check_deps()

    if not Path(args.model_path).exists():
        logger.error(f"Model not found: {args.model_path}")
        logger.error("Run first: python -m spacetext.training.finetune")
        return

    if not Path(args.test_data).exists():
        logger.error(f"Test data not found: {args.test_data}")
        logger.error("Run first: python -m spacetext.training.prepare_dataset")
        return

    # Load data and model
    examples = load_test_data(args.test_data, max_samples=args.max_samples)
    model, tokenizer = load_model(args.model_path)

    # Generate predictions
    predictions, references = generate_predictions(model, tokenizer, examples, args.batch_size)

    # Compute metrics
    metrics = compute_metrics(predictions, references)

    # Save results
    output = {
        "model_path":     args.model_path,
        "num_examples":   len(examples),
        "metrics":        metrics,
        "sample_outputs": [
            {"instruction": ex["instruction"], "reference": ref, "prediction": pred}
            for ex, ref, pred in zip(examples[:5], references[:5], predictions[:5])
        ],
    }
    with open(args.results_out, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info(f"  ROUGE-L:          {metrics['rougeL']['mean']:.4f}")
    logger.info(f"  BERTScore F1:     {metrics['bertscore']['f1']:.4f}")
    logger.info(f"  Taxonomy Acc:     {metrics['taxonomy_accuracy']:.4f}")
    logger.info(f"  Results saved  → {args.results_out}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
