#!/usr/bin/env python3
"""
Korean VQA Evaluation Script

Evaluates models on Korean VQA test set.
Compares multiple models and generates a report.
"""

import os
import json
import argparse
import torch
from typing import Dict, List, Optional
from tqdm import tqdm
from PIL import Image

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


def load_jsonl(filepath: str, max_samples: int = None) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return data


def extract_question_and_answer(item: Dict, image_root: str) -> Dict:
    """Extract question, answer, and image from a data item."""
    messages = item.get("messages", [])

    question = ""
    answer = ""
    images = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            if role == "user":
                question = content
            elif role == "assistant":
                answer = content
        elif isinstance(content, list):
            text_parts = []
            for c in content:
                if c["type"] == "text":
                    text_parts.append(c["text"])
                elif c["type"] == "image":
                    img_path = c.get("path", "")
                    full_path = os.path.join(image_root, img_path)
                    if os.path.exists(full_path):
                        images.append(full_path)

            text = " ".join(text_parts)
            if role == "user":
                question = text
            elif role == "assistant":
                answer = text

    return {
        "question": question,
        "answer": answer,
        "images": images,
    }


def generate_response(
    model,
    processor,
    question: str,
    images: List[str],
    max_new_tokens: int = 512,
) -> str:
    """Generate response from model."""

    # Prepare messages
    content = []
    for img_path in images:
        content.append({"type": "image", "image": img_path})
    content.append({"type": "text", "text": question})

    messages = [{"role": "user", "content": content}]

    # Process input
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Load images
    pil_images = [Image.open(p).convert("RGB") for p in images] if images else None

    if pil_images:
        inputs = processor(text=text, images=pil_images, return_tensors="pt")
    else:
        inputs = processor(text=text, return_tensors="pt")

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    # Decode
    response = processor.decode(outputs[0], skip_special_tokens=True)

    # Extract assistant response
    if "assistant" in response.lower():
        response = response.split("assistant")[-1].strip()

    return response


def simple_accuracy(pred: str, gold: str) -> float:
    """Simple exact match accuracy."""
    pred = pred.strip().lower()
    gold = gold.strip().lower()
    return 1.0 if pred == gold else 0.0


def contains_accuracy(pred: str, gold: str) -> float:
    """Check if prediction contains key parts of gold answer."""
    pred = pred.strip().lower()
    gold = gold.strip().lower()

    # Extract key answer (after "Answer:" if present)
    if "answer:" in gold:
        gold = gold.split("answer:")[-1].strip()

    # Check if gold is contained in pred
    return 1.0 if gold in pred else 0.0


def evaluate_model(
    model_path: str,
    test_data_path: str,
    image_root: str,
    max_samples: int = None,
    output_path: str = None,
) -> Dict:
    """
    Evaluate a model on test data.

    Returns:
        Dictionary with evaluation results
    """
    print(f"\nEvaluating: {model_path}")

    # Load model
    print("Loading model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model.eval()

    # Load test data
    print(f"Loading test data from {test_data_path}...")
    test_data = load_jsonl(test_data_path, max_samples)
    print(f"Loaded {len(test_data)} samples")

    # Evaluate
    results = []
    exact_matches = 0
    contains_matches = 0

    for item in tqdm(test_data, desc="Evaluating"):
        extracted = extract_question_and_answer(item, image_root)

        if not extracted["question"] or not extracted["images"]:
            continue

        try:
            pred = generate_response(
                model, processor,
                extracted["question"],
                extracted["images"],
            )

            exact = simple_accuracy(pred, extracted["answer"])
            contains = contains_accuracy(pred, extracted["answer"])

            exact_matches += exact
            contains_matches += contains

            results.append({
                "question": extracted["question"],
                "gold": extracted["answer"],
                "pred": pred,
                "exact_match": exact,
                "contains_match": contains,
            })

        except Exception as e:
            print(f"Error: {e}")
            continue

    n = len(results)
    metrics = {
        "model": model_path,
        "n_samples": n,
        "exact_match_accuracy": exact_matches / n if n > 0 else 0,
        "contains_accuracy": contains_matches / n if n > 0 else 0,
    }

    print(f"\nResults for {model_path}:")
    print(f"  Samples: {n}")
    print(f"  Exact Match: {metrics['exact_match_accuracy']:.4f}")
    print(f"  Contains Match: {metrics['contains_accuracy']:.4f}")

    # Save results
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "metrics": metrics,
                "results": results[:100],  # Save first 100 examples
            }, f, indent=2, ensure_ascii=False)
        print(f"  Results saved to {output_path}")

    # Clean up
    del model
    torch.cuda.empty_cache()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on Korean VQA")

    parser.add_argument(
        "--models", "-m",
        type=str,
        nargs="+",
        required=True,
        help="Paths to models to evaluate"
    )
    parser.add_argument(
        "--test-data", "-t",
        type=str,
        default="/NetDisk/juyeon/AdaMMS/data/korean_vqa_tiny/test.jsonl",
        help="Path to test data"
    )
    parser.add_argument(
        "--image-root", "-i",
        type=str,
        default="/NetDisk/juyeon/AdaMMS/data/korean_vqa_tiny",
        help="Root directory for images"
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=100,
        help="Maximum samples to evaluate (default: 100)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="/NetDisk/juyeon/AdaMMS/eval_results/korean_vqa",
        help="Output directory for results"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Korean VQA Evaluation")
    print("=" * 70)

    all_metrics = []

    for model_path in args.models:
        model_name = os.path.basename(model_path)
        output_path = os.path.join(args.output_dir, f"{model_name}_results.json")

        metrics = evaluate_model(
            model_path=model_path,
            test_data_path=args.test_data,
            image_root=args.image_root,
            max_samples=args.max_samples,
            output_path=output_path,
        )
        all_metrics.append(metrics)

    # Print comparison
    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    print(f"{'Model':<40} {'Exact':<12} {'Contains':<12}")
    print("-" * 70)

    for m in all_metrics:
        model_name = os.path.basename(m['model'])
        print(f"{model_name:<40} {m['exact_match_accuracy']:<12.4f} {m['contains_accuracy']:<12.4f}")

    # Save comparison
    comparison_path = os.path.join(args.output_dir, "comparison.json")
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nComparison saved to {comparison_path}")


if __name__ == "__main__":
    main()
