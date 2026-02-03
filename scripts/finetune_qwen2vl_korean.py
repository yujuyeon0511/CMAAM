#!/usr/bin/env python3
"""
Qwen2-VL Korean Fine-tuning Script

Fine-tunes Qwen2-VL on Korean VQA data using LoRA for memory efficiency.
Supports:
- LoRA / QLoRA fine-tuning
- Multi-GPU training with DeepSpeed
- Gradient checkpointing
- Mixed precision training
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from PIL import Image
import random

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from torch.utils.data import Dataset


# ============================================================================
# Data Loading
# ============================================================================

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


def format_conversation(messages: List[Dict], image_root: str) -> Dict:
    """
    Format conversation for Qwen2-VL.

    Returns:
        Dict with 'messages' and 'images' (list of PIL images)
    """
    formatted_messages = []
    images = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            formatted_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            msg_content = []
            for c in content:
                if c["type"] == "text":
                    msg_content.append({"type": "text", "text": c["text"]})
                elif c["type"] == "image":
                    img_path = c.get("path", "")
                    full_path = os.path.join(image_root, img_path)
                    if os.path.exists(full_path):
                        try:
                            img = Image.open(full_path).convert("RGB")
                            # Resize large images to prevent OOM
                            max_size = 1024
                            if max(img.size) > max_size:
                                ratio = max_size / max(img.size)
                                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                                img = img.resize(new_size, Image.LANCZOS)
                            images.append(img)
                            msg_content.append({"type": "image", "image": img})
                        except Exception:
                            continue

            if msg_content:
                formatted_messages.append({"role": role, "content": msg_content})

    return {
        "messages": formatted_messages,
        "images": images
    }


class KoreanVQADataset(Dataset):
    """Dataset for Korean VQA fine-tuning with Qwen2-VL."""

    def __init__(
        self,
        data_path: str,
        processor,
        image_root: str,
        max_length: int = 1024,
        max_samples: int = None,
    ):
        self.raw_data = load_jsonl(data_path, max_samples)
        self.processor = processor
        self.image_root = image_root
        self.max_length = max_length

        print(f"Loaded {len(self.raw_data)} samples from {data_path}")

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        item = self.raw_data[idx]
        messages = item.get("messages", [])

        try:
            # Format conversation
            formatted = format_conversation(messages, self.image_root)

            if not formatted["messages"]:
                return None

            # Apply chat template
            text = self.processor.apply_chat_template(
                formatted["messages"],
                tokenize=False,
                add_generation_prompt=False
            )

            # Process with images
            if formatted["images"]:
                inputs = self.processor(
                    text=[text],
                    images=formatted["images"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
            else:
                inputs = self.processor(
                    text=[text],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )

            # Remove batch dimension
            inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            # Create labels (shift is handled internally)
            inputs["labels"] = inputs["input_ids"].clone()

            return inputs

        except Exception as e:
            return None


def custom_collate_fn(batch: List[Optional[Dict]], processor) -> Optional[Dict]:
    """
    Custom collate function for Qwen2-VL with batch_size=1.
    Since Qwen2-VL has complex dynamic resolution handling,
    we simply return the single sample with batch dimension added.
    """
    # Filter out None samples
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    # For batch_size=1, just add batch dimension to all tensors
    if len(batch) == 1:
        result = {}
        for key, value in batch[0].items():
            if isinstance(value, torch.Tensor):
                if value.dim() == 0:
                    result[key] = value.unsqueeze(0)
                else:
                    result[key] = value.unsqueeze(0)
            else:
                result[key] = value
        return result

    # For batch_size > 1, handle padding
    # Separate different types of data
    input_ids_list = [b["input_ids"] for b in batch]
    attention_mask_list = [b["attention_mask"] for b in batch]
    labels_list = [b["labels"] for b in batch]

    # Find max length for padding
    max_len = max(ids.shape[0] for ids in input_ids_list)

    # Pad input_ids, attention_mask, labels
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []

    pad_token_id = processor.tokenizer.pad_token_id or 0

    for input_ids, attn_mask, labels in zip(input_ids_list, attention_mask_list, labels_list):
        pad_len = max_len - input_ids.shape[0]
        if pad_len > 0:
            padded_input_ids.append(torch.cat([input_ids, torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype)]))
            padded_attention_mask.append(torch.cat([attn_mask, torch.zeros(pad_len, dtype=attn_mask.dtype)]))
            padded_labels.append(torch.cat([labels, torch.full((pad_len,), -100, dtype=labels.dtype)]))
        else:
            padded_input_ids.append(input_ids)
            padded_attention_mask.append(attn_mask)
            padded_labels.append(labels)

    result = {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_mask),
        "labels": torch.stack(padded_labels),
    }

    # Handle pixel_values - concatenate along batch dim
    if "pixel_values" in batch[0]:
        result["pixel_values"] = torch.cat([b["pixel_values"].unsqueeze(0) if b["pixel_values"].dim() == 3 else b["pixel_values"] for b in batch], dim=0)

    # Handle image_grid_thw - stack maintaining the shape
    if "image_grid_thw" in batch[0]:
        grid_thws = [b["image_grid_thw"] for b in batch]
        # Ensure each has proper shape (N, 3) where N is number of images
        result["image_grid_thw"] = torch.cat([g if g.dim() == 2 else g.unsqueeze(0) for g in grid_thws], dim=0)

    return result


# ============================================================================
# Model Setup
# ============================================================================

def setup_lora_model(
    model_path: str,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    target_modules: List[str] = None,
    use_4bit: bool = False,
    use_8bit: bool = False,
):
    """
    Setup Qwen2-VL with LoRA.
    """
    print(f"Loading model from {model_path}...")

    # Quantization config
    quantization_config = None
    if use_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif use_8bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # Prepare for k-bit training if quantized
    if use_4bit or use_8bit:
        model = prepare_model_for_kbit_training(model)

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Default target modules for Qwen2-VL
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    # LoRA config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model, processor


# ============================================================================
# Custom Trainer
# ============================================================================

def safe_collate_fn(batch: List[Optional[Dict]], processor) -> Dict:
    """
    Safe collate function that always returns a valid batch.
    Skips None samples and returns a dummy batch if all samples are None.
    """
    result = custom_collate_fn(batch, processor)

    # If result is None (all samples failed), create a dummy batch
    if result is None:
        # This shouldn't happen often, but just in case
        # Return a minimal valid batch that will be skipped
        dummy = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": torch.zeros(1, 10, dtype=torch.long),
            "labels": torch.full((1, 10), -100, dtype=torch.long),
        }
        return dummy

    return result


class Qwen2VLTrainer(Trainer):
    """Custom trainer for Qwen2-VL that handles variable-size batches."""

    def __init__(self, processor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor

    def get_train_dataloader(self):
        from torch.utils.data import DataLoader

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=lambda batch: safe_collate_fn(batch, self.processor),
            pin_memory=True,
            drop_last=True,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        from torch.utils.data import DataLoader

        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return DataLoader(
            dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=lambda batch: safe_collate_fn(batch, self.processor),
            pin_memory=True,
        )


# ============================================================================
# Training
# ============================================================================

class SaveCallback(TrainerCallback):
    """Callback to save model at checkpoints."""

    def on_save(self, args, state, control, **kwargs):
        print(f"\nSaving checkpoint at step {state.global_step}...")


def train(
    model_path: str,
    data_dir: str,
    output_dir: str,
    # LoRA parameters
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    use_4bit: bool = False,
    use_8bit: bool = False,
    # Training parameters
    num_epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    max_length: int = 1024,
    max_train_samples: int = None,
    max_eval_samples: int = None,
    # Other
    seed: int = 42,
    save_steps: int = 500,
    eval_steps: int = 500,
    logging_steps: int = 10,
    resume_from_checkpoint: str = None,
):
    """
    Main training function.
    """
    print("=" * 70)
    print("Qwen2-VL Korean Fine-tuning")
    print("=" * 70)

    # Setup model
    model, processor = setup_lora_model(
        model_path=model_path,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_4bit=use_4bit,
        use_8bit=use_8bit,
    )

    # Get image root
    image_root = data_dir

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = KoreanVQADataset(
        data_path=os.path.join(data_dir, "train.jsonl"),
        processor=processor,
        image_root=image_root,
        max_length=max_length,
        max_samples=max_train_samples,
    )

    eval_dataset = None
    val_path = os.path.join(data_dir, "val.jsonl")
    if os.path.exists(val_path):
        eval_dataset = KoreanVQADataset(
            data_path=val_path,
            processor=processor,
            image_root=image_root,
            max_length=max_length,
            max_samples=max_eval_samples,
        )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        seed=seed,
        report_to="none",
        load_best_model_at_end=True if eval_dataset else False,
        ddp_find_unused_parameters=False,
        dataloader_drop_last=True,
    )

    # Initialize trainer
    trainer = Qwen2VLTrainer(
        processor=processor,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[SaveCallback()],
    )

    # Train
    print("\nStarting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    print("\nSaving final model...")
    final_path = os.path.join(output_dir, "final")
    trainer.save_model(final_path)

    # Merge LoRA weights and save full model
    print("\nMerging LoRA weights...")
    merged_path = os.path.join(output_dir, "merged")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_path)
    processor.save_pretrained(merged_path)

    print(f"\n{'=' * 70}")
    print("Training complete!")
    print(f"LoRA adapter: {final_path}")
    print(f"Merged model: {merged_path}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2-VL on Korean VQA")

    # Model
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="/NetDisk/juyeon/AdaMMS/Qwen2-VL-7B-Instruct",
        help="Path to base Qwen2-VL model"
    )

    # Data
    parser.add_argument(
        "--data", "-d",
        type=str,
        default="/NetDisk/juyeon/AdaMMS/data/korean_vqa",
        help="Path to processed Korean VQA data directory"
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="/NetDisk/juyeon/AdaMMS/checkpoints/qwen2vl_korean",
        help="Output directory for fine-tuned model"
    )

    # LoRA parameters
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--use-4bit", action="store_true", help="Use 4-bit quantization (QLoRA)")
    parser.add_argument("--use-8bit", action="store_true", help="Use 8-bit quantization")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--max-length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Max training samples")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Max eval samples")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--logging-steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    args = parser.parse_args()

    train(
        model_path=args.model,
        data_dir=args.data,
        output_dir=args.output,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        seed=args.seed,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        resume_from_checkpoint=args.resume,
    )


if __name__ == "__main__":
    main()
