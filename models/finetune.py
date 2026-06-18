# models/finetune.py
"""
WHAT THIS FILE DOES:
====================
Fine-tunes Salesforce/codegen-350M-mono on the Python dataset using LoRA.

WHY CODEGEN-350M-MONO?
   - "mono" = trained only on Python (vs "multi" which includes other languages)
   - 350M parameters = small enough to fine-tune on a free Colab T4 GPU (16GB VRAM)
   - Pre-trained on BigCode dataset (billions of Python tokens) — strong starting point

WHY LoRA (Low-Rank Adaptation)?
   Full fine-tuning 350M params would require ~4GB VRAM just for gradients.
   LoRA freezes all original weights and inserts small trainable matrices at
   each attention layer. Specifically, for a weight matrix W (shape d×d):
     Instead of updating W, we learn two small matrices A (d×r) and B (r×d)
     where rank r=8. The update is: W + A×B
   This reduces trainable params from ~350M to ~1.2M (~0.3%) — same quality,
   10x less memory and 5x faster training.

WHAT GETS SAVED:
   models/finetuned-adapter/
     adapter_config.json        — LoRA config (rank, alpha, target layers)
     adapter_model.safetensors  — the trained A and B matrices (~4MB total)
     tokenizer files            — needed to encode/decode text the same way

   At inference time, we load the base model + this adapter. They merge on the fly.

TRAINING SETUP:
   Dataset:    5000 Python prompt/completion pairs from CodeParrot
   Task:       Causal language modeling — predict the next token
   Epochs:     3
   Batch size: 4 (with gradient accumulation of 4 → effective batch = 16)
   LR:         2e-5 with cosine decay
   VRAM used:  ~8GB on T4 (fits comfortably)
   Time:       ~45-60 min on T4
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType


# ─── Configuration ────────────────────────────────────────────────────────────

BASE_MODEL    = "Salesforce/codegen-350M-mono"
DATA_PATH     = "data/python_dataset.json"
OUTPUT_DIR    = "models/finetuned-adapter"
MAX_LENGTH    = 512    # tokens per training example (longer = more VRAM)
EPOCHS        = 3
BATCH_SIZE    = 4
GRAD_ACCUM    = 4      # effective batch = BATCH_SIZE × GRAD_ACCUM = 16
LEARNING_RATE = 2e-5
LORA_RANK     = 8      # LoRA r — higher = more capacity, more VRAM
LORA_ALPHA    = 32     # LoRA scaling factor (convention: alpha = 4 × rank)
LORA_DROPOUT  = 0.05


# ─── Step 1: Load training data ───────────────────────────────────────────────

def load_training_data(path):
    """Load the JSON file created by prepare_data.py."""
    print(f"Loading training data from {path}...")
    with open(path) as f:
        raw = json.load(f)

    # Combine prompt + completion into a single string.
    # The model learns to continue from wherever the prompt ends.
    texts = [item["prompt"] + item["completion"] for item in raw]
    print(f"  Loaded {len(texts)} training examples\n")
    return Dataset.from_dict({"text": texts})


# ─── Step 2: Tokenize ─────────────────────────────────────────────────────────

def tokenize(dataset, tokenizer):
    """
    Convert text → token IDs the model can process.
    truncation=True  → cut anything longer than MAX_LENGTH
    padding=False    → DataCollator will handle padding per batch
    """
    def _tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

    tokenized = dataset.map(_tokenize, batched=True, remove_columns=["text"])
    print(f"Tokenized {len(tokenized)} examples\n")
    return tokenized


# ─── Step 3: Apply LoRA ───────────────────────────────────────────────────────

def apply_lora(model):
    """
    Attach LoRA adapters to the model's attention projection layers.
    target_modules: the weight matrices inside each transformer block
      "qkv_proj" = query/key/value projection (where attention is computed)
      "out_proj"  = output projection after attention
    Freezes all other weights — only these adapter matrices will train.
    """
    lora_config = LoraConfig(
        r             = LORA_RANK,
        lora_alpha    = LORA_ALPHA,
        target_modules= ["qkv_proj", "out_proj"],
        lora_dropout  = LORA_DROPOUT,
        bias          = "none",
        task_type     = TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # shows "trainable params: ~1.2M (0.35%)"
    return model


# ─── Step 4: Train ────────────────────────────────────────────────────────────

def finetune():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: Training on CPU will take many hours. Use a GPU.\n")

    # Load base model
    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # CodeGen has no pad token by default

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    # Apply LoRA
    model = apply_lora(model)

    # Load + tokenize data
    dataset   = load_training_data(DATA_PATH)
    tokenized = tokenize(dataset, tokenizer)

    # DataCollator handles batching: pads sequences to same length within each batch
    # mlm=False means we do causal (next-token prediction), not masked language modeling
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training config
    training_args = TrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate               = LEARNING_RATE,
        lr_scheduler_type           = "cosine",
        warmup_ratio                = 0.05,
        fp16                        = (device == "cuda"),   # half precision on GPU
        logging_steps               = 50,
        save_strategy               = "epoch",
        report_to                   = "none",               # disable wandb/tensorboard
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = tokenized,
        data_collator   = collator,
    )

    print("\nStarting fine-tuning...\n")
    trainer.train()

    # Save only the LoRA adapter (not the full 700MB base model weights)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nAdapter saved to {OUTPUT_DIR}/")
    print("Next step: python evaluation/evaluator.py")


if __name__ == "__main__":
    finetune()
