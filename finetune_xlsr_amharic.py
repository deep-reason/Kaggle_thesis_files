#!/usr/bin/env python3
# training_script.py
"""
Loads a large, prepared HF dataset in streaming mode and fine-tunes a
Wav2Vec2 model. This script is designed for environments with limited memory
like Kaggle, using streaming to avoid loading the entire dataset into RAM.

Requirements:
  - transformers
  - datasets
  - accelerate
  - huggingface_hub
  - torch
  - evaluate
  - jiwer
"""

import os
import sys
import torch
import evaluate
import numpy as np
import logging
from typing import Dict, Any, List
from datasets import load_dataset, Audio, Dataset, disable_caching
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
    is_torch_available,
    is_torch_cuda_available,
)

# ----------------------------
# Config (edit these as needed)
# ----------------------------
# This is the repository you uploaded your prepared data to
PREPARED_DATASET_REPO = "AhunInteligence/w2v2-amharic-prepared"
MODEL_CHECKPOINT = "facebook/wav2vec2-base" # Base model to fine-tune
OUTPUT_DIR = "w2v2-amharic-finetuned"
HUB_MODEL_REPO = f"AhunInteligence/{OUTPUT_DIR}" # Repository for the fine-tuned model
BATCH_SIZE = 8 # Tune this based on your GPU VRAM. Smaller is better for memory.
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-4
MAX_EPOCHS = 5
SAVE_STEPS = 500
LOGGING_STEPS = 100

# ----------------------------
# Environment / Token Check
# ----------------------------
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
if not HF_TOKEN:
    sys.exit(
        "ERROR: HF_TOKEN not found in environment. "
        "Set HF_TOKEN (Kaggle 'Secrets' or export HF_TOKEN=...) and re-run."
    )

# ----------------------------
# Logging Configuration
# ----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
disable_caching()

# ----------------------------
# Load Prepared Dataset
# ----------------------------
logging.info("Loading prepared dataset from Hugging Face Hub in streaming mode...")

try:
    # Use load_dataset to directly stream the parquet files from the repo.
    # The library will automatically infer the splits from the folder structure.
    dataset_dict = load_dataset(PREPARED_DATASET_REPO, streaming=True)
    
    # Check if the splits exist
    if "train" not in dataset_dict or "validation" not in dataset_dict:
        raise ValueError(
            f"The dataset at '{PREPARED_DATASET_REPO}' does not contain 'train' and 'validation' splits."
            "Make sure your data preparation script pushed files into 'train/' and 'validation/' folders."
        )

    train_ds = dataset_dict["train"]
    valid_ds = dataset_dict["validation"]

    # We need a non-streaming version for the Trainer to work with evaluation metrics
    # It's okay because the validation set is much smaller
    valid_ds_non_streaming = load_dataset(PREPARED_DATASET_REPO, split="validation")
    
except Exception as e:
    logging.critical(f"Failed to load dataset: {e}. Please check the repo ID and file structure.")
    sys.exit(1)

# ----------------------------
# Load Model and Processor
# ----------------------------
logging.info("Loading pre-trained Wav2Vec2 model and processor...")
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(TOKENIZER_REPO)
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True,
)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
model = Wav2Vec2ForCTC.from_pretrained(
    MODEL_CHECKPOINT,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)

# Freeze feature extractor
model.freeze_feature_extractor()

# ----------------------------
# Data Collator
# ----------------------------
# A Data Collator is necessary to pad our batches to a uniform length
class DataCollatorCTCWithPadding:
    def __init__(self, processor: Wav2Vec2Processor):
        self.processor = processor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have different tokenizers
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=True,
                return_tensors="pt",
            )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor)

# ----------------------------
# Metrics
# ----------------------------
logging.info("Loading evaluation metrics...")
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    
    pred_str = processor.batch_decode(pred_ids)
    # we do not want to decode padded tokens, we can filter them out
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

# ----------------------------
# Training Arguments
# ----------------------------
logging.info("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    group_by_length=True,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    evaluation_strategy="steps",
    num_train_epochs=MAX_EPOCHS,
    fp16=is_torch_cuda_available(), # Use FP16 for speed on NVIDIA GPUs
    save_steps=SAVE_STEPS,
    eval_steps=SAVE_STEPS,
    logging_steps=LOGGING_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.005,
    warmup_steps=1000,
    save_total_limit=2,
    push_to_hub=True,
    hub_model_id=HUB_MODEL_REPO,
    hub_token=HF_TOKEN,
    # This is critical for streaming datasets to work correctly with Trainer
    remove_unused_columns=False,
)

# ----------------------------
# Trainer
# ----------------------------
logging.info("Initializing Trainer...")
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_ds,
    eval_dataset=valid_ds_non_streaming,
    tokenizer=processor.feature_extractor,
)

# ----------------------------
# Train
# ----------------------------
if __name__ == "__main__":
    logging.info("Starting training...")
    trainer.train()

    # ----------------------------
    # Final push
    # ----------------------------
    logging.info("Training complete. Pushing final model to Hugging Face Hub...")
    trainer.push_to_hub()
    logging.info("Final model pushed to the Hub. Done.")
