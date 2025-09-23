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
    Trainer
)

# ----------------------------
# Config (edit these as needed)
# ----------------------------
# This is the repository you uploaded your prepared data to
PREPARED_DATASET_REPO = "AhunInteligence/w2v2-amharic-prepared"
PRETRAINED_MODEL = "facebook/wav2vec2-xls-r-300m" # Base model to fine-tune
TOKENIZER_REPO = "AhunInteligence/w2v-bert-2.0-amharic-finetunining-tokenizer"
OUTPUT_DIR = "wav2vec2-xls-r-300m-am-asr"
HUB_MODEL_REPO = f"AhunInteligence/{OUTPUT_DIR}" # Repository for the fine-tuned model
BATCH_SIZE = 8 # Tune this based on your GPU VRAM. Smaller is better for memory.
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-4
MAX_EPOCHS = 30
SAVE_STEPS = 400
LOGGING_STEPS = 50

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
    # Use load_dataset to stream the train split
    train_ds = load_dataset(PREPARED_DATASET_REPO, split="train", streaming=True)
    
    # Load a small, manageable portion of the validation data for evaluation
    # This avoids loading the entire set into memory/disk.
    # We load it first with streaming=True to avoid disk space issues, then take a sample.
    valid_ds_streaming = load_dataset(PREPARED_DATASET_REPO, split="validation", streaming=True)
    
    # The `take` method loads a fixed number of samples into a "map-style" dataset,
    # which is what the Trainer needs for evaluation.
    valid_ds_non_streaming = valid_ds_streaming.take(2000)
    
    # Ensure this is a Dataset and not an IterableDataset
    if isinstance(valid_ds_non_streaming, (IterableDataset, IterableDatasetDict)):
        raise TypeError("The validation dataset is still a streaming type.")

except Exception as e:
    logging.critical(f"Failed to load dataset: {e}. Please check the repo ID and file structure.")
    sys.exit(1)
logging.info("Dataset loading complete. Continuing to model setup...")
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
    PRETRAINED_MODEL,
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.0,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)
# Freeze feature extractor
model.freeze_feature_extractor()

# ----------------------------
# Data Collator
# ----------------------------
# A Data Collator is necessary to pad our batches to a uniform length
class DataCollatorCTCWithPadding:
    def __init__(self, processor: Wav2Vec2Processor, padding=True):
        self.processor = processor
        self.padding = padding

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
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

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

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
    fp16=False
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
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    ddp_find_unused_parameters=False,
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
