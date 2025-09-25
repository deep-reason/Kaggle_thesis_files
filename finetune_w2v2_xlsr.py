#!/usr/bin/env python3
# finetune_w2v2_streaming_kaggle.py

"""
Single-GPU fine-tuning of Wav2Vec2 on a large prepared HF dataset using streaming.
Designed for Kaggle (limited RAM/disk). Only a small validation subset is loaded into memory.
"""

import os
import sys
import torch
import logging
import numpy as np
from typing import Dict, Any, List

from datasets import load_dataset, disable_caching, Dataset
import evaluate

from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)

# ----------------------------
# CONFIG
# ----------------------------
PREPARED_DATASET_REPO = "AhunInteligence/w2v2-amharic-prepared"
PRETRAINED_MODEL = "facebook/wav2vec2-xls-r-300m"
TOKENIZER_REPO = "AhunInteligence/wav2vec2-amharic-finetuning-tokenizer"
OUTPUT_DIR = "wav2vec2-xls-r-300m-am-asr"
HUB_MODEL_REPO = f"AhunInteligence/{OUTPUT_DIR}"

BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-4
MAX_EPOCHS = 30
SAVE_STEPS = 400
LOGGING_STEPS = 50
NUM_VALIDATION_SAMPLES = 2000  # small subset

# ----------------------------
# HF TOKEN
# ----------------------------
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
if not HF_TOKEN:
    sys.exit("ERROR: HF_TOKEN not found in environment. Set HF_TOKEN in Kaggle 'Secrets'.")

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
disable_caching()

# ----------------------------
# Load Datasets (streaming)
# ----------------------------
logging.info("Loading datasets in streaming mode...")
train_ds = load_dataset(PREPARED_DATASET_REPO, split="train", streaming=True)

valid_stream = load_dataset(PREPARED_DATASET_REPO, split="validation", streaming=True)
valid_list = list(valid_stream.take(NUM_VALIDATION_SAMPLES))  # take small subset
valid_ds = Dataset.from_list(valid_list)

logging.info(f"Train dataset is streaming. Validation dataset: {len(valid_ds)} samples.")

# ----------------------------
# Load Processor and Model
# ----------------------------
logging.info("Loading tokenizer, feature extractor, processor, and model...")
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
model.freeze_feature_extractor()

# ----------------------------
# Data Collator
# ----------------------------
class DataCollatorCTCWithPadding:
    def __init__(self, processor: Wav2Vec2Processor, padding=True):
        self.processor = processor
        self.padding = padding

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(input_features, padding=True, return_tensors="pt")
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=True, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor)

# ----------------------------
# Metrics
# ----------------------------
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # replace -100 with pad_token_id for decoding
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# ----------------------------
# Training Arguments
# ----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    evaluation_strategy="steps",
    eval_steps=SAVE_STEPS,
    save_steps=SAVE_STEPS,
    logging_steps=LOGGING_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=MAX_EPOCHS,
    fp16=False,
    save_total_limit=2,
    push_to_hub=True,
    hub_model_id=HUB_MODEL_REPO,
    hub_token=HF_TOKEN,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    group_by_length=True,
    remove_unused_columns=False,
)

# ----------------------------
# Trainer
# ----------------------------
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=processor.feature_extractor,
)

# ----------------------------
# Train
# ----------------------------
if __name__ == "__main__":
    logging.info("Starting training...")
    trainer.train()
    logging.info("Training complete. Pushing model to Hugging Face Hub...")
    trainer.push_to_hub()
    logging.info("Done!")
