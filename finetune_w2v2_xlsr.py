# single_gpu_optimized.py
import os
import sys
import torch
import logging
import numpy as np
from typing import Dict, Any, List
from datasets import load_dataset, disable_caching
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer,
)
import evaluate

# ----------------------------
# Config
# ----------------------------
PREPARED_DATASET_REPO = "AhunInteligence/w2v2-amharic-prepared"
PRETRAINED_MODEL = "facebook/wav2vec2-xls-r-300m"
TOKENIZER_REPO = "AhunInteligence/w2v-bert-2.0-amharic-finetunining-tokenizer"
OUTPUT_DIR = "wav2vec2-xls-r-300m-am-asr"
HUB_MODEL_REPO = f"AhunInteligence/{OUTPUT_DIR}"

BASE_BATCH_SIZE = 4          # Reduced for P100 memory
GRAD_ACCUM_STEPS = 4         # Accumulate to emulate larger batch
LR = 1e-4
EPOCHS = 30
SAVE_STEPS = 400
LOG_STEPS = 50

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
if not HF_TOKEN:
    sys.exit("ERROR: HF_TOKEN not found in environment.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
disable_caching()

# ----------------------------
# Load Dataset
# ----------------------------
logging.info("Loading dataset...")
train_ds = load_dataset(PREPARED_DATASET_REPO, split="train").shuffle(seed=42)
valid_ds = load_dataset(PREPARED_DATASET_REPO, split="validation").shuffle(seed=42).select(range(2000))

# ----------------------------
# Load Model and Processor
# ----------------------------
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
# Data Collator (Dynamic Padding)
# ----------------------------
class DynamicPaddingCollator:
    def __init__(self, processor: Wav2Vec2Processor):
        self.processor = processor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Separate input values and labels
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        # Dynamically pad input features (to the longest sequence in the batch)
        batch = self.processor.pad(input_features, padding=True, return_tensors="pt")
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=True, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DynamicPaddingCollator(processor)

# ----------------------------
# Metrics
# ----------------------------
wer_metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

# ----------------------------
# Training Arguments
# ----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    group_by_length=True,          # Speed + memory optimization
    per_device_train_batch_size=BASE_BATCH_SIZE,
    per_device_eval_batch_size=BASE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    evaluation_strategy="steps",
    num_train_epochs=EPOCHS,
    fp16=True,                     # Use mixed precision to reduce memory
    save_steps=SAVE_STEPS,
    eval_steps=SAVE_STEPS,
    logging_steps=LOG_STEPS,
    learning_rate=LR,
    weight_decay=0.005,
    warmup_steps=1000,
    save_total_limit=2,
    push_to_hub=True,
    hub_model_id=HUB_MODEL_REPO,
    hub_token=HF_TOKEN,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    remove_unused_columns=False,
)

# ----------------------------
# Trainer
# ----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ----------------------------
# Train
# ----------------------------
if __name__ == "__main__":
    logging.info("Starting single-GPU optimized training...")
    trainer.train()
    logging.info("Training complete. Pushing final model to HF Hub...")
    trainer.push_to_hub()
    logging.info("Done.")
