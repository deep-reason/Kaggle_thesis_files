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
    vocab_size=len(processor_
