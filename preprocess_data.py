# Kaggle-optimized prepare_dataset.py
import os
import torch
from datasets import load_dataset, Audio, DatasetDict
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

# ----------------------------
# Config
# ----------------------------
DATASET_NAME = "AhunInteligence/w2v-bert-2.0-finetuning-amharic"
DATASET_CONFIG = "default"

TOKENIZER_REPO = "AhunInteligence/w2v-bert-2.0-amharic-finetunining-tokenizer"
OUTPUT_DATASET_REPO = "AhunInteligence/w2v2-amharic-prepared"

BATCH_SIZE = 1024

torch.backends.cudnn.benchmark = True

# ----------------------------
# Processor
# ----------------------------
print("Loading tokenizer/feature extractor...")
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(TOKENIZER_REPO)
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True,
)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

def prepare_batch(batch):
    audio_arrays = [a["array"] for a in batch["audio"]]
    inputs = processor(audio_arrays, sampling_rate=16000, return_tensors="np", padding=True)

    with processor.as_target_processor():
        labels = processor(batch["transcription"], padding=True, return_tensors="np").input_ids

    return {
        "input_values": inputs["input_values"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels,
    }

# ----------------------------
# Streaming dataset
# ----------------------------
print("Loading dataset in streaming mode...")
streaming_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, streaming=True)

# Cast audio columns lazily
for split in streaming_dataset:
    streaming_dataset[split] = streaming_dataset[split].cast_column("audio", Audio(sampling_rate=16_000))

# ----------------------------
# Process each split lazily
# ----------------------------
processed_splits = {}
for split in ["train", "valid", "test"]:
    print(f"Processing {split} split...")
    processed_splits[split] = streaming_dataset[split].map(
        prepare_batch,
        batched=True,
        batch_size=BATCH_SIZE,
        remove_columns=["audio", "transcription"],
    )

# ----------------------------
# Push whole DatasetDict once
# ----------------------------
print("Pushing all splits to Hub (one shot)...")
processed_dataset = DatasetDict(processed_splits)
processed_dataset.push_to_hub(OUTPUT_DATASET_REPO, private=True, max_shard_size="500MB")

print("✅ Dataset preparation complete & pushed to:", OUTPUT_DATASET_REPO)
