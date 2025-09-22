# ==========================
# Kaggle-friendly prepare_dataset_streaming.py
# ==========================
import os
import shutil
import gc
from datasets import Dataset, Audio, load_dataset
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
import numpy as np

# ----------------------------
# Config
# ----------------------------
DATASET_NAME = "AhunInteligence/w2v-bert-2.0-finetuning-amharic"
DATASET_CONFIG = "default"
TOKENIZER_REPO = "AhunInteligence/w2v-bert-2.0-amharic-finetunining-tokenizer"
OUTPUT_DATASET_REPO = "AhunInteligence/w2v2-amharic-prepared"
LOCAL_SHARD_DIR = "./prepared_shards"
BATCH_SIZE = 2048  # adjust as needed

os.makedirs(LOCAL_SHARD_DIR, exist_ok=True)

# ----------------------------
# Load tokenizer/processor
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

# ----------------------------
# Prepare batch
# ----------------------------
def prepare_batch(batch):
    audio_arrays = [a["array"] for a in batch["audio"]]
    inputs = processor(audio_arrays, sampling_rate=16000, return_tensors="np", padding=True)
    labels = tokenizer(batch["transcription"], padding=True, return_tensors="np").input_ids
    return {
        "input_values": inputs["input_values"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels,
    }

# ----------------------------
# Process a split using streaming
# ----------------------------
def process_split(split_name):
    print(f"Streaming and processing split: {split_name}")
    split_dir = os.path.join(LOCAL_SHARD_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)

    ds_stream = load_dataset(DATASET_NAME, split=split_name, streaming=True)
    ds_stream = ds_stream.cast_column("audio", Audio(sampling_rate=16000))

    batch_buffer = {"audio": [], "transcription": []}
    shard_idx = 0

    for example in ds_stream:
        batch_buffer["audio"].append(example["audio"])
        batch_buffer["transcription"].append(example["transcription"])

        if len(batch_buffer["audio"]) >= BATCH_SIZE:
            processed = prepare_batch(batch_buffer)
            shard_ds = Dataset.from_dict(processed)

            shard_file = os.path.join(split_dir, f"{split_name}-{shard_idx:05d}.parquet")
            shard_ds.to_parquet(shard_file)
            print(f"Saved shard: {shard_file}")

            # Push shard immediately to HF
            shard_ds.push_to_hub(OUTPUT_DATASET_REPO, private=False)

            # Cleanup
            del shard_ds, processed
            batch_buffer = {"audio": [], "transcription": []}
            gc.collect()
            shard_idx += 1

    # Remaining examples
    if len(batch_buffer["audio"]) > 0:
        processed = prepare_batch(batch_buffer)
        shard_ds = Dataset.from_dict(processed)
        shard_file = os.path.join(split_dir, f"{split_name}-{shard_idx:05d}.parquet")
        shard_ds.to_parquet(shard_file)
        print(f"Saved final shard: {shard_file}")

        shard_ds.push_to_hub(OUTPUT_DATASET_REPO, private=False)

        del shard_ds, processed
        gc.collect()
        shard_idx += 1

    # Cleanup
    gc.collect()
    print(f"✅ Finished split: {split_name}")

# ----------------------------
# Main
# ----------------------------
def main():
    split_names = ["train", "valid", "test"]
    for split_name in split_names:
        process_split(split_name)

    # Final cleanup
    print("Cleaning local shards...")
    shutil.rmtree(LOCAL_SHARD_DIR, ignore_errors=True)
    print("✅ All done.")

if __name__ == "__main__":
    main()
