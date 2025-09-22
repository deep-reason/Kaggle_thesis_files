# ==========================
# Kaggle-optimized prepare_dataset.py (parquet-by-parquet)
# ==========================
import os
import shutil
import gc
from datasets import load_dataset, Dataset, DatasetDict, Audio
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
import numpy as np

# ----------------------------
# Config
# ----------------------------
DATASET_NAME = "AhunInteligence/w2v-bert-2.0-finetuning-amharic"
DATASET_CONFIG = "default"
TOKENIZER_REPO = "AhunInteligence/w2v-bert-2.0-amharic-finetunining-tokenizer"
OUTPUT_DATASET_REPO = "AhunInteligence/w2v2-amharic-prepared"
LOCAL_OUTPUT_DIR = "./prepared_dataset"

BATCH_SIZE = 1024  # Adjust to control RAM usage

os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

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
# Prepare a batch
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
# Process a split (streaming) and save parquet shards
# ----------------------------
def process_and_save_split(split_name, split_ds, batch_size=BATCH_SIZE):
    print(f"Processing split: {split_name} ...")
    split_dir = os.path.join(LOCAL_OUTPUT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)

    batch_buffer = {"audio": [], "transcription": []}
    shard_idx = 0

    for example in split_ds:
        batch_buffer["audio"].append(example["audio"])
        batch_buffer["transcription"].append(example["transcription"])

        if len(batch_buffer["audio"]) >= batch_size:
            # Process batch
            processed = prepare_batch(batch_buffer)
            ds = Dataset.from_dict(processed)

            # Save shard
            shard_file = os.path.join(split_dir, f"{split_name}-{shard_idx:05d}.parquet")
            ds.to_parquet(shard_file)
            print(f"Saved shard: {shard_file}")

            # Clear RAM
            del ds, processed, batch_buffer
            gc.collect()

            # Reset batch
            batch_buffer = {"audio": [], "transcription": []}
            shard_idx += 1

    # Save any remaining examples
    if len(batch_buffer["audio"]) > 0:
        processed = prepare_batch(batch_buffer)
        ds = Dataset.from_dict(processed)
        shard_file = os.path.join(split_dir, f"{split_name}-{shard_idx:05d}.parquet")
        ds.to_parquet(shard_file)
        print(f"Saved final shard: {shard_file}")
        del ds, processed, batch_buffer
        gc.collect()

# ----------------------------
# Main script
# ----------------------------
def main():
    print("Loading streaming dataset...")
    streaming_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, streaming=True)

    # Cast audio lazily
    for split in streaming_dataset:
        streaming_dataset[split] = streaming_dataset[split].cast_column("audio", Audio(sampling_rate=16000))

    # Process each split and save shards
    for split_name in ["train", "valid", "test"]:
        process_and_save_split(split_name, streaming_dataset[split_name])

    # ----------------------------
    # Load all shards into DatasetDict for final HF push
    # ----------------------------
    print("Loading all shards into DatasetDict for HF push...")
    dataset_dict = {}
    for split_name in ["train", "valid", "test"]:
        split_dir = os.path.join(LOCAL_OUTPUT_DIR, split_name)
        parquet_files = sorted([os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith(".parquet")])
        ds_list = [Dataset.from_parquet(f) for f in parquet_files]

        # Concatenate all shards efficiently
        concatenated = ds_list[0]
        for d in ds_list[1:]:
            concatenated = Dataset.from_dict({
                k: np.concatenate([concatenated[k], d[k]]) for k in concatenated.column_names
            })
            del d
            gc.collect()

        dataset_dict[split_name] = concatenated
        del ds_list, concatenated
        gc.collect()

    final_dataset = DatasetDict(dataset_dict)

    # Push to Hugging Face Hub (not private)
    print("Pushing dataset to HF Hub...")
    final_dataset.push_to_hub(OUTPUT_DATASET_REPO, private=False)
    print("✅ Dataset pushed successfully!")

    # ----------------------------
    # Cleanup local shards to save Kaggle disk
    # ----------------------------
    print("Cleaning up local shards...")
    shutil.rmtree(LOCAL_OUTPUT_DIR, ignore_errors=True)
    gc.collect()
    print("✅ Cleanup complete.")

if __name__ == "__main__":
    main()
