# ==========================
# Kaggle-friendly prepare_dataset.py (streaming + HF push)
# ==========================
import gc
from datasets import load_dataset, Dataset
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
import numpy as np

# ----------------------------
# Config
# ----------------------------
DATASET_NAME = "AhunInteligence/w2v-bert-2.0-finetuning-amharic"
DATASET_CONFIG = "default"
TOKENIZER_REPO = "AhunInteligence/w2v-bert-2.0-amharic-finetunining-tokenizer"
OUTPUT_DATASET_REPO = "AhunInteligence/w2v2-amharic-prepared"
BATCH_SIZE = 1024  # Adjust to fit Kaggle RAM

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
    labels = tokenizer(batch["transcription"], return_tensors="np").input_ids
    return {
        "input_values": inputs["input_values"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels,
    }

# ----------------------------
# Process split streaming & push
# ----------------------------
def process_and_push_split(split_name, split_ds, batch_size=BATCH_SIZE):
    print(f"Processing split: {split_name} ...")
    batch_buffer = {"audio": [], "transcription": []}
    shard_idx = 0

    for example in split_ds:
        batch_buffer["audio"].append(example["audio"])
        batch_buffer["transcription"].append(example["transcription"])

        if len(batch_buffer["audio"]) >= batch_size:
            processed = prepare_batch(batch_buffer)
            ds = Dataset.from_dict(processed)

            # Push to HF immediately
            ds.push_to_hub(
                repo_id=OUTPUT_DATASET_REPO,
                split=split_name,
                path_in_repo=f"{split_name}_{shard_idx:05d}.parquet",
                private=False,
                max_shard_size="2GB",  # optional
            )
            print(f"Pushed shard {shard_idx} of split {split_name} to HF")

            # Clear memory
            del ds, processed, batch_buffer
            gc.collect()
            batch_buffer = {"audio": [], "transcription": []}
            shard_idx += 1

    # Push remaining examples
    if len(batch_buffer["audio"]) > 0:
        processed = prepare_batch(batch_buffer)
        ds = Dataset.from_dict(processed)
        ds.push_to_hub(
            repo_id=OUTPUT_DATASET_REPO,
            split=split_name,
            path_in_repo=f"{split_name}_{shard_idx:05d}.parquet",
            private=False,
            max_shard_size="2GB",
        )
        print(f"Pushed final shard {shard_idx} of split {split_name} to HF")
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
        streaming_dataset[split] = streaming_dataset[split].cast_column("audio", "audio")

    # Process each split
    for split_name in ["train", "valid", "test"]:
        process_and_push_split(split_name, streaming_dataset[split_name])

    print("✅ All splits processed and pushed to HF Hub!")

if __name__ == "__main__":
    main()
