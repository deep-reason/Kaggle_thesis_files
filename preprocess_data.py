# ==========================
# Kaggle-optimized prepare_dataset.py
# ==========================
import os
import gc
from datasets import load_dataset, Audio, Dataset, DatasetDict
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

# ----------------------------
# Config
# ----------------------------
DATASET_NAME = "AhunInteligence/w2v-bert-2.0-finetuning-amharic"
DATASET_CONFIG = "default"
TOKENIZER_REPO = "AhunInteligence/w2v-bert-2.0-amharic-finetunining-tokenizer"
OUTPUT_DATASET_REPO = "AhunInteligence/w2v2-amharic-prepared"
BATCH_SIZE = 1024  # small enough to fit in RAM

# ----------------------------
# Load tokenizer & feature extractor
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
# Batch preprocessing
# ----------------------------
def prepare_batch(batch):
    audio_arrays = [a["array"] for a in batch["audio"]]
    inputs = processor(
        audio_arrays,
        sampling_rate=16000,
        return_tensors="np",
        padding=True
    )
    labels = [processor.tokenizer(t).input_ids for t in batch["transcription"]]
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

# Cast audio lazily
for split in streaming_dataset:
    streaming_dataset[split] = streaming_dataset[split].cast_column("audio", Audio(sampling_rate=16_000))

# ----------------------------
# Process each split in batches
# ----------------------------
def process_and_push_split(split_name, split_iterable):
    print(f"Processing split: {split_name} ...")
    batch_buffer = []
    batch_counter = 0

    for example in split_iterable:
        batch_buffer.append(example)
        if len(batch_buffer) >= BATCH_SIZE:
            # Process batch
            processed = prepare_batch(batch_buffer)
            # Convert to Dataset
            batch_ds = Dataset.from_dict(processed)
            # Push to HF
            shard_name = f"{OUTPUT_DATASET_REPO}-{split_name}-shard-{batch_counter}"
            batch_ds.push_to_hub(shard_name, private=True)
            print(f"Pushed {shard_name}")
            # Clear memory
            batch_buffer = []
            del batch_ds, processed
            gc.collect()
            batch_counter += 1

    # Push remaining examples in last partial batch
    if batch_buffer:
        processed = prepare_batch(batch_buffer)
        batch_ds = Dataset.from_dict(processed)
        shard_name = f"{OUTPUT_DATASET_REPO}-{split_name}-shard-{batch_counter}"
        batch_ds.push_to_hub(shard_name, private=True)
        print(f"Pushed {shard_name}")
        del batch_ds, processed, batch_buffer
        gc.collect()

# ----------------------------
# Main loop
# ----------------------------
def main():
    for split_name in ["train", "valid", "test"]:
        process_and_push_split(split_name, streaming_dataset[split_name])

if __name__ == "__main__":
    main()
    print("✅ Dataset preparation complete & all shards pushed to HF.")
