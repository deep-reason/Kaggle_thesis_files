# ==============================
# Kaggle-optimized prepare_dataset.py (Option 2: Batched, No Shard Size)
# ==============================
import os
import gc
import torch
from datasets import load_dataset, Audio, Dataset, DatasetDict
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

# ----------------------------
# Config
# ----------------------------
DATASET_NAME = "AhunInteligence/w2v-bert-2.0-finetuning-amharic"
DATASET_CONFIG = "default"
TOKENIZER_REPO = "AhunInteligence/w2v-bert-2.0-amharic-finetunining-tokenizer"
OUTPUT_DATASET_REPO = "AhunInteligence/w2v2-amharic-prepared"

BATCH_SIZE = 1024  # Adjust depending on available memory
torch.backends.cudnn.benchmark = True

# ----------------------------
# Load processor
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
# Batch preparation function
# ----------------------------
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
# Process a single split
# ----------------------------
def process_and_push_split(split_name, streaming_split):
    print(f"\nProcessing {split_name} split...")

    # Cast audio column lazily
    streaming_split = streaming_split.cast_column("audio", Audio(sampling_rate=16_000))

    examples = []
    batch_buffer = {"audio": [], "transcription": []}
    total_counter = 0

    for example in streaming_split:
        batch_buffer["audio"].append(example["audio"])
        batch_buffer["transcription"].append(example["transcription"])
        total_counter += 1

        # Process batch when buffer is full
        if len(batch_buffer["audio"]) >= BATCH_SIZE:
            processed = prepare_batch(batch_buffer)
            for i in range(len(processed["input_values"])):
                examples.append({
                    "input_values": processed["input_values"][i],
                    "attention_mask": processed["attention_mask"][i],
                    "labels": processed["labels"][i],
                })
            batch_buffer = {"audio": [], "transcription": []}  # reset batch

    # Process any remaining examples
    if len(batch_buffer["audio"]) > 0:
        processed = prepare_batch(batch_buffer)
        for i in range(len(processed["input_values"])):
            examples.append({
                "input_values": processed["input_values"][i],
                "attention_mask": processed["attention_mask"][i],
                "labels": processed["labels"][i],
            })

    # Convert to Dataset and push to Hub
    ds = Dataset.from_list(examples)
    print(f"Pushing {split_name} split ({len(examples)} examples) to Hub...")
    ds.push_to_hub(f"{OUTPUT_DATASET_REPO}-{split_name}", private=True)

    # Cleanup
    del ds
    del examples
    gc.collect()

    print(f"✅ Finished processing {split_name}. Total examples processed: {total_counter}")

# ----------------------------
# Main
# ----------------------------
def main():
    print("Loading dataset in streaming mode...")
    streaming_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, streaming=True)

    for split_name in ["train", "valid", "test"]:
        process_and_push_split(split_name, streaming_dataset[split_name])

    print("\nAll splits processed and pushed to Hugging Face Hub.")

if __name__ == "__main__":
    main()
