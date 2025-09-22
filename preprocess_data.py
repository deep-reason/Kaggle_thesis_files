# ==========================
# Kaggle-Optimized Dataset Preparation (Streaming + Safe Push)
# ==========================
import os
import gc
import torch
from datasets import load_dataset, Audio, IterableDataset
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

# ----------------------------
# Config
# ----------------------------
DATASET_NAME = "AhunInteligence/w2v-bert-2.0-finetuning-amharic"
DATASET_CONFIG = "default"

TOKENIZER_REPO = "AhunInteligence/w2v-bert-2.0-amharic-finetunining-tokenizer"
OUTPUT_DATASET_REPO = "AhunInteligence/w2v2-amharic-prepared"

BATCH_SIZE = 1024  # Adjust depending on available RAM

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

# ----------------------------
# Streaming dataset
# ----------------------------
print("Loading dataset in streaming mode...")
streaming_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, streaming=True)

# Cast audio columns lazily
for split in streaming_dataset:
    streaming_dataset[split] = streaming_dataset[split].cast_column("audio", Audio(sampling_rate=16_000))

# ----------------------------
# Batch processing function
# ----------------------------
def prepare_batch(batch):
    audio_arrays = [a["array"] for a in batch["audio"]]
    inputs = processor(audio_arrays, sampling_rate=16_000, return_tensors="np", padding=True)
    with processor.as_target_processor():
        labels = processor(batch["transcription"], padding=True, return_tensors="np").input_ids
    return {
        "input_values": inputs["input_values"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels,
    }

# ----------------------------
# Split-by-split processing & push
# ----------------------------
for split_name in ["train", "valid", "test"]:
    print(f"\nProcessing {split_name} split...")
    
    def generator():
        batch = []
        for example in streaming_dataset[split_name]:
            batch.append(example)
            if len(batch) >= BATCH_SIZE:
                processed = prepare_batch(batch)
                for i in range(len(processed["input_values"])):
                    yield {
                        "input_values": processed["input_values"][i],
                        "attention_mask": processed["attention_mask"][i],
                        "labels": processed["labels"][i],
                    }
                batch = []
                gc.collect()
        # Process remaining examples
        if batch:
            processed = prepare_batch(batch)
            for i in range(len(processed["input_values"])):
                yield {
                    "input_values": processed["input_values"][i],
                    "attention_mask": processed["attention_mask"][i],
                    "labels": processed["labels"][i],
                }
            batch = []
            gc.collect()

    iterable_ds = IterableDataset.from_generator(lambda: generator())
    print(f"Pushing {split_name} split to Hub...")
    iterable_ds.push_to_hub(f"{OUTPUT_DATASET_REPO}-{split_name}", private=True)
    print(f"✅ {split_name} split pushed successfully.")

    # Cleanup memory
    del iterable_ds
    gc.collect()

print("\n✅ All splits processed and pushed successfully.")
