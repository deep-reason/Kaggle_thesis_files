# ==========================
# Script 1: Preprocess & Push to Hub (Batch-wise)
# ==================================
import os
import gc
#from huggingface_hub import create_repo
from datasets import load_dataset, Audio, DatasetDict, set_caching_enabled, Dataset
from transformers import Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor

# Disable global caching for datasets
set_caching_enabled(False)

# ----------------------------
# Hard-coded defaults
# ----------------------------
DATASET_NAME = "AhunInteligence/w2v-bert-2.0-finetuning-amharic"
DATASET_CONFIG = "default"
TRAIN_SPLIT = "train"
VALID_SPLIT = "valid"
TEST_SPLIT = "test"
TOKENIZER_REPO = "AhunInteligence/w2v-bert-2.0-amharic-finetunining-tokenizer"
PROCESSED_DATASET_REPO = "AhunInteligence/w2v-bert-2.0-amharic-preprocessed"
CHUNK_SIZE = 1024  # Process this many examples at a time

# ----------------------------
# Load tokenizer and processor once
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

# ----------------------------
# Preprocessing function
# ----------------------------
def prepare_dataset_batch(examples):
    audio_arrays = [audio["array"] for audio in examples["audio"]]
    
    # Initialize the batch dictionary
    batch = {} 
    
    # Pad all audio in the batch to the longest sample
    batch["input_values"] = processor(audio_arrays, sampling_rate=16000, padding=True).input_values
    batch["labels"] = processor.tokenizer(examples["transcription"]).input_ids
    return batch

# ----------------------------
# Main processing loop
# ----------------------------
splits_to_process = [TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT]

for split_name in splits_to_process:
    print(f"\nProcessing '{split_name}' split...")
    
    # Load the original dataset split in streaming mode for efficient iteration
    raw_dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=split_name,
        streaming=True,
    ).cast_column("audio", Audio(sampling_rate=16_000))

    chunked_list = []
    chunk_count = 0
    total_samples_processed = 0

    for example in raw_dataset:
        chunked_list.append(example)
        if len(chunked_list) == CHUNK_SIZE:
            chunk_count += 1
            print(f"  -> Processing chunk {chunk_count}...")
            
            # Convert list of examples to a Dataset object
            chunk_dataset = Dataset.from_list(chunked_list)
            
            # Apply preprocessing to the current chunk
            processed_chunk = chunk_dataset.map(
                prepare_dataset_batch,
                batched=True,
                batch_size=CHUNK_SIZE,
                remove_columns=["audio", "transcription"],
                num_proc=os.cpu_count(),
            )
            
            # Push to the new, unique repository
            processed_chunk.push_to_hub(
                repo_id=PROCESSED_DATASET_REPO,
                split=split_name,
                file_name=f"{split_name}_chunk_{chunk_count}.parquet", # Unique filename
                commit_message=f"Add processed {split_name} chunk {chunk_count}",
                private=False,
                # Setting append=True adds to the repo instead of overwriting.
                # It is the most critical change here.
                append=True
            )

            # Clean up memory and temporary files for the next chunk
            del chunked_list
            del chunk_dataset
            del processed_chunk
            gc.collect()
            
            chunked_list = []
            total_samples_processed += CHUNK_SIZE

    # Process any remaining examples in the last, incomplete batch
    if chunked_list:
        chunk_count += 1
        print(f"  -> Processing final chunk {chunk_count}...")
        chunk_dataset = Dataset.from_list(chunked_list)
        processed_chunk = chunk_dataset.map(
            prepare_dataset_batch,
            batched=True,
            batch_size=len(chunked_list),
            remove_columns=["audio", "transcription"],
            num_proc=os.cpu_count(),
        )
        
        # Push to the new, unique repository
        processed_chunk.push_to_hub(
            repo_id=PROCESSED_DATASET_REPO,
            split=split_name,
            file_name=f"{split_name}_final_chunk_{chunk_count}.parquet",
            commit_message=f"Add processed {split_name} final chunk {chunk_count}",
            private=False,
            append=True
        )
        del chunked_list
        del chunk_dataset
        del processed_chunk
        gc.collect()
        total_samples_processed += len(chunked_list)

print("\nAll data processing and uploading complete.")
