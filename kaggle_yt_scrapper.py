import os
import subprocess
import json
import logging
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import glob
import math
import gc

# External Libraries
from yt_dlp import YoutubeDL
import torch
import torchaudio
from datasets import Dataset, Features, Value, Audio
import pandas as pd
from tqdm.auto import tqdm
from huggingface_hub import HfApi, login, create_repo, hf_hub_download

# --- Configuration ---
CHANNELS = {
    "DW Amharic": "https://www.youtube.com/playlist?list=UU3-RNH75BEZslJLEoIAxa2A",
    "Fana BC": "https://www.youtube.com/playlist?list=UUZtXd8pSeqURf5MT2fqE51g",
    "Sheger FM": "https://www.youtube.com/playlist?list=UU9uvm3QhajePOVdct3Jv8Hg",
    "VOA Amharic": "https://www.youtube.com/playlist?list=UU5OePohJjdkd0Uwm3r1cThA",
    "ESAT": "https://www.youtube.com/playlist?list=UUSYM-vgRrMYsZbG-Z7Kz0Pw",
    "EBC ": "https://www.youtube.com/playlist?list=UUOhrz3uRCOHmK6ueUstw7_Q",
    "LTV": "https://www.youtube.com/playlist?list=UUKmDyLU_IH0wrtxHPK5xWlA",
    "Nahoo TV": "https://www.youtube.com/playlist?list=UUUhnQSskjPYay4TC_hq9uQQ",
    "EBS TV": "https://www.youtube.com/playlist?list=UUVcc_sbg3AcXLV9vVufJrGg",
    "Ahadu TV": "https://www.youtube.com/playlist?list=UU-x3kCrGEczM1Dr0UOKy-BQ",
    "Bisrat FM": "https://www.youtube.com/playlist?list=UUzyjEIJKYXExBGwP4fW_lNg",
    "Netsebraq TV": "https://www.youtube.com/playlist?list=UUsgwJqcvme0NsrVJl7bfsVg",
}

# --- Paths and HF Repository Configuration ---
KAGGLE_WORK_DIR = "/kaggle/working"
OUTPUT_DIR = os.path.join(KAGGLE_WORK_DIR, "temp_audio")
# !!! USER MUST UPDATE THIS TO THEIR OWN HF REPO ID !!!
HF_REPO_ID = "your-username/amharic-audio-chunks" 
# Log files local paths
URL_LOG_FILE = os.path.join(KAGGLE_WORK_DIR, "all_video_urls.jsonl")
BATCH_PROGRESS_LOG = os.path.join(KAGGLE_WORK_DIR, "batch_progress.log")
MANIFEST_FILE = os.path.join(KAGGLE_WORK_DIR, "manifest.tsv") 

# Processing Limits
NUM_VIDEOS_PER_CHANNEL = 5000
MIN_VIDEO_DURATION_SECONDS = 5
BATCH_SIZE = 250 
NUM_CPU_WORKERS = 4 
NUM_GPU_DEVICES = torch.cuda.device_count() 

# VAD Configuration
SAMPLE_RATE = 16000
VAD_MIN_CHUNK = 1 * SAMPLE_RATE 
VAD_MAX_CHUNK = 22 * SAMPLE_RATE 
VAD_THRESHOLD = 0.5 

# --- Logger Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("KaggleDataLoader")

# --- HF Authentication and API Setup ---
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    logger.error("HF_TOKEN environment variable not found. Please add your Hugging Face token to Kaggle Secrets.")
    sys.exit(1)

# Authenticate and initialize HfApi
login(token=HF_TOKEN, add_to_git_credential=True)
api = HfApi(token=HF_TOKEN)

try:
    api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)
    logger.info(f"Hugging Face repository {HF_REPO_ID} ensured.")
except Exception as e:
    logger.error(f"Failed to create/check HF repo {HF_REPO_ID}. Error: {e}")
    sys.exit(1)

# --- HF Log Management Helpers ---
def load_log_from_hf(repo_id, filename, local_path):
    """Downloads a log file from the HF repo if it exists."""
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=KAGGLE_WORK_DIR,
            local_dir_use_symlinks=False,
            repo_type="dataset",
            token=HF_TOKEN
        )
        logger.info(f"Successfully loaded {filename} from HF.")
        return True
    except Exception:
        logger.info(f"Could not find or download {filename} from HF. Starting fresh.")
        if os.path.exists(local_path):
            os.remove(local_path)
        return False

def save_log_to_hf(repo_id, local_path, filename, commit_message):
    """Uploads a local log file to the HF repo."""
    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message,
        )
        logger.info(f"Successfully uploaded/updated {filename} to HF.")
    except Exception as e:
        logger.error(f"Failed to upload {filename} to HF: {e}")

# --- Initial Setup & Log Loaders ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_video_urls(file_path):
    """Loads all video URLs and metadata."""
    load_log_from_hf(HF_REPO_ID, os.path.basename(file_path), file_path)
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def save_video_urls(file_path, all_entries):
    """Saves video URLs locally and pushes to HF."""
    with open(file_path, 'w') as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + "\n")
    save_log_to_hf(HF_REPO_ID, file_path, os.path.basename(file_path), "Initial commit of all video URLs and metadata (Step 0).")

def load_batch_progress(file_path):
    """Loads the last successfully completed batch index."""
    load_log_from_hf(HF_REPO_ID, os.path.basename(file_path), file_path)
    if not os.path.exists(file_path):
        return -1
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if lines:
                return int(lines[-1].strip())
            return -1
    except Exception:
        return -1

def save_batch_progress(file_path, batch_index):
    """Appends the completed batch index and pushes to HF."""
    try:
        with open(file_path, 'a') as f:
            f.write(f"{batch_index}\n")
    except Exception as e:
        logger.error(f"Error saving completed URL {batch_index} to {file_path}: {e}")

    save_log_to_hf(HF_REPO_ID, file_path, os.path.basename(file_path), f"Updated batch progress: completed batch {batch_index}.")

# --- Step 0: Initial URL Fetching ---
def fetch_video_urls(channel_name, channel_url, num_videos, min_duration_seconds):
    ydl_opts = {'quiet': True, 'extract_flat': True, 'dump_single_json': True, 'skip_download': True}
    video_entries = []
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(channel_url, download=False)
            entries = info.get('entries', [])
            for e in entries:
                # Use 'url' if available, otherwise construct from 'id'
                video_url = e.get('url') or (f"https://www.youtube.com/watch?v={e['id']}" if 'id' in e else None)
                if video_url and 'duration' in e and e['duration'] is not None and e['duration'] >= min_duration_seconds:
                    video_entries.append({
                        "url": video_url,
                        "title": e.get("title", "Unknown Title"),
                        "duration": e["duration"]
                    })
        return video_entries[:num_videos]
    except Exception as e:
        logger.error(f"Failed to fetch video info from {channel_name}: {e}")
        return []

def step_0_fetch_urls():
    all_video_entries = load_video_urls(URL_LOG_FILE)
    if all_video_entries:
        logger.info(f"Loaded {len(all_video_entries)} URLs from HF/local log. Skipping URL fetching.")
        return all_video_entries

    logger.info("--- Starting Step 0: Initial URL Fetching ---")
    all_video_entries = []
    with ThreadPoolExecutor(max_workers=len(CHANNELS)) as executor:
        future_to_channel = {
            executor.submit(fetch_video_urls, name, url, NUM_VIDEOS_PER_CHANNEL, MIN_VIDEO_DURATION_SECONDS): name
            for name, url in CHANNELS.items()
        }
        for future in as_completed(future_to_channel):
            channel_name = future_to_channel[future]
            try:
                video_entries = future.result()
                all_video_entries.extend(video_entries)
            except Exception as exc:
                logger.error(f"{channel_name} caused exception during fetch: {exc}")

    save_video_urls(URL_LOG_FILE, all_video_entries)
    logger.info(f"Total {len(all_video_entries)} video URLs collected and saved to HF.")
    return all_video_entries
# --------------------------------------------------------------------------

# --- Step 1: Multiprocessor Audio Downloading (UPDATED) ---

def download_single_url(url_entry, output_dir):
    """
    Downloads and converts a single URL to 16kHz mono WAV using yt-dlp API.
    Returns the path to the downloaded file or None on failure.
    """
    url = url_entry['url']
    video_id = url.split("v=")[-1].split('&')[0] # Get clean video ID
    
    # Filename template (yt-dlp automatically fills the ID/ext)
    # We use '%(id)s.%(ext)s' to ensure a clean, predictable name.
    # Note: The video ID will be part of the final filename.
    raw_audio_path_template = os.path.join(output_dir, "%(id)s.%(ext)s")
    final_audio_path = os.path.join(output_dir, f"{video_id}.wav")

    # If the final file already exists (from a previous run), skip download
    if os.path.exists(final_audio_path):
        return final_audio_path

    ydl_opts = {
        # General Options
        'format': 'bestaudio/best',             # Select best audio track
        'outtmpl': raw_audio_path_template,     # Use the path template
        'quiet': True,
        'no_warnings': True,
        'restrictfilenames': True,
        'continue': True,                       # Resume partially downloaded file
        'nocoverwrites': True,                  # Skip if file exists
        'ignoreerrors': True,                   # Crucial for multiprocessing stability

        # Post-Processing: Extract and Convert Audio
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',        # --extract-audio
            'preferredcodec': 'wav',            # --audio-format wav
            'preferredquality': '0',            # Best quality
        }],
        
        # Postprocessor Arguments: Apply specific FFmpeg filters
        'postprocessor_args': {
            'extract_audio': [
                '-ar', str(SAMPLE_RATE),         # Set sample rate to 16kHz
                '-ac', '1',                      # Convert to mono (1 channel)
                '-sample_fmt', 's16'             # Set sample format to 16-bit signed integer
            ]
        }
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
            # Check if the file was successfully created
            if os.path.exists(final_audio_path):
                return final_audio_path
            else:
                logger.warning(f"yt-dlp failed to produce the final WAV file for {video_id}.")
                return None
    except Exception as e:
        logger.error(f"[Download Error] {video_id}: {e}")
        return None

def step_1_download_batch(batch_urls):
    logger.info(f"--- Starting Step 1: Downloading {len(batch_urls)} Audios (Parallel) ---")
    downloaded_paths = []

    # Use ThreadPoolExecutor for I/O bound tasks like downloading
    with ThreadPoolExecutor(max_workers=NUM_CPU_WORKERS) as executor:
        # Submit tasks: (url_entry, output_dir)
        future_to_url = {
            executor.submit(download_single_url, entry, OUTPUT_DIR): entry 
            for entry in batch_urls
        }
        
        # Collect results
        for future in tqdm(as_completed(future_to_url), total=len(batch_urls), desc="Downloading"):
            result_path = future.result()
            if result_path:
                downloaded_paths.append(result_path)

    logger.info(f"Successfully downloaded {len(downloaded_paths)} audio files.")
    return downloaded_paths
# --------------------------------------------------------------------------

# --- Step 2: VAD Chunking (Remains Unchanged) ---
VAD_MODELS = []
def initialize_vad_models():
    global VAD_MODELS
    if VAD_MODELS: return
    logger.info(f"Initializing VAD models on {NUM_GPU_DEVICES} GPU(s)...")
    for i in range(NUM_GPU_DEVICES):
        device = f'cuda:{i}'
        # Using specific commit hash or tag is often safer in production
        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        model.to(device)
        model.eval()
        VAD_MODELS.append(model)
    logger.info("VAD models initialized.")

def get_vad_model(index):
    if not VAD_MODELS: raise RuntimeError("VAD models not initialized.")
    return VAD_MODELS[index % len(VAD_MODELS)]

def apply_vad_single(filepath, model_index):
    model = get_vad_model(model_index)
    device = next(model.parameters()).device
    video_id = os.path.splitext(os.path.basename(filepath))[0]
    
    try:
        waveform, sr = torchaudio.load(filepath)
        if sr != SAMPLE_RATE:
            # Although yt-dlp forces 16kHz, this is a safety net
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        audio_tensor = waveform.to(device)

        speech_timestamps = model.get_speech_timestamps(
            audio_tensor, 
            sampling_rate=SAMPLE_RATE, 
            threshold=VAD_THRESHOLD, 
            min_speech_samples=VAD_MIN_CHUNK, 
            max_speech_samples=VAD_MAX_CHUNK
        )
        timestamps = [(ts['start'], ts['end']) for ts in speech_timestamps]
        
        # Explicit RAM cleanup for the full audio waveform and tensor
        del audio_tensor
        del waveform
        torch.cuda.empty_cache()

        return video_id, timestamps

    except Exception as e:
        logger.error(f"VAD failed for {filepath}: {e}")
        return video_id, []

def step_2_vad_chunking(audio_paths):
    logger.info(f"--- Starting Step 2: VAD Chunking on {len(audio_paths)} Files (Multi-GPU) ---")
    initialize_vad_models()
    all_timestamps = {}
    
    # Use ThreadPoolExecutor for running VAD processing in parallel
    # The VAD processing itself uses the GPU(s) assigned to the model
    with ThreadPoolExecutor(max_workers=NUM_CPU_WORKERS) as executor:
        future_to_path = {
            executor.submit(apply_vad_single, path, i): path
            for i, path in enumerate(audio_paths)
        }
        
        for future in tqdm(as_completed(future_to_path), total=len(audio_paths), desc="VAD Processing"):
            video_id, timestamps = future.result()
            if timestamps:
                all_timestamps[video_id] = timestamps
            
    logger.info(f"Generated timestamps for {len(all_timestamps)} videos.")
    return all_timestamps
# --------------------------------------------------------------------------

# --- Step 3: Segmenting, Upload, and Cleanup (Remains Unchanged) ---

def segment_and_upload(video_id, audio_path, timestamps, new_batch_dataset, manifest_file):
    """
    Segments the audio, adds chunks to the NEW_BATCH_DATASET (in-memory), and
    CLEANS UP the full audio file and temporary chunks immediately.
    Returns True on success, False on failure.
    """
    temp_chunks_to_clean = []
    
    try:
        # Load the full audio file (small memory footprint)
        # Note: torchaudio.load is RAM-efficient for this temporary load
        waveform, sr = torchaudio.load(audio_path)
        
        # --- Segmentation Loop (Per-Chunk) ---
        for i, (start_sample, end_sample) in enumerate(timestamps):
            chunk = waveform[:, start_sample:end_sample]
            chunk_filename = f"{video_id}_{i:04d}.wav"
            temp_chunk_path = os.path.join(OUTPUT_DIR, chunk_filename)
            
            # 1. Save chunk to disk (REQUIRED for Audio feature)
            torchaudio.save(temp_chunk_path, chunk, sample_rate=SAMPLE_RATE, encoding="PCM_S", bits_per_sample=16)
            
            chunk_length_seconds = chunk.shape[1] / SAMPLE_RATE
            
            new_item = {
                "id": f"{video_id}_{i:04d}",
                "audio": temp_chunk_path, 
                "video_id": video_id,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "duration_s": chunk_length_seconds
            }
            
            # 2. Add item to the small, in-memory batch dataset
            new_batch_dataset.add_item(new_item)
            
            # 3. Update manifest
            manifest_file.write(f"{new_item['id']}\t{chunk_filename}\t{chunk_length_seconds}\n")
            
            temp_chunks_to_clean.append(temp_chunk_path)

            # 4. Cleanup chunk from disk IMMEDIATELY after it's added to the dataset object
            if os.path.exists(temp_chunk_path):
                os.remove(temp_chunk_path)
        
        # --- PER-VIDEO CLEANUP (IMMEDIATE) ---
        
        # 1. Clear full audio file from disk
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
        # 2. Explicit RAM cleanup
        del waveform
        gc.collect()
        
        return True

    except Exception as e:
        logger.error(f"Error segmenting/uploading chunks for {video_id}: {e}")
        
        # Ensure cleanup even on failure
        if os.path.exists(audio_path):
            os.remove(audio_path)
        for chunk_path in temp_chunks_to_clean:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
        
        return False


def step_3_segment_upload_cleanup(audio_paths, all_timestamps, batch_index):
    """
    Step 3: Segments audios, creates a small temporary batch dataset, pushes it in APPEND mode,
    and performs per-video disk/RAM cleanup.
    """
    logger.info(f"--- Starting Step 3: Segmenting, Streaming Upload, and Cleaning Batch {batch_index} ---")
    
    # 1. Initialize a small, temporary dataset for the current batch (RAM SAFE)
    features = Features({
        'id': Value('string'),
        'audio': Audio(sampling_rate=SAMPLE_RATE),
        'video_id': Value('string'),
        'start_sample': Value('int32'),
        'end_sample': Value('int32'),
        'duration_s': Value('float32')
    })
    new_batch_dataset = Dataset.from_dict({
        'id': [], 'audio': [], 'video_id': [], 
        'start_sample': [], 'end_sample': [], 'duration_s': []
    }, features=features)
        
    # 2. Segment and populate the temporary dataset and manifest file
    load_log_from_hf(HF_REPO_ID, os.path.basename(MANIFEST_FILE), MANIFEST_FILE)
    
    with open(MANIFEST_FILE, 'a') as manifest_file:
        is_empty = os.stat(MANIFEST_FILE).st_size == 0
        if is_empty: 
             manifest_file.write("chunk_id\tfilename\tduration_s\n")
             
        successful_videos = 0
        for audio_path in tqdm(audio_paths, desc="Segmenting/Cleanup"):
            video_id = os.path.splitext(os.path.basename(audio_path))[0]
            
            if video_id in all_timestamps:
                timestamps = all_timestamps[video_id]
                success = segment_and_upload(
                    video_id, audio_path, timestamps, new_batch_dataset, manifest_file
                )
                if success:
                    successful_videos += 1
            else:
                logger.warning(f"No valid speech chunks found for {video_id}. Clearing full audio file.")
                if os.path.exists(audio_path):
                    os.remove(audio_path)
    
    logger.info(f"Successfully processed {successful_videos} videos into the batch object.")

    # 3. Push dataset and manifest to Hugging Face (APPEND MODE - RAM SAFE)
    
    if len(new_batch_dataset) > 0:
        logger.info("Pushing latest audio batch to Hugging Face Hub in APPEND mode...")
        
        # *** CRITICAL: APPEND MODE TO PREVENT OOM ***
        new_batch_dataset.push_to_hub(
            HF_REPO_ID, 
            split='train', 
            commit_message=f"Batch {batch_index} audio chunks and metadata (APPEND).", 
            token=HF_TOKEN,
            append=True # <--- Tells HF to merge/append remotely
        )
        logger.info("Remote append complete.")
        
    # Push the manifest file explicitly using HfApi
    save_log_to_hf(
        HF_REPO_ID,
        MANIFEST_FILE,
        os.path.basename(MANIFEST_FILE),
        f"Updated manifest file after completing batch {batch_index}.",
    )
    
    # 4. Final Cleanup of the temporary directory (should be very small now)
    logger.info("Final cleanup of temporary working directory.")
    # Cleans up any non-.wav files (e.g., yt-dlp temp files)
    for temp_file in glob.glob(os.path.join(OUTPUT_DIR, "*")):
        if not temp_file.endswith(".wav"): # The full audio files are deleted in segment_and_upload
            os.remove(temp_file)
            
    logger.info(f"Cleanup complete for Batch {batch_index}.")

# --- Main Batch Loop Execution (Remains Unchanged) ---
def main():
    logger.info("--- Starting Data Collection Pipeline ---")
    
    all_video_entries = step_0_fetch_urls()
    if not all_video_entries:
        logger.error("No video URLs collected. Exiting.")
        return

    total_urls = len(all_video_entries)
    num_batches = math.ceil(total_urls / BATCH_SIZE)
    last_completed_batch = load_batch_progress(BATCH_PROGRESS_LOG)
    start_batch_index = last_completed_batch + 1

    if start_batch_index >= num_batches:
        logger.info(f"All {num_batches} batches are complete. Exiting.")
        return

    logger.info(f"Starting batch processing from Batch {start_batch_index} out of {num_batches} total batches.")

    for batch_index in range(start_batch_index, num_batches):
        logger.info(f"\n==================== Starting Batch {batch_index}/{num_batches-1} ====================")
        
        start_idx = batch_index * BATCH_SIZE
        end_idx = min((batch_index + 1) * BATCH_SIZE, total_urls)
        current_batch_urls = all_video_entries[start_idx:end_idx]
        
        # --- STEP 1: DOWNLOAD (UPDATED) ---
        downloaded_audio_paths = step_1_download_batch(current_batch_urls)
        if not downloaded_audio_paths:
            logger.warning(f"Batch {batch_index}: No audio files downloaded. Skipping.")
            save_batch_progress(BATCH_PROGRESS_LOG, batch_index)
            continue

        # --- STEP 2: VAD CHUNKING ---
        all_timestamps = step_2_vad_chunking(downloaded_audio_paths)
        
        # --- STEP 3: SEGMENT, UPLOAD, CLEANUP ---
        step_3_segment_upload_cleanup(downloaded_audio_paths, all_timestamps, batch_index)
        
        save_batch_progress(BATCH_PROGRESS_LOG, batch_index)

    logger.info("--- Pipeline finished successfully. ---")


if __name__ == "__main__":
    main()
  
