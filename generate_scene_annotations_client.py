import json
import logging
import base64
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple, Dict, Any

import imageio
from PIL import Image
from tqdm import tqdm
from openai import OpenAI

# --- Configuration ---
DATASET_DIR = Path("/mnt/nas/synnas/docker2/robocoin-pipeline/robocoin-datasets/RMC-AIDA-L_box_up_down/format_convert")
CAMERA_MATCH = "observation.images.cam_high_rgb"
OUTPUT_FILE = Path("scene_annotations.jsonl")
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
NUM_EXTRACTORS = 10
# Number of concurrent HTTP requests to the server
# vLLM handles batching internally, so we can fire many requests at once.
# 64 concurrent requests usually saturate the GPU well without overwhelming the queue.
NUM_REQUEST_WORKERS = 64
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}
API_BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_episode_idx(file_path: Path) -> int:
    """Extract episode index from filename."""
    try:
        stem = file_path.stem
        parts = stem.split('_')
        if parts:
            return int(parts[-1])
        return -1
    except ValueError:
        return -1


def find_videos(root_dir: Path, match_str: str) -> List[Path]:
    """Find all video files matching the criteria recursively."""
    videos = []
    for path in root_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            if match_str in str(path):
                videos.append(path)
    return sorted(videos)


def extract_frames_worker(video_paths: List[Path], output_dir: Path) -> List[Tuple[int, str]]:
    """Worker function to extract the first frame from a list of videos."""
    results = []
    for video_path in video_paths:
        episode_idx = get_episode_idx(video_path)
        if episode_idx == -1:
            continue

        try:
            with imageio.get_reader(video_path) as reader:
                first_frame = reader.get_data(0)
            
            save_path = output_dir / f"{episode_idx}.jpg"
            Image.fromarray(first_frame).save(save_path)
            results.append((episode_idx, str(save_path)))
            
        except Exception as e:
            pass
            
    return results


def encode_image_base64(image_path: str) -> str:
    """Read image file and encode it as base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def send_request_task(client: OpenAI, idx: int, image_path: str, prompt: str) -> Dict[str, Any]:
    """Send a single request to the vLLM server."""
    try:
        base64_image = encode_image_base64(image_path)
        
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=128,
            temperature=0.0, # Greedy decoding
        )
        
        generated_text = response.choices[0].message.content.strip()
        return {
            "episode_idx": idx,
            "scene_annotation": generated_text
        }
    except Exception as e:
        logger.error(f"Request failed for episode {idx}: {e}")
        return {
            "episode_idx": idx,
            "scene_annotation": "" # Or handle error appropriately
        }


def main():
    # 1. Discovery
    logger.info(f"Scanning for videos in {DATASET_DIR}...")
    all_videos = find_videos(DATASET_DIR, CAMERA_MATCH)
    logger.info(f"Found {len(all_videos)} videos.")
    
    if not all_videos:
        logger.warning("No videos found. Exiting.")
        return

    # Initialize OpenAI Client
    # Note: Client is thread-safe
    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
    )

    # 2. Parallel Extraction
    with tempfile.TemporaryDirectory(prefix="vllm_client_frames_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        logger.info(f"Extracting frames to temporary directory: {temp_dir}")

        buckets: List[List[Path]] = [[] for _ in range(NUM_EXTRACTORS)]
        for video in all_videos:
            idx = get_episode_idx(video)
            if idx != -1:
                buckets[idx % NUM_EXTRACTORS].append(video)

        valid_inputs: List[Tuple[int, str]] = []
        
        with ProcessPoolExecutor(max_workers=NUM_EXTRACTORS) as executor:
            futures = [
                executor.submit(extract_frames_worker, bucket, temp_dir)
                for bucket in buckets if bucket
            ]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting frames"):
                valid_inputs.extend(future.result())

        valid_inputs.sort(key=lambda x: x[0])
        logger.info(f"Successfully extracted {len(valid_inputs)} frames.")

        if not valid_inputs:
            logger.error("No frames were extracted.")
            return

        # 3. Parallel Inference Requests
        logger.info(f"Sending requests to vLLM server at {API_BASE_URL}...")
        logger.info(f"Concurrency: {NUM_REQUEST_WORKERS} workers")
        
        prompt_text = "Describe the scene in this image in a single sentence."
        results = []

        # Use ThreadPoolExecutor for I/O bound HTTP requests
        with ThreadPoolExecutor(max_workers=NUM_REQUEST_WORKERS) as executor:
            futures = [
                executor.submit(send_request_task, client, idx, path, prompt_text)
                for idx, path in valid_inputs
            ]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing requests"):
                res = future.result()
                if res["scene_annotation"]: # Filter out failures
                    results.append(res)

        # 4. Save Results
        logger.info(f"Saving results to {OUTPUT_FILE}...")
        results.sort(key=lambda x: x["episode_idx"])
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for record in results:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Processing complete.")


if __name__ == "__main__":
    main()
