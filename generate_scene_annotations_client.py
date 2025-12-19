import json
import logging
import base64
import shutil
import subprocess
import time
import socket
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

import imageio
from PIL import Image
from tqdm import tqdm
from openai import OpenAI
import httpx

# --- Configuration ---
@dataclass
class Config:
    DATASET_DIR: Path = Path("/mnt/nas/synnas/docker2/robocoin-datasets/AIRBOT_MMK2_bowl_storage_pepper")
    CAMERA_MATCH: str = "observation.images.cam_high_rgb"
    OUTPUT_FILE: Path = Path("scene_annotations_client.jsonl")
    MODEL_ID: str = "Qwen/Qwen2-VL-7B-Instruct"
    NUM_EXTRACTORS: int = 10
    NUM_REQUEST_WORKERS: int = 64
    VIDEO_EXTENSIONS: frozenset = frozenset({'.mp4', '.avi', '.mov', '.mkv'})
    
    # Image Server
    IMAGE_ROOT: Path = Path("/home/baai/SceneAnnotation/image")
    IMAGE_SERVER_PORT: int = 8081
    DATASET_NAME: str = DATASET_DIR.name

    # UDS Connection
    UDS_PATH: str = "/tmp/vllm-server.sock"
    API_BASE_URL: str = "http://localhost/v1"
    API_KEY: str = "EMPTY"
    
    # Logging
    SHOW_HTTP_LOGS: bool = False # Set to True to see detailed HTTP request logs
    
    # Retry Logic
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0 # Seconds

config = Config()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Suppress httpx logging if configured
if not config.SHOW_HTTP_LOGS:
    logging.getLogger("httpx").setLevel(logging.WARNING)


class LocalImageServer:
    """Context manager for a local HTTP image server."""
    def __init__(self, root_dir: Path, port: int):
        self.root_dir = root_dir
        self.port = port
        self.process = None

    def __enter__(self):
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if port is in use
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', self.port)) == 0:
                logger.warning(f"Port {self.port} is already in use. Assuming it's compatible.")
                return self

        logger.info(f"Starting local image server at http://localhost:{self.port} serving {self.root_dir}...")
        self.process = subprocess.Popen(
            ["python", "-m", "http.server", str(self.port), "--directory", str(self.root_dir)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(1) # Wait for startup
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
            logger.info("Local image server stopped.")


def get_episode_idx(file_path: Path) -> int:
    """Extract episode index from filename."""
    try:
        return int(file_path.stem.split('_')[-1])
    except (ValueError, IndexError):
        return -1


def find_videos(root_dir: Path, match_str: str) -> List[Path]:
    """Find all video files matching the criteria recursively."""
    return sorted([
        p for p in root_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in config.VIDEO_EXTENSIONS and match_str in str(p)
    ])


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
            
        except Exception:
            # logger is not process-safe without configuration, silent fail or print
            pass
            
    return results


def send_request_task(idx: int, image_path: str, prompt: str) -> Dict[str, Any]:
    """Send a single request to the vLLM server via UDS using Image URL.
    Retries on failures (including 400 errors). Deletes image only on success."""
    image_path_obj = Path(image_path)
    
    # Construct local URL
    try:
        rel_path = image_path_obj.relative_to(config.IMAGE_ROOT)
        image_url = f"http://localhost:{config.IMAGE_SERVER_PORT}/{rel_path}"
    except ValueError:
        print(f"Error: Image path {image_path} is not within IMAGE_ROOT")
        return {"episode_idx": idx, "scene_annotation": ""}

    transport = httpx.HTTPTransport(uds=config.UDS_PATH)
    
    # We create the client once per task (process-safe)
    client = OpenAI(
        api_key=config.API_KEY,
        base_url=config.API_BASE_URL,
        http_client=httpx.Client(transport=transport),
        max_retries=0, # We handle retries manually to control image deletion
    )

    last_error = None
    
    for attempt in range(config.MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=config.MODEL_ID,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                max_tokens=128,
                temperature=0.0,
            )
            
            generated_text = response.choices[0].message.content.strip()
            
            # Success! Delete image
            try:
                image_path_obj.unlink(missing_ok=True)
            except OSError:
                pass
                
            return {
                "episode_idx": idx,
                "scene_annotation": generated_text
            }
            
        except Exception as e:
            last_error = e
            # Log warning but continue to retry
            if attempt < config.MAX_RETRIES:
                # Calculate backoff: 1s, 2s, 4s...
                sleep_time = config.RETRY_DELAY * (2 ** attempt)
                # Using print for process-safety logging
                print(f"Warning: Request failed for episode {idx} (Attempt {attempt+1}/{config.MAX_RETRIES+1}). Retrying in {sleep_time}s... Error: {e}")
                time.sleep(sleep_time)
            else:
                # Final failure
                print(f"Error: Request failed for episode {idx} after {config.MAX_RETRIES+1} attempts. Keeping image for inspection. Error: {e}")
    
    return {
        "episode_idx": idx,
        "scene_annotation": "" 
    }


def main():
    # 1. Discovery
    logger.info(f"Scanning for videos in {config.DATASET_DIR}...")
    all_videos = find_videos(config.DATASET_DIR, config.CAMERA_MATCH)
    logger.info(f"Found {len(all_videos)} videos.")
    
    if not all_videos:
        logger.warning("No videos found. Exiting.")
        return

    # 2. Execution with Context Management
    with LocalImageServer(config.IMAGE_ROOT, config.IMAGE_SERVER_PORT):
        
        image_output_dir = config.IMAGE_ROOT / config.DATASET_NAME
        image_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Extracting frames to {image_output_dir}...")
            
            # Bucket videos for workers
            buckets = [[] for _ in range(config.NUM_EXTRACTORS)]
            for video in all_videos:
                idx = get_episode_idx(video)
                if idx != -1:
                    buckets[idx % config.NUM_EXTRACTORS].append(video)

            valid_inputs = []
            
            # Parallel Extraction
            with ProcessPoolExecutor(max_workers=config.NUM_EXTRACTORS) as executor:
                futures = [
                    executor.submit(extract_frames_worker, bucket, image_output_dir)
                    for bucket in buckets if bucket
                ]
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting frames"):
                    valid_inputs.extend(future.result())

            valid_inputs.sort(key=lambda x: x[0])
            logger.info(f"Successfully extracted {len(valid_inputs)} frames.")

            if not valid_inputs:
                logger.error("No frames were extracted.")
                return

            # Parallel Inference
            logger.info(f"Sending requests to vLLM server at {config.API_BASE_URL}...")
            #prompt_text = "List the objects in the image and their positions in a single line of plain text."
            prompt_text = "Describe the scene by listing each visible object in the format \"[object] is in the [direction].\" using directional terms such as front, back, left, right, front-left, front-right, back-left, or back-right. Output all descriptions in a single line of plain text with no markdown formatting."
            results = []

            with ProcessPoolExecutor(max_workers=config.NUM_REQUEST_WORKERS) as executor:
                futures = [
                    executor.submit(send_request_task, idx, path, prompt_text)
                    for idx, path in valid_inputs
                ]
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing requests"):
                    res = future.result()
                    if res["scene_annotation"]:
                        results.append(res)
                        
        finally:
            # Cleanup directory
            if image_output_dir.exists():
                try:
                    shutil.rmtree(image_output_dir)
                    logger.info(f"Cleaned up directory {image_output_dir}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup directory: {e}")

    # 3. Save Results
    logger.info(f"Saving results to {config.OUTPUT_FILE}...")
    results.sort(key=lambda x: x["episode_idx"])
        
    with open(config.OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Processing complete.")


if __name__ == "__main__":
    main()
