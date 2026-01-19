import json
import logging
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional

import imageio
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# --- Configuration ---
DATASET_DIR = Path("/mnt/nas/synnas/docker2/robocoin-pipeline/robocoin-datasets/RMC-AIDA-L_box_up_down/format_convert")
CAMERA_MATCH = "observation.images.cam_high_rgb"
OUTPUT_FILE = Path("scene_annotations.jsonl")
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
NUM_EXTRACTORS = 10
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_episode_idx(file_path: Path) -> int:
    """Extract episode index from filename (e.g., 'episode_000000.mp4' -> 0)."""
    try:
        # Assuming format like '..._123.mp4' or just numbers
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
    """
    Worker function to extract the first frame from a list of videos.
    Returns a list of (episode_idx, frame_path_str).
    """
    results = []
    for video_path in video_paths:
        episode_idx = get_episode_idx(video_path)
        if episode_idx == -1:
            continue

        try:
            # Using context manager for imageio reader
            with imageio.get_reader(video_path) as reader:
                first_frame = reader.get_data(0)
            
            save_path = output_dir / f"{episode_idx}.jpg"
            Image.fromarray(first_frame).save(save_path)
            results.append((episode_idx, str(save_path)))
            
        except Exception as e:
            # Log error but don't crash the worker
            # logger.warning(f"Failed to extract {video_path.name}: {e}")
            pass
            
    return results


def main():
    # 1. Discovery
    logger.info(f"Scanning for videos in {DATASET_DIR}...")
    all_videos = find_videos(DATASET_DIR, CAMERA_MATCH)
    logger.info(f"Found {len(all_videos)} videos.")
    
    if not all_videos:
        logger.warning("No videos found. Exiting.")
        return

    # 2. Parallel Extraction
    # Use a TemporaryDirectory context manager for automatic cleanup
    with tempfile.TemporaryDirectory(prefix="vllm_frames_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        logger.info(f"Extracting frames to temporary directory: {temp_dir}")

        # Distribute videos to workers based on idx % NUM_EXTRACTORS
        # This pre-sorting avoids having every worker scan the full list
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

        # 3. vLLM Inference
        logger.info("Initializing vLLM Processor...")
        try:
            processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load processor: {e}")
            return

        logger.info("Preparing vLLM inputs...")
        prompt_text = "Describe the scene in this image in a single sentence."
        vllm_inputs = []

        for idx, image_path_str in valid_inputs:
            # Create the prompt using the chat template
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path_str},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            # Generate the full prompt string (handling <|vision_start|>, etc.)
            text_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # vLLM expects raw images (PIL) for multi_modal_data
            image = Image.open(image_path_str).convert("RGB")
            
            vllm_inputs.append({
                "prompt": text_prompt,
                "multi_modal_data": {"image": image},
            })

        logger.info("Initializing vLLM Engine...")
        llm = LLM(
            model=MODEL_ID,
            max_model_len=4096,
            limit_mm_per_prompt={"image": 1},
            trust_remote_code=True,
            gpu_memory_utilization=0.9,  # Tuned for stability
            max_num_seqs=64,             # Tuned to prevent OOM
        )

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=128,
        )

        logger.info("Running inference...")
        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)

        # 4. Save Results
        logger.info(f"Saving results to {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for i, output in enumerate(outputs):
                idx = valid_inputs[i][0]
                generated_text = output.outputs[0].text.strip()
                
                record = {
                    "episode_idx": idx,
                    "scene_annotation": generated_text
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Processing complete.")


if __name__ == "__main__":
    main()
