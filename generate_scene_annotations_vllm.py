import os
import json
import imageio
import shutil
import tempfile
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# Configuration
DATASET_DIR = "/mnt/nas/synnas/docker2/robocoin-pipeline/robocoin-datasets/RMC-AIDA-L_box_up_down/format_convert"
CAMERA = "observation.images.cam_high_rgb"
OUTPUT_FILE = "scene_annotations.jsonl"
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
NUM_EXTRACTORS = 10

def find_all_videos(dir_path, camera):
    """
    Recursively find all video files in the dataset directory that match the camera view.
    """
    paths = []
    for root, dirnames, filenames in os.walk(dir_path):
        if camera in root:
            for filename in filenames:
                if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    paths.append(os.path.join(root, filename))
    paths.sort()
    return paths

def get_episode_idx(video_path):
    """
    Extract episode index from filename (e.g., 'episode_000000.mp4' -> 0).
    """
    try:
        filename = os.path.basename(video_path)
        episode_str = filename.split('_')[-1].split('.')[0]
        return int(episode_str)
    except:
        return -1

def extract_frame_task(worker_id, video_paths, temp_dir):
    """
    Worker function to extract first frame from assigned videos.
    Assigns videos based on episode_idx % NUM_EXTRACTORS.
    Returns a list of (episode_idx, temp_image_path).
    """
    results = []
    # Assign tasks: only process videos where idx % NUM_EXTRACTORS == worker_id
    my_videos = [v for v in video_paths if get_episode_idx(v) % NUM_EXTRACTORS == worker_id]
    
    for video_path in my_videos:
        try:
            episode_idx = get_episode_idx(video_path)
            
            # Read first frame using imageio (ffmpeg backend)
            reader = imageio.get_reader(video_path)
            first_frame = reader.get_data(0)
            reader.close()
            
            # Save frame to temp directory
            save_path = os.path.join(temp_dir, f"{episode_idx}.jpg")
            Image.fromarray(first_frame).save(save_path)
            
            results.append((episode_idx, save_path))
        except Exception as e:
            # Silently ignore errors or log them if needed
            # print(f"Error extracting {video_path}: {e}")
            pass
            
    return results

def main():
    # 1. Find Videos
    print(f"Scanning for videos in {DATASET_DIR}...")
    video_paths = find_all_videos(DATASET_DIR, CAMERA)
    print(f"Found {len(video_paths)} videos.")
    
    if not video_paths:
        return

    # 2. Extract Frames (Parallel)
    # We create a temporary directory to store extracted frames
    temp_dir = tempfile.mkdtemp(prefix="vllm_frames_")
    print(f"Extracting frames to {temp_dir} with {NUM_EXTRACTORS} processes...")
    
    valid_inputs = [] # List of (episode_idx, image_path)
    
    # Use ProcessPoolExecutor to launch 10 processes
    with ProcessPoolExecutor(max_workers=NUM_EXTRACTORS) as executor:
        # Submit tasks: each worker gets the FULL list of videos but only processes its share
        futures = [executor.submit(extract_frame_task, i, video_paths, temp_dir) for i in range(NUM_EXTRACTORS)]
        
        # Wait for all to complete
        for future in tqdm(as_completed(futures), total=NUM_EXTRACTORS, desc="Extracting"):
            results = future.result()
            valid_inputs.extend(results)
    
    # Sort by episode_idx for cleaner output
    valid_inputs.sort(key=lambda x: x[0])
    print(f"Successfully extracted {len(valid_inputs)} frames.")

    if not valid_inputs:
        print("No frames extracted. Exiting.")
        shutil.rmtree(temp_dir)
        return

    # 3. vLLM Inference
    print("Initializing vLLM...")
    
    # Initialize Processor for prompt formatting
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Construct prompts for vLLM
    prompt_text = "Describe the scene in this image in a single sentence."
    
    vllm_inputs = []
    
    print("Preparing inputs for vLLM...")
    for idx, image_path in valid_inputs:
        image = Image.open(image_path).convert("RGB")
        
        # Use processor to format the prompt correctly with special tokens
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path}, # Valid path or placeholder
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        vllm_inputs.append({
            "prompt": prompt,
            "multi_modal_data": {
                "image": image
            },
        })

    # Initialize Engine
    # gpu_memory_utilization=0.9 ensures we use most VRAM but leave some for activation overhead
    # max_model_len limit helps to fit in memory
    # max_num_seqs limited to reduce sampler memory usage
    llm = LLM(
        model=MODEL_ID,
        max_model_len=4096, 
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_num_seqs=64,
    )

    sampling_params = SamplingParams(
        temperature=0.0, # Greedy decoding for deterministic results
        max_tokens=128,
        stop_token_ids=None
    )

    print("Generating responses...")
    # vLLM handles batching internally. It's much faster than manual loop.
    outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)

    # 4. Save Results
    print(f"Writing results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for i, output in enumerate(outputs):
            # vLLM outputs maintain the same order as inputs
            idx = valid_inputs[i][0]
            generated_text = output.outputs[0].text.strip()
            
            record = {
                "episode_idx": idx,
                "scene_annotation": generated_text
            }
            f.write(json.dumps(record) + "\n")

    # Cleanup
    print("Cleaning up temp files...")
    shutil.rmtree(temp_dir)
    print("Done!")

if __name__ == "__main__":
    # Ensure start method is spawn for compatibility if we used mp directly, 
    # but here we use ProcessPoolExecutor which defaults to fork on Linux.
    # Since we do extraction BEFORE loading vLLM (CUDA), fork is safe and faster.
    # vLLM is loaded AFTER extraction is done.
    main()
