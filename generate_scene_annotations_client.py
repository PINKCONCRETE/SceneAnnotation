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
from dataclasses import dataclass, field

import imageio
from PIL import Image
from tqdm import tqdm
from openai import OpenAI
import httpx

# --- Configuration --- 
@dataclass
class Config:
    # 数据集输入路径
    DATASET_DIR: Path = Path("/mnt/nas/synnas/成功区/Agilex_Cobot_Magic_fold_towel")
    
    # JSON输出根目录，可手动修改
    OUTPUT_ROOT: Path = Path(__file__).parent / "result test"
    
    CAMERA_MATCH: str = "observation.images.cam_high_rgb"
    OUTPUT_FILE: Path = field(init=False)
    
    MODEL_ID: str = "OpenGVLab/InternVL2-2B"
    NUM_EXTRACTORS: int = 10
    NUM_REQUEST_WORKERS: int = 64
    VIDEO_EXTENSIONS: frozenset = frozenset({'.mp4', '.avi', '.mov', '.mkv'})
    
    # Image Server
    IMAGE_ROOT: Path = Path("/home/key/Downloads/SceneAnnotation-feat-vlm/image_cache")
    IMAGE_SERVER_PORT: int = 8081
    DATASET_NAME: str = field(init=False)

    # UDS Connection
    UDS_PATH: str = "/tmp/vllm-server.sock"
    API_BASE_URL: str = "http://localhost/v1"
    API_KEY: str = "EMPTY"
    
    # Logging
    SHOW_HTTP_LOGS: bool = False
    
    # Retry Logic
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    
    def __post_init__(self):
        self.DATASET_NAME = self.DATASET_DIR.name
        self.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_FILE = self.OUTPUT_ROOT / f"{self.DATASET_NAME}_scene_annotations.jsonl"
@dataclass
class SpatialCoordinateSystem:
    # 空间区域阈值（可根据相机角度微调）
    FRONT_THRESHOLD: float = 0.35  # 前35%区域为front
    BACK_THRESHOLD: float = 0.65   # 后35%区域为back
    LEFT_THRESHOLD: float = 0.4    # 左40%区域为left
    RIGHT_THRESHOLD: float = 0.6   # 右40%区域为right
# 【核心】强化Prompt文本定义，紧邻Config类
# 用一句英文描述这个场景，格式是'sth. is in the back-left, back-right, front-left, front-right, front, back, left right or center.'(using ONE of these terms).
prompt_text = """
 # 用一句英文描述这个场景，格式是'sth. is in the back-left, back-right, front-left, front-right, front, back, left right or center.'(using ONE of these terms).
用英文严格按这写:The towel is in the center

注意:不要有关于robot和object的任何描述!!!请不要遗漏物体和重复描述"""

config = Config()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

if not config.SHOW_HTTP_LOGS:
    logging.getLogger("httpx").setLevel(logging.WARNING)


class LocalImageServer:
    # ... 保持不变的 LocalImageServer 类 ...
    """Context manager for a local HTTP image server."""
    def __init__(self, root_dir: Path, port: int):
        self.root_dir = root_dir
        self.port = port
        self.process = None

    def __enter__(self):
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', self.port)) == 0:
                logger.warning(f"Port {self.port} is already in use. Assuming it's compatible.")
                return self

        logger.info(f"Starting local image server at http://localhost:{self.port} serving {self.root_dir}...")
        self.process = subprocess.Popen(
            ["python3", "-m", "http.server", str(self.port), "--directory", str(self.root_dir)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(1)
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
    # ... 保持不变的函数 ...
    """Extract episode index from filename."""
    try:
        return int(file_path.stem.split('_')[-1])
    except (ValueError, IndexError):
        return -1


def find_videos(root_dir: Path, match_str: str) -> List[Path]:
    """只匹配RGB视频文件夹，排除depth文件夹"""
    return sorted([
        p for p in root_dir.rglob("*")
        if p.is_file() 
        and p.suffix.lower() in config.VIDEO_EXTENSIONS 
        and match_str in str(p)  # 路径包含目标字符串
        and "depth" not in p.parent.name  # ✅ 关键：父文件夹名不含"depth"
    ])


def extract_frames_worker(video_paths: List[Path], output_dir: Path) -> List[Tuple[int, str]]:
    # ... 保持不变的函数 ...
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
            pass
            
    return results


def send_request_task(idx: int, image_path: str, prompt: str) -> Dict[str, Any]:
    # ... 保持不变的函数 ...
    """Send a single request to the vLLM server via UDS using Image URL."""
    image_path_obj = Path(image_path)
    
    try:
        rel_path = image_path_obj.relative_to(config.IMAGE_ROOT)
        image_url = f"http://localhost:{config.IMAGE_SERVER_PORT}/{rel_path}"
    except ValueError:
        print(f"Error: Image path {image_path} is not within IMAGE_ROOT")
        return {"episode_idx": idx, "scene_annotation": ""}

    transport = httpx.HTTPTransport(uds=config.UDS_PATH)
    
    client = OpenAI(
        api_key=config.API_KEY,
        base_url=config.API_BASE_URL,
        http_client=httpx.Client(transport=transport),
        max_retries=0,
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
                # image_path_obj.unlink(missing_ok=True)
                pass
            except OSError:
                pass
                
            return {
                "episode_idx": idx,
                "scene_annotation": generated_text
            }
            
        except Exception as e:
            last_error = e
            if attempt < config.MAX_RETRIES:
                sleep_time = config.RETRY_DELAY * (2 ** attempt)
                print(f"Warning: Request failed for episode {idx} (Attempt {attempt+1}/{config.MAX_RETRIES+1}). Retrying in {sleep_time}s... Error: {e}")
                time.sleep(sleep_time)
            else:
                print(f"Error: Request failed for episode {idx} after {config.MAX_RETRIES+1} attempts. Keeping image for inspection. Error: {e}")
    
    return {
        "episode_idx": idx,
        "scene_annotation": "" 
    }


def clean_annotations(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # ... 保持不变的函数 ...
    """
    后处理：移除行首序号，保留物体数量，转换为单行格式
    - 将换行符转为逗号
    - 移除行首的序号标记（如 "1.", "2.", "3." 或 "1", "2", "3"）
    - 保留物体数量描述（如 "2 monitors"）
    - 去重后输出单行逗号分隔格式
    """
    cleaned = []
    for res in results:
        text = res.get("scene_annotation", "")
        
        # 将换行符替换为逗号，转为单行
        text = text.replace('\n', ', ')
        
        if not text or len(text) < 10:
            print(f"Skipping episode {res['episode_idx']}: empty or too short annotation")
            continue
            
        # 分割成各个物体描述
        items = [item.strip() for item in text.split(',') if item.strip()]
        
        # 清理每个项目：移除行首序号
        cleaned_items = []
        for item in items:
            # 移除行首的数字序号（如 "1.", "2.", "1", "2"）
            # 但不移除物体数量（如 "2 monitors" 中的 "2"）
            words = item.split()
            if len(words) >= 2 and words[0].rstrip('.').isdigit():
                # 检查是否为序号：数字后直接跟冠词（a, an, the）或物体名称
                second_word = words[1].lower().rstrip('.')
                # 如果第二个词是物体名称的开头（不是位置介词），则这是序号
                # 保留描述部分，移除序号
                cleaned_item = ' '.join(words[1:])
            else:
                cleaned_item = item
            
            # 移除末尾可能残留的点号
            cleaned_item = cleaned_item.rstrip('.')
            cleaned_items.append(cleaned_item)
        
        items = cleaned_items
        
        # 去重处理
        seen_objects = set()
        unique_items = []
        
        for item in items:
            if " is in the " in item:
                # 提取物体描述用于去重
                obj_info = item.split(" is in the ")
                if len(obj_info) >= 2:
                    obj_desc = obj_info[0].strip().lower()
                    # 移除冠词进行标准化
                    obj_key = obj_desc.replace('a ', '').replace('an ', '').replace('the ', '').strip()
                    
                    if obj_key and obj_key not in seen_objects:
                        seen_objects.add(obj_key)
                        unique_items.append(item)
                else:
                    unique_items.append(item)
            else:
                unique_items.append(item)
        
        if not unique_items:
            print(f"Skipping episode {res['episode_idx']}: all items were filtered out")
            continue
            
        # 使用逗号连接成单行输出
        res["scene_annotation"] = ", ".join(unique_items)
        cleaned.append(res)
    
    print(f"Cleaned {len(results)} raw annotations -> {len(cleaned)} valid entries")
    return cleaned


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
            logger.info(f"Successfully extracted {len(valid_inputs)} raw frames.")

            # ✅ 去重：确保每个episode只处理一次
            seen_idx = set()
            unique_inputs = []
            for idx, path in valid_inputs:
                if idx not in seen_idx:
                    seen_idx.add(idx)
                    unique_inputs.append((idx, path))

            valid_inputs = unique_inputs
            logger.info(f"After deduplication: {len(valid_inputs)} unique frames to process.")

            if not valid_inputs:
                logger.error("No frames were extracted.")
                return
            
            # 3. Parallel Inference
            logger.info(f"Sending requests to vLLM server at {config.API_BASE_URL}...")
            
            # 直接使用模块级别的 prompt_text
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
                    #shutil.rmtree(image_output_dir)
                    logger.info(f"Cleaned up directory {image_output_dir}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup directory: {e}")

    # 4. 后处理：移除行首序号，保留数量，转换为单行
    logger.info(f"Cleaning annotations (removing line numbers)...")
    results = clean_annotations(results)
    
    # 5. Save Results
    logger.info(f"Saving {len(results)} cleaned results to {config.OUTPUT_FILE}...")
    config.OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.sort(key=lambda x: x["episode_idx"])
        
    with open(config.OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Processing complete. Successfully annotated {len(results)} episodes.")


if __name__ == "__main__":
    main()