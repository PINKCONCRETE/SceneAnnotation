import os
import math
import imageio
import matplotlib.pyplot as plt
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from utils import (
    ensure_dir,
    get_filename_without_suffix,
)


@dataclass
class ExtractConfig:
    repo_dir: str
    camera: str = "observation.images.cam_high_rgb"
    save_dir: str = "first_frames"
    max_workers: int = 16


def find_all_videos(dir_path, camera):
    paths = []
    for root, dirnames, filenames in os.walk(dir_path):
        if camera in root:
            for filename in filenames:
                if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    paths.append(os.path.join(root, filename))
    paths.sort()
    return paths


def extract_first_frame(video_path, save_path):
    reader = imageio.get_reader(video_path)
    first_frame = reader.get_data(0)
    reader.close()
    imageio.imwrite(save_path, first_frame)
    return first_frame


def show_frames(frames):
    width, height = math.ceil(math.sqrt(len(frames))), math.ceil(math.sqrt(len(frames)))
    fig, axs = plt.subplots(height, width, figsize=(width, height))
    for i, frame in enumerate(frames):
        ax = axs[i // width, i % width]
        ax.imshow(frame)
        ax.axis('off')
    for i in range(len(frames), width * height):
        axs[i // width, i % width].axis('off')
    plt.tight_layout()
    plt.show()


def extract_first_frames_threaded(config: ExtractConfig):
    os.makedirs(config.save_dir, exist_ok=True)
    video_paths = find_all_videos(config.repo_dir, config.camera)
    save_paths = [
        os.path.join(
            config.save_dir,
            get_filename_without_suffix(config.repo_dir),
            get_filename_without_suffix(vp) + ".png",
        )
        for vp in video_paths
    ]
    for sp in save_paths:
        ensure_dir(sp)

    frames = [None] * len(video_paths)
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {
            executor.submit(extract_first_frame, vp, sp): idx
            for idx, (vp, sp) in enumerate(zip(video_paths, save_paths))
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting first frames (threads)"):
            idx = futures[future]
            frames[idx] = future.result()

    frames_compact = [f for f in frames if f is not None]
    show_frames(frames_compact)

