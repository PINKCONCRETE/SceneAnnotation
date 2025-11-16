"""
python scripts/extract_first_frame.py \
    --repo_dir="/home/koorye/.cache/huggingface/lerobot/unitree_g1_food_storage" \
    --camera="observation.images.cam_high_rgb" \
    --save_dir="results/frames"
"""

import draccus
import imageio
import math
import matplotlib.pyplot as plt
import multiprocessing
import os
from dataclasses import dataclass
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


@draccus.wrap()
def main(config: ExtractConfig):
    os.makedirs(config.save_dir, exist_ok=True)
    frames = []
    video_paths = find_all_videos(config.repo_dir, config.camera)
    for video_path in tqdm(video_paths, desc="Extracting first frames"):
        save_path = os.path.join(config.save_dir, get_filename_without_suffix(config.repo_dir), get_filename_without_suffix(video_path) + ".png")
        ensure_dir(save_path)
        frames.append(extract_first_frame(video_path, save_path))
    show_frames(frames)


def extract_first_frame_wrapper(video_paths, save_paths, frames):
    for video_path, save_path in tqdm(list(zip(video_paths, save_paths))):
        print(video_path)
        frame = extract_first_frame(video_path, save_path)
        frames.append(frame)


@draccus.wrap()
def main_multiprocess(config: ExtractConfig):
    os.makedirs(config.save_dir, exist_ok=True)
    video_paths = find_all_videos(config.repo_dir, config.camera)
    processes = []
    frames = multiprocessing.Manager().list()
    num_processes = 16
    chunk_size = (len(video_paths) + num_processes - 1) // num_processes
    for i in range(num_processes):
        chunk_video_paths = video_paths[i * chunk_size:(i + 1) * chunk_size]
        chunk_save_paths = [os.path.join(config.save_dir, get_filename_without_suffix(config.repo_dir), get_filename_without_suffix(vp) + ".png") for vp in chunk_video_paths]
        for save_path in chunk_save_paths:
            ensure_dir(save_path)
        p = multiprocessing.Process(target=extract_first_frame_wrapper, args=(chunk_video_paths, chunk_save_paths, frames))
        processes.append(p)
        print(f"Process {i}: {len(chunk_video_paths)} videos")
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    show_frames(list(frames))


if __name__ == "__main__":
    # main()
    main_multiprocess()