#!/usr/bin/env python3
"""
测试每个 episode 对应一个 parquet 文件的数据结构
"""

import json
import pandas as pd
from pathlib import Path
import re

def test_episode_parquet_structure():
    """测试每个 episode 对应一个 parquet 文件的结构"""
    
    # 模拟数据集名称
    dataset_name = "test_dataset_episodes"
    
    # 设置路径
    base_path = Path(f"/tmp/test_robocoin_datasets/{dataset_name}")
    annotations_path = base_path / "annotations"
    parquet_path = base_path / "scene_annotation_data"
    
    # 创建目录
    annotations_path.mkdir(parents=True, exist_ok=True)
    parquet_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 测试路径: {base_path}")
    print(f"📁 标注路径: {annotations_path}")
    print(f"📁 Parquet路径: {parquet_path}")
    
    # 创建测试数据
    test_episodes = [
        {"episode": 0, "description": "机器人抓取红色苹果"},
        {"episode": 1, "description": "机器人放置物品到篮子"},
        {"episode": 2, "description": "机器人清理桌面"}
    ]
    
    # 创建场景标注数据
    scene_annotations = []
    
    for idx, episode_data in enumerate(test_episodes):
        episode_num = episode_data["episode"]
        description = episode_data["description"]
        
        # 创建场景标注字典
        scene_annotation_dict = {
            "scene_annotation_index": idx,
            "scene_annotation": description
        }
        scene_annotations.append(scene_annotation_dict)
        
        # 为每个 episode 创建单独的 parquet 文件，只包含索引
        episode_df = pd.DataFrame({
            "scene_annotation": [idx]  # 只存储索引，不存储描述文本
        })
        
        # 保存为对应的 parquet 文件
        parquet_filename = f"episode_{episode_num}.parquet"
        episode_parquet_path = parquet_path / parquet_filename
        episode_df.to_parquet(episode_parquet_path, index=False)
        
        print(f"   ✅ 创建 {parquet_filename}")
    
    # 保存 scene_annotation.jsonl 文件
    jsonl_file = annotations_path / "scene_annotation.jsonl"
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for annotation in scene_annotations:
            f.write(json.dumps(annotation, ensure_ascii=False) + '\n')
    
    print(f"\n✅ 测试数据创建完成!")
    print(f"💾 JSONL文件: {jsonl_file}")
    print(f"💾 Parquet文件夹: {parquet_path}")
    print(f"📁 生成 {len(test_episodes)} 个 parquet 文件")
    
    # 验证数据
    print(f"\n🔍 验证数据...")
    
    # 读取 JSONL 文件
    print("📖 读取 JSONL 文件:")
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            data = json.loads(line.strip())
            print(f"   行 {line_num}: index={data['scene_annotation_index']}, content={data['scene_annotation']}")
    
    # 读取每个 parquet 文件
    print(f"\n📖 读取 Parquet 文件:")
    for episode_data in test_episodes:
        episode_num = episode_data["episode"]
        parquet_filename = f"episode_{episode_num}.parquet"
        parquet_file_path = parquet_path / parquet_filename
        if parquet_file_path.exists():
            df = pd.read_parquet(parquet_file_path)
            print(f"   📄 {parquet_filename}:")
            print(f"      索引值: {df.to_dict('records')[0]}")
    
    # 演示如何根据 episode 号查找对应的内容
    print(f"\n🔗 演示通过索引查找实际内容:")
    for episode_num in [0, 1, 2]:
        parquet_file_path = parquet_path / f"episode_{episode_num}.parquet"
        if parquet_file_path.exists():
            df = pd.read_parquet(parquet_file_path)
            scene_index = df['scene_annotation'].iloc[0]  # 获取索引
            actual_content = scene_annotations[scene_index]['scene_annotation']  # 通过索引查找实际内容
            print(f"   Episode {episode_num}: 索引={scene_index}, 内容='{actual_content}'")
    
    print(f"\n🎉 测试完成!")
    
    # 显示文件结构
    print(f"\n📂 生成的文件结构:")
    print(f"{base_path}/")
    print(f"├── annotations/")
    print(f"│   └── scene_annotation.jsonl")
    print(f"└── scene_annotation_data/")
    for episode_data in test_episodes:
        episode_num = episode_data["episode"]
        print(f"    ├── episode_{episode_num}.parquet")

if __name__ == "__main__":
    test_episode_parquet_structure()