#!/usr/bin/env python3
"""
测试客户端 - 验证与服务端的基本通信
"""

import asyncio
import logging
import sys
from pathlib import Path

# 添加项目路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from simple_scene_client import SimpleSceneAnnotationClient

async def test_connection():
    """测试连接"""
    print("测试客户端连接...")
    
    client = SimpleSceneAnnotationClient("ws://localhost:8769")
    
    try:
        print("尝试连接服务器...")
        await client.run()
    except Exception as e:
        print(f"连接失败: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_connection())