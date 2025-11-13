#!/usr/bin/env python3
"""简单场景标注客户端 - 用于测试通信和执行场景标注pipeline"""
import sys
import asyncio
import logging
import sys
import subprocess
import os
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))
from task_client import TaskClient
from constant import(
    TASK_SUCCESS,
    TASK_FAILED,
)

class SimpleSceneAnnotationClient(TaskClient):
    def get_task_category(self) -> str:
        return "scene_annotation"

    def generate_task_request_desc(self) -> dict:
        return {"client_type": "simple_scene_annotation"}

    def _sync_process_task(self, task_content: dict) -> dict:
        """处理任务 - 调用外部pipeline脚本进行场景标注"""
        print(f"收到任务内容: {task_content}")
        
        dataset_uuid = task_content.get("dataset_uuid")
        leformat_path = task_content.get("leformat_path")
        
        print(f"收到任务: {dataset_uuid}, 路径: {leformat_path}")
        
        if not dataset_uuid:
            raise ValueError("❌ 错误: dataset_uuid 为空!")
        if not leformat_path:
            raise ValueError("❌ 错误: leformat_path 为空!")
        
        try:
            # 调用外部pipeline脚本进行场景标注
            result = self._run_scene_annotation_pipeline(dataset_uuid, leformat_path)

            if result["success"]:
                print("✅ 场景标注处理成功!")
                return {
                    "task_result_status": TASK_SUCCESS,
                    "processed": True,
                    "dataset_uuid": dataset_uuid,
                    "output_path": result.get("output_path"),
                    "annotations_count": result.get("annotations_count", 0),
                    "descriptions": result.get("descriptions", []),
                }
            else:
                raise ValueError(f"❌ 场景标注处理失败: {result.get('error')}")
                
        except Exception as e:
            print(f"❌ 处理任务时发生异常: {e}")
            import traceback
            traceback.print_exc()
            return {
                "task_result_status": TASK_FAILED,
                "error": str(e),
                "dataset_uuid": dataset_uuid
            }
    
    def _run_scene_annotation_pipeline(self, dataset_uuid: str, leformat_path: str) -> dict:
        """运行场景标注pipeline脚本"""
        
        print("\n" + "🚀 " + "="*58)
        print("🎯 开始执行场景标注 Pipeline")
        print("="*60)
        
        try:
            # 从leformat_path提取数据集信息
            dataset_path = leformat_path
            dataset_name = os.path.basename(dataset_path)
            repo_root = os.path.dirname(dataset_path) + "/"
            repo_id = dataset_name
            
            print(f"📦 数据集UUID: {dataset_uuid}")
            print(f"📁 数据集完整路径: {dataset_path}")
            print(f"📂 repo_root: {repo_root}")
            print(f"🏷️  repo_id: {repo_id}")
            
            # Pipeline配置
            pipeline_script = "./scripts/run_pipeline.py"
            save_root = "/tmp/scene_annotation_results"
            
            # 构建pipeline命令
            cmd = [
                "python", pipeline_script,
                f"--repo_id={repo_id}",
                f"--repo_root={repo_root}",
                f"--save_root={save_root}",
                "--camera=observation.images.cam_high",
                "--detector.type=grounding_dino",
                "--detector.device=cuda",
                "--detector.visualize_first=5",
                "--language_model.type=ollama",
                "--language_model.model=deepseek-r1:14b",
                "--language_model.think=False"
            ]
            
            print("⚙️  Pipeline 配置:")
            print(f"   📂 repo_root: {repo_root}")
            print(f"   🏷️  repo_id: {repo_id}")
            print(f"   💾 输出路径: {save_root}")
            print(f"   🤖 模型: deepseek-r1:14b")
            print(f"   🎯 检测器: grounding_dino (CUDA)")
            print(f"   📷 相机: observation.images.cam_high")
            
            print("\n🚀 正在执行 Pipeline... (这可能需要一些时间)")
            print(f"命令: {' '.join(cmd)}")
            
            # 执行pipeline命令
            result = subprocess.run(
                cmd,
                cwd=".",
                capture_output=False,
                text=True,
                timeout=3600  # 1小时超时
            )
            
            if result.returncode != 0:
                print(f"❌ Pipeline 执行失败!")
                print(f"错误输出: {result.stderr}")
                return {
                    "success": False,
                    "error": f"Pipeline执行失败: {result.stderr}"
                }
            
            print("✅ Pipeline 执行成功!")
            print(f"输出: {result.stdout}")
            
            # 检查输出文件
            annotations_path = os.path.join(save_root, "annotations_refined", dataset_name)
            annotations_count = 0
            descriptions = []
            if os.path.exists(annotations_path):
                # 统计生成的标注文件数量
                json_files = list(Path(annotations_path).glob("episode_*.json"))
                annotations_count = len(json_files)
                print(f"📄 生成了 {annotations_count} 个标注文件")
                # 读取所有json文件中的"description"条目
                for json_file in json_files:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if "description" in data:
                            descriptions.append(data["description"])
            else:
                print("⚠️  未找到标注输出文件夹")
            
            print("="*60 + "\n")
            
            return {
                "success": True,
                "output_path": annotations_path,
                "annotations_count": annotations_count,
                "descriptions": descriptions,
                "stdout": result.stdout
            }
            
        except subprocess.TimeoutExpired:
            error_msg = "Pipeline执行超时 (1小时)"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
        except Exception as e:
            error_msg = f"执行Pipeline时发生异常: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }

async def main():
    print("启动场景标注客户端...")
    client = SimpleSceneAnnotationClient("ws://localhost:8769")
    
    try:
        await client.run()
    except KeyboardInterrupt:
        print("客户端停止")
    except Exception as e:
        print(f"错误: {e}")

def run_client():
    """运行客户端"""
    print("🚀 启动场景标注客户端...")
    print("="*60)
    print("📡 连接服务器: ws://172.16.13.93:2080")
    print("🎯 任务类型: scene_annotation")
    print("="*60)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 用户中断，客户端停止")
    except Exception as e:
        print(f"\n❌ 客户端运行错误: {e}")
        import traceback
        traceback.print_exc()

def test_client():
    """测试客户端连接"""
    print("🧪 测试场景标注客户端连接...")
    print("="*60)
    
    async def test_connection():
        client = SimpleSceneAnnotationClient("ws://localhost:8769")
        print("📡 尝试连接服务器...")
        
        try:
            # 简单的连接测试
            await asyncio.wait_for(client.run(), timeout=10.0)
        except asyncio.TimeoutError:
            print("⏰ 连接测试超时（10秒）")
        except Exception as e:
            print(f"❌ 连接测试失败: {e}")
        else:
            print("✅ 连接测试成功")
    
    asyncio.run(test_connection())

if __name__ == "__main__":
    
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "run":
            # 运行客户端
            run_client()
        elif command == "test":
            # 测试连接
            test_client()
        else:
            print("❌ 未知命令")
            print("使用方法:")
            print("  python simple_scene_client.py run   # 运行客户端")
            print("  python simple_scene_client.py test  # 测试连接")
            print("  python simple_scene_client.py       # 显示帮助信息")
    else:
        # 显示帮助信息
        print("🎬 场景标注客户端")
        print("="*60)
        print("使用方法:")
        print("  python simple_scene_client.py run   # 运行客户端")
        print("  python simple_scene_client.py test  # 测试连接")
        print("  python simple_scene_client.py       # 显示帮助信息")
        print()
        print("📝 说明:")
        print("  - run:  启动客户端并连接到服务器处理任务")
        print("  - test: 测试与服务器的连接是否正常")
        print("="*60)
        print()
        print("🚀 默认运行客户端...")
        run_client()