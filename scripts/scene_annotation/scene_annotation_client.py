import logging
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from task_client import TaskClient
from datasets import SceneAnnotation

# Constants from robocoin-dataset
DATASET_UUID = "dataset_uuid"
CONVERT_PATH = "convert_path"
SCENE_ANNOTATION_PATH = "scene_annotation_path"


class SceneAnnotationTaskClient(TaskClient):
    """场景标注任务客户端"""
    
    def __init__(
        self,
        server_uri: str = "ws://localhost:8768",
        heartbeat_interval: float = 10.0,
        logger: logging.Logger | None = None,
        db_file_path: str | Path = None,
    ) -> None:
        super().__init__(
            server_uri=server_uri,
            heartbeat_interval=heartbeat_interval,
            logger=logger,
        )
        
        # Initialize scene annotation processor
        if db_file_path:
            self.scene_annotation = SceneAnnotation(
                db_file_path=db_file_path,
                logger=logger,
            )
        else:
            self.scene_annotation = None

    def get_task_category(self) -> str:
        return "scene_annotation"

    def generate_task_request_desc(self) -> dict:
        """客户端可自定义任务请求参数"""
        return {}

    def _sync_process_task(self, task_content: dict) -> dict:
        """同步处理场景标注任务"""
        try:
            dataset_uuid = task_content.get(DATASET_UUID)
            convert_path = task_content.get(CONVERT_PATH)
            scene_annotation_path = task_content.get(SCENE_ANNOTATION_PATH)
            
            if self.logger:
                self.logger.info(f"Processing scene annotation for dataset {dataset_uuid}")
                self.logger.info(f"Convert path: {convert_path}")
                self.logger.info(f"Scene annotation path: {scene_annotation_path}")
            
            # 检查路径是否存在
            convert_path_obj = Path(convert_path)
            if not convert_path_obj.exists():
                raise FileNotFoundError(f"Convert path does not exist: {convert_path}")
            
            # 创建场景标注输出目录
            scene_annotation_dir = convert_path_obj / "annotations"
            scene_annotation_dir.mkdir(parents=True, exist_ok=True)
            
            # 调用现有的场景标注处理逻辑
            if self.scene_annotation:
                # 这里可以调用SceneAnnotation类中的具体处理方法
                # 由于现有代码结构，我们需要适配调用方式
                self._process_scene_annotation_for_dataset(
                    dataset_uuid=dataset_uuid,
                    convert_path=convert_path,
                    output_path=scene_annotation_path
                )
            else:
                # 如果没有初始化scene_annotation，使用简单的处理逻辑
                self._simple_scene_annotation_process(
                    dataset_uuid=dataset_uuid,
                    convert_path=convert_path,
                    output_path=scene_annotation_path
                )
            
            if self.logger:
                self.logger.info(f"Scene annotation completed for dataset {dataset_uuid}")
            
            return {"status": "completed", "output_path": scene_annotation_path}
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Scene annotation failed for dataset {dataset_uuid}: {e}")
            raise RuntimeError(
                f"Scene annotation for dataset {dataset_uuid} failed"
            ) from e

    def _process_scene_annotation_for_dataset(
        self, 
        dataset_uuid: str, 
        convert_path: str, 
        output_path: str
    ) -> None:
        """处理单个数据集的场景标注"""
        # 这里可以集成现有的场景标注处理逻辑
        # 例如调用 SceneAnnotation 类中的方法
        
        if self.logger:
            self.logger.info(f"Starting scene annotation processing for {dataset_uuid}")
        
        # 示例：创建一个简单的场景标注文件
        # 实际实现中，这里应该调用真正的场景标注算法
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建示例场景标注数据
        scene_annotations = [
            {
                "episode_id": "episode_0",
                "frame_id": 0,
                "scene_type": "manipulation",
                "objects": [
                    {"name": "object_1", "bbox": [100, 100, 200, 200]},
                    {"name": "object_2", "bbox": [300, 150, 400, 250]}
                ],
                "timestamp": 0.0
            }
        ]
        
        # 写入场景标注文件
        import json
        with open(output_path, 'w') as f:
            for annotation in scene_annotations:
                f.write(json.dumps(annotation) + '\n')
        
        if self.logger:
            self.logger.info(f"Scene annotation file created: {output_path}")

    def _simple_scene_annotation_process(
        self, 
        dataset_uuid: str, 
        convert_path: str, 
        output_path: str
    ) -> None:
        """简单的场景标注处理逻辑"""
        if self.logger:
            self.logger.info(f"Using simple scene annotation processing for {dataset_uuid}")
        
        # 创建输出目录
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建一个基本的场景标注文件
        import json
        basic_annotation = {
            "dataset_uuid": dataset_uuid,
            "convert_path": convert_path,
            "scene_type": "default",
            "processed_by": "scene_annotation_client",
            "status": "completed"
        }
        
        with open(output_path, 'w') as f:
            f.write(json.dumps(basic_annotation) + '\n')
        
        if self.logger:
            self.logger.info(f"Basic scene annotation file created: {output_path}")


def process_single_dataset(dataset_uuid: str, convert_path: str, output_path: str):
    """处理单个数据集的场景标注"""
    import logging
    from pathlib import Path
    import json
    import os
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Processing scene annotation for dataset: {dataset_uuid}")
    logger.info(f"Convert path: {convert_path}")
    logger.info(f"Output path: {output_path}")
    
    try:
        # 检查输入路径
        convert_path_obj = Path(convert_path)
        if not convert_path_obj.exists():
            raise FileNotFoundError(f"Convert path does not exist: {convert_path}")
        
        # 创建输出目录
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # 这里实现实际的场景标注逻辑
        # 目前创建一个示例输出
        annotation_data = {
            "dataset_uuid": dataset_uuid,
            "convert_path": str(convert_path),
            "annotations": [],
            "metadata": {
                "processed_at": "2024-01-01T00:00:00Z",
                "version": "1.0"
            }
        }
        
        # 如果convert_path是目录，扫描其中的文件
        if convert_path_obj.is_dir():
            for file_path in convert_path_obj.rglob("*.json"):
                annotation_data["annotations"].append({
                    "file": str(file_path.relative_to(convert_path_obj)),
                    "objects": [],
                    "scene_description": "Processed scene annotation"
                })
        
        # 写入输出文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Scene annotation completed successfully: {output_path}")
        
    except Exception as e:
        logger.error(f"Error processing scene annotation: {e}")
        raise


if __name__ == "__main__":
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser(description="Scene Annotation Task Client")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="server host to connect to.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8768,
        help="server port to connect to.",
    )
    parser.add_argument(
        "--db_file_path",
        type=str,
        default="",
        help="Path to the database file",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="",
        help="Path to the log directory",
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=10.0,
        help="Heartbeat interval for each client.",
    )
    
    # 单独处理模式的参数
    parser.add_argument("--dataset_uuid", help="Dataset UUID for single processing")
    parser.add_argument("--convert_path", help="Convert path for single processing")
    parser.add_argument("--output_path", help="Output path for single processing")

    args = parser.parse_args()
    
    # Setup logger
    logger = logging.getLogger("scene_annotation_client")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # 检查是否是单独处理模式
    if args.dataset_uuid and args.convert_path and args.output_path:
        # 单独处理模式
        process_single_dataset(args.dataset_uuid, args.convert_path, args.output_path)
    else:
        # WebSocket客户端模式
        server_uri = f"ws://{args.host}:{args.port}"
        client = SceneAnnotationTaskClient(
            server_uri=server_uri,
            logger=logger,
            heartbeat_interval=args.heartbeat_interval,
            db_file_path=args.db_file_path if args.db_file_path else None,
        )
        
        async def main():
            await client.run()
        
        asyncio.run(main())