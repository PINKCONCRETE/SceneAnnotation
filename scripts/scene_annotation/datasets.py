from asyncio import tasks
from sqlalchemy import and_, not_
from sqlalchemy.orm import Session
import uuid
import logging
import json
import re
import logging
import subprocess
import os
import logging
import uuid
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from database import DatasetDatabase
from models import (
    DmvAnnotationDB,
    LeFormatConvertDB,
    LeformatDatasetSceneAnnotationDB,
    LeformatDatasetSceneAnnotationStatusDB,
    LeformatDatasetSceneAnnotationEmbeddingStatusDB,
    TaskStatus,
)

class SceneAnnotation:
    def __init__(
        self,
        db_file_path: str | Path,
        logger: logging.Logger | None = None,
    ) -> None:
        self.db_file_path: Path = Path(db_file_path).expanduser().absolute()
        self.db = DatasetDatabase(self.db_file_path)
        self.logger = logger or logging.getLogger(__name__)
        self.version_uuid = str(uuid.uuid4())


    def _upsert_leformat_dataset_scene_annotation_status(
        self,
        session: Session,
        dataset_uuid: str,
        convert_path: str,
        status: TaskStatus,
        prestage_version_uuid: str,
        device_model: str,
        device_model_version: str,
        err_msg: str = "",
    ) -> None:
        item = (
            session.query(LeformatDatasetSceneAnnotationStatusDB)
            .filter(LeformatDatasetSceneAnnotationStatusDB.dataset_uuid == dataset_uuid)
            .first()
        )
        version_uuid = "v0"
        # version_uuid = self.version_uuid = str(uuid.uuid4())
        if item:
            item.convert_path = convert_path
            item.status = status
            item.prestage_version_uuid = prestage_version_uuid
            item.device_model = device_model
            item.device_model_version = device_model_version
            item.version_uuid = version_uuid
            item.err_msg = err_msg
        else:
            item = LeformatDatasetSceneAnnotationStatusDB(
                dataset_uuid=dataset_uuid,
                convert_path=convert_path,
                status=status,
                prestage_version_uuid=prestage_version_uuid,
                device_model=device_model,
                device_model_version=device_model_version,
                version_uuid=version_uuid,
                err_msg=err_msg,
            )
            session.add(item)
        session.commit()
    
    def _sync_scene_annotation_tasks(
        self, device_model: str | None = None, device_model_version: str | None = None
    ) -> None:
        print("\n" + "="*60)
        print("🔄 开始同步场景标注任务...")
        print("="*60)
        
        with self.db.with_session() as session:
            query = (
                session.query(LeFormatConvertDB)
                .filter(LeFormatConvertDB.convert_status == TaskStatus.COMPLETED)
                .filter(
                    not_(
                        session.query(LeFormatConvertDB)
                        .filter(
                            LeFormatConvertDB.dataset_uuid
                            == LeformatDatasetSceneAnnotationStatusDB.dataset_uuid
                        )
                        .filter(
                            LeFormatConvertDB.version_uuid
                            == LeformatDatasetSceneAnnotationStatusDB.prestage_version_uuid     
                        )
                        .exists()
                    )
                )
            )

            # 从Dmv查询device信息
            device_info = session.query(DmvAnnotationDB).filter(
                 DmvAnnotationDB.dataset_uuid == LeFormatConvertDB.dataset_uuid,
            ).first()

            # 方便查询
            # if device_model is not None:
            #     query = query.filter(
            #         LeFormatConvertDB.device_model == device_model
            #     )

            # if device_model_version is not None:
            #     query = query.filter(
            #         LeFormatConvertDB.device_model_version
            #         == device_model_version
            #     )

        items = query.all()
        print(f"📊 发现 {len(items)} 个需要同步的任务")
        
        if len(items) == 0:
            print("✅ 没有新任务需要同步")
            print("="*60 + "\n")
            return
            
        for i, item in enumerate(items, 1):
            print(f"📝 [{i}/{len(items)}] 同步任务: {item.dataset_uuid[:8]}...")
            with self.db.with_session() as session:
                self._upsert_leformat_dataset_scene_annotation_status(
                    session,
                    dataset_uuid=item.dataset_uuid,
                    convert_path=item.convert_path,
                    status=TaskStatus.PENDING,
                    prestage_version_uuid=item.version_uuid,
                    device_model=device_info.device_model,
                    device_model_version=device_info.device_model_version,
                )
            print(f"   ✅ 任务 {item.dataset_uuid[:8]} 已添加到待处理队列")
        
        print(f"\n🎉 任务同步完成！共同步了 {len(items)} 个任务")
        print("="*60 + "\n")
    
    def _gen_one_scene_annotation_task(
        self, device_model: str, device_model_version: str | None = None
    ) -> tuple[str, str, str, str]:
        with self.db.with_session() as session:
            query = session.query(LeformatDatasetSceneAnnotationStatusDB).filter(
                LeformatDatasetSceneAnnotationStatusDB.status == TaskStatus.PENDING,
            )
            if device_model:
                query = query.filter(LeformatDatasetSceneAnnotationStatusDB.device_model == device_model)
                if device_model_version:
                    query = query.filter(
                        LeformatDatasetSceneAnnotationStatusDB.device_model_version
                        == device_model_version
                    )
            item = query.first()
            if not item:
                return None, None, None, None
            item.status = TaskStatus.PROCESSING
            session.commit()
            return (
                item.dataset_uuid,
                item.prestage_version_uuid,
                item.device_model,
                item.device_model_version,
            )

    def _get_convert_path(self, dataset_uuid: str) -> str:
        with self.db.with_session() as session:
            item = (
                session.query(LeFormatConvertDB)
                .filter(LeFormatConvertDB.dataset_uuid == dataset_uuid)
                .first()
            )
            if item:
                return item.convert_path
            return None

    def scene_annotation_datasets(self, device_model: str, device_model_version: str = "") -> None:
        print("\n" + "🌟 " + "="*58)
        print("🎬 场景标注数据集处理开始")
        print(f"🔧 设备型号: {device_model}")
        print(f"📱 设备版本: {device_model_version or '未指定'}")
        print("="*60)
        
        self._sync_scene_annotation_tasks(device_model)
        
        print("🔍 正在获取待处理任务...")
        dataset_uuid, prestage_version_uuid, device_model, device_model_version = (
            self._gen_one_scene_annotation_task(
                device_model=device_model, device_model_version=device_model_version
            )
        )

        if dataset_uuid is None:
            print("✅ 没有待处理的任务")
            print("="*60 + "\n")
            return
            
        print(f"📦 找到待处理数据集: {dataset_uuid}")
        print(f"🔗 预处理版本: {prestage_version_uuid}")
        print("-" * 50)
        print("🔍 正在获取数据集转换路径...")
        convert_path = self._get_convert_path(dataset_uuid)
        print(f"📁 转换路径: {convert_path}")
        
        try:
            print("🚀 开始执行场景标注处理...")
            self._scene_annotation_dataset()
            
            print("💾 正在更新最终状态...")
            with self.db.with_session() as session:
                self._upsert_leformat_dataset_scene_annotation_status(
                    session=session,
                    dataset_uuid=dataset_uuid,
                    convert_path=convert_path,
                    prestage_version_uuid=prestage_version_uuid,
                    device_model=device_model,
                    device_model_version=device_model_version,
                    status=TaskStatus.COMPLETED,
                )
            
            print("\n🎉 " + "="*58)
            print("✨ 场景标注数据集处理完成!")
            print(f"📦 数据集: {dataset_uuid}")
            print(f"✅ 状态: COMPLETED")
            print("="*60 + "\n")

        except Exception as e:
            print(f"\n❌ 处理过程中发生错误: {e}")
            print("🔄 正在更新错误状态...")
            
            with self.db.with_session() as session:
                self._upsert_leformat_dataset_scene_annotation_status(
                    session=session,
                    dataset_uuid=dataset_uuid,
                    convert_path=convert_path,
                    status=TaskStatus.FAILED,
                    prestage_version_uuid=prestage_version_uuid,
                    device_model=device_model,
                    device_model_version=device_model_version,
                    err_msg=str(e),
                )
            
            print("\n💥 " + "="*58)
            print("❌ 场景标注数据集处理失败!")
            print(f"📦 数据集: {dataset_uuid}")
            print(f"❌ 状态: FAILED")
            print(f"🔍 错误信息: {str(e)}")
            print("="*60 + "\n")
    
    def _scene_upload_local_dataset(self, base_path: str)->None:
        """根据本地标注文件夹名称查找对应的dataset_uuid并上传标注数据"""
        # annotations_base_path = Path("/home/diy02/RoboCoin-scene-annotator/results/annotations_refined")
        annotations_base_path = Path(base_path)
        if not annotations_base_path.exists():
            self.logger.error(f"标注目录不存在: {annotations_base_path}")
            return
            
        with self.db.with_session() as session:
            # 遍历标注目录下的所有文件夹
            for dataset_folder in annotations_base_path.iterdir():
                if not dataset_folder.is_dir():
                    continue

                dataset_name = dataset_folder.name
                self.logger.info(f"处理数据集: {dataset_name}")
                print(20*"=" + f"数据集{dataset_name} 开始!" + 20*"=")
                
                # 数据集名查找dataset_uuid
                dataset_record = (
                    session.query(LeFormatConvertDB)
                    # 转换路径中包含数据集名 匹配正则表达式:*dataset_name*
                    .filter(LeFormatConvertDB.convert_path.like(f'%{dataset_name}%'))
                    .first()
                )
                
                if dataset_record is None:
                    print(f"未找到数据集 {dataset_name} 对应的UUID记录")
                    continue
                    
                dataset_uuid = dataset_record.dataset_uuid
                print(f"找到数据集UUID: {dataset_uuid}")
                # 如果该数据集已经有标注完成的记录，跳过
                status_record = (
                    session.query(LeformatDatasetSceneAnnotationStatusDB)
                    .filter(LeformatDatasetSceneAnnotationStatusDB.dataset_uuid == dataset_uuid)
                    .first()
                )
                if status_record and status_record.status == TaskStatus.COMPLETED:
                    print(f"数据集 {dataset_name} 已完成标注，跳过")
                    continue
                
                # 扫描该数据集文件夹下的所有JSON文件
                self._process_dataset_annotations(session, dataset_folder, dataset_uuid, dataset_name)

                # 从前一级建立关系
                pre_record = (
                    session.query(LeFormatConvertDB)
                    .filter(LeFormatConvertDB.dataset_uuid == dataset_uuid)
                    .first()
                )
                # DeviceModel获取
                device_record = (
                    session.query(DmvAnnotationDB)
                    .filter(DmvAnnotationDB.dataset_uuid == pre_record.dataset_uuid)
                    .first()
                )
                if pre_record is None:
                    self.logger.error(f"未找到数据集 {dataset_uuid} 对应的转换记录")
                    continue

                # 状态数据库更新为完成
                self._upsert_leformat_dataset_scene_annotation_status(
                    session=session,
                    dataset_uuid=dataset_uuid,
                    convert_path=f"/mnt/nas/synnas/docker2/robocoin-datasets/{dataset_name}",
                    status=TaskStatus.COMPLETED,
                    prestage_version_uuid=pre_record.version_uuid,
                    device_model=device_record.device_model,
                    device_model_version=device_record.device_model_version,
                )
                print(20*"=" + f"数据集{dataset_name} 完成!" + 20*"=")

    def _process_dataset_annotations(self, session: Session, dataset_folder: Path, dataset_uuid: str, dataset_name: str) -> None:
        try:
            # 扫描文件夹中的所有episode_*.json文件
            json_files = list(dataset_folder.glob("episode_*.json"))
            self.logger.info(f"在数据集 {dataset_folder.name} 中找到 {len(json_files)} 个JSON文件")
            
            for json_file in json_files:
                try:
                    # 从文件名中提取episode_idx
                    match = re.match(r'episode_(\d+)\.json', json_file.name)
                    if not match:
                        self.logger.warning(f"无法从文件名 {json_file.name} 中提取episode索引")
                        continue
                    
                    episode_idx = int(match.group(1))
                    print(f"处理文件: {dataset_name}/{json_file.name}")
                    
                    # 读取JSON文件
                    with open(json_file, 'r', encoding='utf-8') as f:
                        annotation_data = json.load(f)

                    # 解析成功后先删除该episode_idx的所有标注记录
                    session.query(LeformatDatasetSceneAnnotationDB) \
                        .filter(LeformatDatasetSceneAnnotationDB.dataset_uuid == dataset_uuid,
                                LeformatDatasetSceneAnnotationDB.episode_idx == episode_idx) \
                        .delete()
                    session.commit()
                    # 保存到数据库
                    self._save_annotation_to_db(session, dataset_uuid, episode_idx, annotation_data)

                    # 转存json_file 到nas, 名字不变，文件夹位dataset_name
                    nas_annotation_path = Path(f"/mnt/nas/synnas/docker2/robocoin-datasets/{dataset_name}/annotations/{json_file.name}")
                    nas_annotation_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(nas_annotation_path, 'w', encoding='utf-8') as f:
                        json.dump(annotation_data, f, ensure_ascii=False, indent=4)
                    print(f"{dataset_name}/{json_file.name} Finish!")

                except json.JSONDecodeError as e:
                    self.logger.error(f"解析JSON文件 {json_file} 时出错: {e}")
                    continue
                except Exception as e:
                    self.logger.error(f"处理文件 {json_file} 时出错: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"处理数据集文件夹 {dataset_folder} 时出错: {e}")
            raise
    
    def _save_annotation_to_db(self, session: Session, dataset_uuid: str, episode_idx: int, annotation_data: dict) -> None:
        """将标注数据保存到数据库"""
        try:
            # 获取描述信息
            description = annotation_data.get("description", "")
            
            # 处理每个检测到的对象
            objects = annotation_data.get("object", [])
            for obj in objects:
                # 创建标注记录
                annotation_record = LeformatDatasetSceneAnnotationDB(
                    dataset_uuid=dataset_uuid,
                    episode_idx=episode_idx,
                    description=description,
                    object_name=obj.get("name", ""),
                    box_x_center=float(obj["box"]["x_center"]),
                    box_y_center=float(obj["box"]["y_center"]),
                    box_width=float(obj["box"]["width"]),
                    box_height=float(obj["box"]["height"]),
                    object_logit=float(obj.get("logit", 0.0)),
                )
                
                # 直接添加标注记录（已去掉唯一约束）
                session.add(annotation_record)
                self.logger.debug(f"添加标注记录: dataset_uuid={dataset_uuid}, episode_idx={episode_idx}, object={obj.get('name', '')}")
            
            session.commit()
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"保存标注数据到数据库时出错: {e}")
            raise

    def _scene_annotation_dataset(self) -> None:
        print("\n" + "🚀 " + "="*58)
        print("🎯 开始执行场景标注数据集处理流程")
        print("="*60)
        
        self._sync_scene_annotation_tasks()# 同步任务列表
        
        task_count = 0
        # 直到所有pending都被处理完
        while True:
            dataset_uuid, prestage_version_uuid, device_model, device_model_version = self._gen_one_scene_annotation_task(device_model=None, device_model_version=None)
            if dataset_uuid is None:
                print(f"\n🏁 所有任务处理完成！共处理了 {task_count} 个数据集")
                print("="*60 + "\n")
                break
            
            task_count += 1
            print(f"\n🔥 开始处理第 {task_count} 个数据集...")
            print(f"📦 数据集UUID: {dataset_uuid}")
            print("-" * 50)
            
            with self.db.with_session() as session:
                current_item = None
                try:
                    print("查询数据库中数据集状态...")
                    # 重新查询任务状态，确保数据最新
                    current_item = session.query(LeformatDatasetSceneAnnotationStatusDB).filter(
                        LeformatDatasetSceneAnnotationStatusDB.dataset_uuid == dataset_uuid
                    ).first()
                    print(f"当前数据集状态: {current_item.status if current_item else 'None'}")
                    
                    if current_item is None and current_item.status != TaskStatus.COMPLETED:
                        continue
                    
                    # 更新任务状态为processing
                    current_item.status = "PROCESSING"
                    session.commit()
                    print("   ⚡ 任务状态已更新为 PROCESSING")
                    
                    # 提取数据集路径
                    print("   🔍 正在查找数据集路径...")
                    pre_datasets = (
                        session.query(LeFormatConvertDB.convert_path) 
                        .filter(LeFormatConvertDB.dataset_uuid == dataset_uuid) 
                        .filter(LeFormatConvertDB.convert_status == TaskStatus.COMPLETED)
                        .first()
                    )
                    print(f"   🔍 转换数据集路径: {pre_datasets.convert_path if pre_datasets else 'None'}")
                    
                    if pre_datasets is None:
                        raise Exception(f"未找到dataset_uuid {dataset_uuid} 对应的转换完成的数据集")
                    
                    dataset_path = pre_datasets[0]  # convert_path
                    dataset_name = os.path.basename(dataset_path)
                    # 获取数据集的父目录作为repo_root，数据集名作为repo_id
                    repo_root = os.path.dirname(dataset_path) + "/"
                    repo_id = dataset_name
                    
                    print(f"   📁 数据集完整路径: {dataset_path}")
                    print(f"   📂 repo_root: {repo_root}")
                    print(f"   🏷️  repo_id: {repo_id}")
                    
                    # 执行pipeline
                    print("   🔧 准备执行 Pipeline...")
                    pipeline_script = "/home/diy02/RoboCoin-scene-annotator/scripts/run_pipeline.py"
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
                    
                    print("   ⚙️  Pipeline 配置:")
                    print(f"      📂 repo_root: {repo_root}")
                    print(f"      🏷️  repo_id: {repo_id}")
                    print(f"      💾 输出路径: {save_root}")
                    print(f"      🤖 模型: deepseek-r1:14b")
                    print(f"      🎯 检测器: grounding_dino (CUDA)")
                    
                    self.logger.info(f"开始执行pipeline: {' '.join(cmd)}")
                    print("   🚀 正在执行 Pipeline... (这可能需要一些时间)")
                    
                    # 执行pipeline命令
                    result = subprocess.run(
                        cmd,
                        cwd="/home/diy02/RoboCoin-scene-annotator",
                        capture_output=True,
                        text=True,
                        timeout=3600  # 1小时超时
                    )
                    
                    if result.returncode != 0:
                        print(f"   ❌ Pipeline 执行失败!")
                        raise Exception(f"Pipeline执行失败: {result.stderr}")
                    
                    print("   ✅ Pipeline 执行成功!")
                    self.logger.info(f"Pipeline执行成功: {result.stdout}")
                    
                    # Pipeline执行成功，调用上传方法
                    print("   📤 开始上传标注结果...")
                    annotations_path = os.path.join(save_root, "annotations_refined")
                    self._scene_upload_local_dataset(annotations_path)
                    print("   ✅ 标注结果上传完成!")
                    
                    # 更新任务状态为completed
                    current_item.status = "completed"
                    current_item.err_msg = None  # 清空错误信息
                    session.commit()
                    print("   🎉 任务状态已更新为 COMPLETED")
                    
                    self.logger.info(f"数据集 {dataset_uuid} 处理完成并上传成功")
                    print(f"   🏆 数据集 {dataset_name} 处理完成!")
                    print("-" * 50)
                    
                except Exception as e:
                    # 处理失败，将状态改为failed并记录错误信息
                    error_msg = str(e)
                    print(f"   ❌ 处理失败: {error_msg}")
                    self.logger.error(f"处理数据集 {dataset_uuid} 时出错: {error_msg}")
                    
                    try:
                        if current_item is not None:
                            current_item.status = "failed"
                            current_item.err_msg = error_msg
                            session.commit()
                            print("   🔄 任务状态已更新为 FAILED")
                            self.logger.info(f"已将数据集 {dataset_uuid} 状态更新为failed")
                    except Exception as commit_error:
                        session.rollback()
                        print(f"   ⚠️  数据库更新失败: {commit_error}")
                        self.logger.error(f"更新数据库状态时出错: {commit_error}")
                    
                    print("-" * 50)
                    continue

    def convert_json_to_parquet(self, annotations_folder: str, dataset_name: str) -> str:
        """
        将指定文件夹下的 JSON 标注文件转换为对应的 parquet 格式，每个 episode_*.json 对应一个 episode_*.parquet
        根据 episodes.jsonl 文件中的 length 信息扩展 parquet 文件到相应长度
        
        Args:
            annotations_folder: 包含 episode_*.json 文件的文件夹路径
            dataset_name: 数据集名称
            
        Returns:
            str: parquet 文件夹的路径
        """
        print("\n" + "📦 " + "="*58)
        print("🔄 开始将 JSON 标注文件转换为 parquet 格式")
        print("="*60)
        
        annotations_path = Path(annotations_folder)
        if not annotations_path.exists():
            raise FileNotFoundError(f"标注文件夹不存在: {annotations_folder}")
        
        # 设置新的输出路径结构
        base_output_path = Path(f"/mnt/nas/synnas/docker2/robocoin-datasets/{dataset_name}")
        annotations_output_path = base_output_path / "annotations"
        parquet_output_path = base_output_path / "scene_annotation_data"
        meta_path = base_output_path / "meta"
        
        # 创建输出文件夹
        annotations_output_path.mkdir(parents=True, exist_ok=True)
        parquet_output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 输入文件夹: {annotations_path}")
        print(f"📁 数据集输出路径: {base_output_path}")
        print(f"📁 标注输出路径: {annotations_output_path}")
        print(f"📁 Parquet输出路径: {parquet_output_path}")
        
        # 读取 episodes.jsonl 文件获取每个 episode 的长度信息
        episodes_jsonl_path = meta_path / "episodes.jsonl"
        episode_lengths = {}
        
        if episodes_jsonl_path.exists():
            print(f"📄 读取 episodes.jsonl 文件: {episodes_jsonl_path}")
            try:
                with open(episodes_jsonl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        episode_data = json.loads(line.strip())
                        episode_index = episode_data.get("episode_index")
                        length = episode_data.get("length", 1)  # 默认长度为1
                        if episode_index is not None:
                            episode_lengths[episode_index] = length
                print(f"📊 读取到 {len(episode_lengths)} 个 episode 的长度信息")
            except Exception as e:
                print(f"⚠️  读取 episodes.jsonl 文件失败: {e}")
                self.logger.warning(f"读取 episodes.jsonl 文件失败: {e}")
        else:
            print(f"⚠️  未找到 episodes.jsonl 文件: {episodes_jsonl_path}")
        
        # 查找所有 episode_*.json 文件
        json_files = list(annotations_path.glob("episode_*.json"))
        if not json_files:
            print("⚠️  未找到任何 episode_*.json 文件")
            return None
        
        # 按文件名排序，确保顺序正确
        json_files.sort(key=lambda x: int(re.search(r'episode_(\d+)', x.name).group(1)))
        
        print(f"📊 找到 {len(json_files)} 个 JSON 文件")
        print("-" * 50)
        
        # 收集所有场景标注数据
        scene_annotations = []
        processed_files = []
        
        for idx, json_file in enumerate(tqdm(json_files, desc="处理 JSON 文件", unit="文件")):
            try:
                # 读取 JSON 文件
                with open(json_file, 'r', encoding='utf-8') as f:
                    annotation_data = json.load(f)
                
                # 提取 description 字段
                description = annotation_data.get("description", "")
                
                # 创建场景标注字典
                scene_annotation_dict = {
                    "scene_annotation_index": idx,
                    "scene_annotation": description
                }
                scene_annotations.append(scene_annotation_dict)
                
                # 为每个 episode 创建单独的 parquet 文件
                episode_match = re.search(r'episode_(\d+)', json_file.name)
                if episode_match:
                    episode_num = int(episode_match.group(1))
                    episode_num_str = episode_match.group(1)
                    
                    # 获取该 episode 的长度，如果没有找到则默认为1
                    episode_length = episode_lengths.get(episode_num, 1)
                    
                    # 创建单个 episode 的 DataFrame，根据 episode 长度扩展
                    # 所有行都使用相同的 scene_annotation 索引值
                    episode_df = pd.DataFrame({
                        "scene_annotation": [idx] * episode_length
                    })
                    
                    # 保存为对应的 parquet 文件
                    parquet_filename = f"episode_{episode_num_str}.parquet"
                    episode_parquet_path = parquet_output_path / parquet_filename
                    episode_df.to_parquet(episode_parquet_path, index=False)
                    
                    processed_files.append(parquet_filename)
                    
                    self.logger.debug(f"处理文件: {json_file.name} -> {parquet_filename}, index: {idx}, length: {episode_length}")
                    
                    if episode_length > 1:
                        print(f"   📏 Episode {episode_num_str}: 扩展到长度 {episode_length}")
                
            except json.JSONDecodeError as e:
                self.logger.error(f"解析 JSON 文件 {json_file} 时出错: {e}")
                # 添加空的场景标注保持索引一致
                scene_annotation_dict = {
                    "scene_annotation_index": idx,
                    "scene_annotation": ""
                }
                scene_annotations.append(scene_annotation_dict)
                
                # 即使出错也要创建对应的 parquet 文件
                episode_match = re.search(r'episode_(\d+)', json_file.name)
                if episode_match:
                    episode_num = int(episode_match.group(1))
                    episode_num_str = episode_match.group(1)
                    episode_length = episode_lengths.get(episode_num, 1)
                    
                    episode_df = pd.DataFrame({
                        "scene_annotation": [idx] * episode_length
                    })
                    
                    parquet_filename = f"episode_{episode_num_str}.parquet"
                    episode_parquet_path = parquet_output_path / parquet_filename
                    episode_df.to_parquet(episode_parquet_path, index=False)
                    processed_files.append(parquet_filename)
                    
            except Exception as e:
                self.logger.error(f"处理文件 {json_file} 时出错: {e}")
                # 添加空的场景标注保持索引一致
                scene_annotation_dict = {
                    "scene_annotation_index": idx,
                    "scene_annotation": ""
                }
                scene_annotations.append(scene_annotation_dict)
                
                # 即使出错也要创建对应的 parquet 文件
                episode_match = re.search(r'episode_(\d+)', json_file.name)
                if episode_match:
                    episode_num = int(episode_match.group(1))
                    episode_num_str = episode_match.group(1)
                    episode_length = episode_lengths.get(episode_num, 1)
                    
                    episode_df = pd.DataFrame({
                        "scene_annotation": [idx] * episode_length
                    })
                    
                    parquet_filename = f"episode_{episode_num_str}.parquet"
                    episode_parquet_path = parquet_output_path / parquet_filename
                    episode_df.to_parquet(episode_parquet_path, index=False)
                    processed_files.append(parquet_filename)
        
        # 保存 scene_annotation.jsonl 文件
        jsonl_file_path = annotations_output_path / "scene_annotation.jsonl"
        with open(jsonl_file_path, 'w', encoding='utf-8') as f:
            for annotation in scene_annotations:
                f.write(json.dumps(annotation, ensure_ascii=False) + '\n')
        
        # 移除索引映射文件的生成
        
        print(f"\n✅ 转换完成!")
        print(f"📄 共处理 {len(scene_annotations)} 个描述")
        print(f"📁 生成 {len(processed_files)} 个 parquet 文件")
        print(f"💾 JSONL文件: {jsonl_file_path}")
        print(f"💾 Parquet文件夹: {parquet_output_path}")
        print(f"📊 JSONL文件大小: {jsonl_file_path.stat().st_size / 1024:.2f} KB")
        
        # 计算所有 parquet 文件的总大小
        total_parquet_size = sum(
            (parquet_output_path / pf).stat().st_size 
            for pf in processed_files 
            if (parquet_output_path / pf).exists()
        )
        print(f"📊 Parquet文件总大小: {total_parquet_size / 1024:.2f} KB")
        print("="*60 + "\n")
        
        self.logger.info(f"成功将 {len(scene_annotations)} 个 JSON 文件转换为 {len(processed_files)} 个 parquet 文件: {parquet_output_path}")
        self.logger.info(f"成功创建场景标注 JSONL 文件: {jsonl_file_path}")
        
        return str(parquet_output_path)

    def convert_dataset_json_to_parquet(self, base_annotations_folder: str) -> None:
        """
        批量处理多个数据集的 JSON 标注文件，转换为 parquet 格式
        
        Args:
            base_annotations_folder: 包含多个数据集文件夹的基础路径
        """
        print("\n" + "🌟 " + "="*58)
        print("🔄 开始批量转换数据集 JSON 标注文件为 parquet 格式")
        print("="*60)
        
        base_path = Path(base_annotations_folder)
        if not base_path.exists():
            raise FileNotFoundError(f"基础标注文件夹不存在: {base_annotations_folder}")
        
        # 查找所有数据集文件夹
        dataset_folders = [d for d in base_path.iterdir() if d.is_dir()]
        if not dataset_folders:
            print("⚠️  未找到任何数据集文件夹")
            return
        
        print(f"📊 找到 {len(dataset_folders)} 个数据集文件夹")
        print("-" * 50)
        
        success_count = 0
        failed_count = 0
        
        for dataset_folder in tqdm(dataset_folders, desc="处理数据集", unit="数据集"):
            try:
                dataset_name = dataset_folder.name
                print(f"\n🔥 处理数据集: {dataset_name}")
                
                # 首先检查是否已经完成，避免重复转换
                try:
                    print(f"   🔍 检查数据集 {dataset_name} 的完成状态...")
                    with self.db.with_session() as session:
                        # 数据集名查找dataset_uuid
                        dataset_record = (
                            session.query(LeFormatConvertDB)
                            # 转换路径中包含数据集名 匹配正则表达式:*dataset_name*
                            .filter(LeFormatConvertDB.convert_path.like(f'%{dataset_name}%'))
                            .first()
                        )
                        
                        if not dataset_record:
                            print(f"   ⚠️  未找到数据集 {dataset_name} 的记录，跳过")
                            failed_count += 1
                            continue
                            
                        # 检查embedding状态是否已经完成
                        embedding_status_record = (
                            session.query(LeformatDatasetSceneAnnotationEmbeddingStatusDB)
                            .filter(LeformatDatasetSceneAnnotationEmbeddingStatusDB.dataset_uuid == dataset_record.dataset_uuid)
                            .first()
                        )
                        
                        if embedding_status_record and embedding_status_record.status == TaskStatus.COMPLETED:
                            print(f"   ✅ 数据集 {dataset_name} 已完成转换，跳过")
                            continue
                            
                except Exception as check_error:
                    print(f"   ❌ 检查完成状态失败: {check_error}")
                    self.logger.error(f"检查数据集 {dataset_name} 状态时出错: {check_error}")
                    # 继续处理，不让检查错误阻止转换
                except Exception as check_error:
                    print(f"   ❌ 检查状态失败: {check_error}")
                    self.logger.error(f"检查数据集 {dataset_name} 状态时出错: {check_error}")
                    failed_count += 1
                    continue
                
                # 转换当前数据集，传递 dataset_name 参数
                parquet_path = self.convert_json_to_parquet(str(dataset_folder), dataset_name)
                
                if parquet_path is None:
                    print(f"   ⚠️  数据集 {dataset_name} 转换失败，跳过")
                    failed_count += 1
                    continue
                    
                success_count += 1
                print(f"   ✅ {dataset_name} 转换成功")
                
                # 将处理结果添加到embedding数据库，给出转换结果路径和dataset_uuid
                try:
                    print(f"   🔍 查询数据集 {dataset_name} 的UUID...")
                    with self.db.with_session() as session:
                        # 重新查询dataset_record（因为前面的session已经关闭）
                        dataset_record = (
                            session.query(LeFormatConvertDB)
                            .filter(LeFormatConvertDB.convert_path.like(f'%{dataset_name}%'))
                            .first()
                        )
                        
                        if dataset_record:
                            dataset_uuid = dataset_record.dataset_uuid
                            print(f"   📋 找到数据集UUID: {dataset_uuid}")
                            print(f"   📁 Parquet文件路径: {parquet_path}")
                            
                            # 添加处理结果到LeformatDatasetSceneAnnotationEmbeddingStatusDB中
                            # 使用同一个session，避免重复打开
                            self.add_embedding_to_db(session, dataset_uuid, parquet_path)
                        else:
                            print(f"   ⚠️  未找到数据集 {dataset_name} 的记录，跳过数据库更新")
                            failed_count += 1
                            continue
                except Exception as db_error:
                    print(f"   ❌ 数据库查询失败: {db_error}")
                    self.logger.error(f"查询数据集 {dataset_name} 的UUID时出错: {db_error}")
                    failed_count += 1
                    # 继续处理下一个数据集，不让数据库错误中断整个流程
            except Exception as e:
                failed_count += 1
                print(f"   ❌ {dataset_folder.name} 转换失败: {e}")
                self.logger.error(f"转换数据集 {dataset_folder.name} 时出错: {e}")
        
        print(f"\n🎉 批量转换完成!")
        print(f"✅ 成功: {success_count} 个数据集")
        print(f"❌ 失败: {failed_count} 个数据集")
        print("="*60 + "\n")

    def add_embedding_to_db(self, session: Session, dataset_uuid: str, parquet_path: str) -> None:
        """
        添加处理结果到LeformatDatasetSceneAnnotationEmbeddingStatusDB中
        
        Args:
            session: 数据库会话
            dataset_uuid: 数据集UUID
            parquet_path: parquet文件路径
        """
        print(f"   🔄 开始添加数据集 {dataset_uuid} 的embedding状态记录到数据库")
        
        try:
            # 查找是否已存在记录
            print(f"   🔍 查询数据集 {dataset_uuid} 的现有记录...")
            existing_record = (
                session.query(LeformatDatasetSceneAnnotationEmbeddingStatusDB)
                .filter(LeformatDatasetSceneAnnotationEmbeddingStatusDB.dataset_uuid == dataset_uuid)
                .first()
            )
            
            if existing_record:
                # 更新现有记录
                print(f"   🔄 更新现有记录...")
                existing_record.convert_path = parquet_path
                existing_record.status = TaskStatus.COMPLETED
                existing_record.err_msg = None
                self.logger.info(f"更新数据集 {dataset_uuid} 的embedding状态记录")
            else:
                # 创建新记录
                print(f"   ➕ 创建新记录...")
                new_record = LeformatDatasetSceneAnnotationEmbeddingStatusDB(
                    dataset_uuid=dataset_uuid,
                    convert_path=parquet_path,
                    status=TaskStatus.COMPLETED,
                    err_msg=None
                )
                session.add(new_record)
                self.logger.info(f"创建数据集 {dataset_uuid} 的embedding状态记录")
            
            print(f"   💾 提交数据库事务...")
            session.commit()
            print(f"   ✅ 数据库操作完成")
            print(f"   📊 已将embedding结果记录到数据库: {parquet_path}")
                
        except Exception as e:
            print(f"   ❌ 数据库操作失败: {e}")
            self.logger.error(f"添加embedding状态记录到数据库时出错: {e}")
            session.rollback()  # 回滚事务
            # 不抛出异常，让程序继续运行
            return

if __name__ == "__main__":
    # 初始化场景标注处理器
    scene_annotation = SceneAnnotation("/mnt/nas/synnas/docker2/database-backups/10.30/datasets.db")
    
    # 选择运行模式
    # 模式1: 处理pending的任务（场景标注流程）
    # scene_annotation.scene_annotation_datasets(device_model=None, device_model_version=None)
    
    # 模式2: 将NAS上已完成的JSON标注文件转换为parquet格式
    print("🚀 开始将NAS上的JSON标注文件转换为parquet格式...")
    
    # 首先检查NAS上的数据集结构
    nas_base_path = Path("/mnt/nas/synnas/docker2/scene_annotation")
    if nas_base_path.exists():
        dataset_folders = [d for d in nas_base_path.iterdir() if d.is_dir()]
        print(f"📊 在NAS上发现 {len(dataset_folders)} 个数据集文件夹:")
        for folder in dataset_folders:
            json_files = list(folder.glob("episode_*.json"))
            print(f"   📁 {folder.name}: {len(json_files)} 个JSON文件")
    else:
        print(f"❌ NAS路径不存在: {nas_base_path}")
        exit(1)
    
    print("-" * 60)
    
    # 选项A: 转换单个数据集（取消注释以使用）
    # dataset_name = "agilex_cobot_decoupled_magic_basket_stoarge_egg"
    # scene_annotation.convert_json_to_parquet(
    #     annotations_folder=f"/mnt/nas/synnas/docker2/scene_annotation/{dataset_name}",
    #     dataset_name=dataset_name
    # )
    
    # 选项B: 批量转换所有数据集
    scene_annotation.convert_dataset_json_to_parquet(
        base_annotations_folder="/mnt/nas/synnas/docker2/scene_annotation"
    )
    
    print("✅ 所有数据集的JSON到parquet转换完成！")