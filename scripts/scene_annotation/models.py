from enum import Enum as PyEnum

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func
from sqlalchemy.types import LargeBinary

Base = declarative_base()


# =====================
# 枚举类型
# =====================


class TaskStatus(str, PyEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# =====================
# 主表：DatasetDB
# =====================


class DatasetDB(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    dataset_name = Column(String(255), unique=False, index=True, nullable=False)

    # ✅ 核心：dataset_uuid 作为全局唯一业务标识
    dataset_uuid = Column(String(255), unique=True, index=True, nullable=False)

    device_model = Column(String(100), nullable=False)
    end_effector_type = Column(String(100), nullable=False)
    operation_platform_height = Column(Float, nullable=True)
    yaml_file_path = Column(String(255), nullable=True, unique=True)

    # 多对多关系
    scene_types = relationship(
        "SceneTypeDB", secondary="dataset_scene_types", back_populates="datasets"
    )
    task_descriptions = relationship(
        "TaskDescriptionDB", secondary="dataset_task_descriptions", back_populates="datasets"
    )
    objects = relationship("ObjectDB", secondary="dataset_objects", back_populates="datasets")


# =====================
# 分类与多对多表
# =====================


class SceneTypeDB(Base):
    __tablename__ = "scene_types"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)

    # 反向关系
    datasets = relationship(
        "DatasetDB", secondary="dataset_scene_types", back_populates="scene_types"
    )


class TaskDescriptionDB(Base):
    __tablename__ = "task_descriptions"
    id = Column(Integer, primary_key=True, index=True)
    desc = Column(String(255), unique=True, index=True, nullable=False)

    # 反向关系：注意是 "datasets"，不是 "task_descriptions"
    datasets = relationship(
        "DatasetDB", secondary="dataset_task_descriptions", back_populates="task_descriptions"
    )


class ObjectDB(Base):
    __tablename__ = "object"
    id = Column(Integer, primary_key=True, index=True)
    object_name = Column(String(100), nullable=False, index=True)

    level1_category = Column(String(100), unique=False, nullable=True)
    level2_category = Column(String(100), unique=False, nullable=True)
    level3_category = Column(String(100), unique=False, nullable=True)
    level4_category = Column(String(100), unique=False, nullable=True)
    level5_category = Column(String(100), unique=False, nullable=True)

    # 反向关系：ObjectDB -> DatasetDB 多对多
    datasets = relationship("DatasetDB", secondary="dataset_objects", back_populates="objects")


# =====================
# 多对多关联表
# =====================

dataset_scene_types = Table(
    "dataset_scene_types",
    Base.metadata,
    Column("dataset_id", Integer, ForeignKey("datasets.id"), primary_key=True),
    Column("scene_type_id", Integer, ForeignKey("scene_types.id"), primary_key=True),
)

dataset_task_descriptions = Table(
    "dataset_task_descriptions",
    Base.metadata,
    Column("dataset_id", Integer, ForeignKey("datasets.id"), primary_key=True),
    Column("task_description_id", Integer, ForeignKey("task_descriptions.id"), primary_key=True),
)

dataset_objects = Table(
    "dataset_objects",
    Base.metadata,
    Column("dataset_id", Integer, ForeignKey("datasets.id"), primary_key=True),
    Column("object_id", Integer, ForeignKey("object.id"), primary_key=True),
)


class LeFormatConvertDB(Base):
    __tablename__ = "lerobot_format_convert"

    id = Column(Integer, primary_key=True, index=True)

    # ✅ 使用 dataset_uuid 作为关联字段
    dataset_uuid = Column(String(255), index=True, nullable=False)

    convert_status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=False)
    convert_path = Column(String(255), nullable=True)  # 移除 unique=True，允许多个不同路径

    updated_at = Column(
        DateTime(timezone=True),
        default=func.now(),  # 插入时默认时间
        onupdate=func.now(),  # 更新时自动更新为当前时间
        nullable=False,
    )

    # 📝 新增字段：最后更新信息（可用于记录状态变更详情、错误信息等）
    err_message = Column(
        Text,  # 使用 Text 类型支持较长内容
        nullable=True,  # 允许为空，初始无信息
    )

    version_uuid = Column(String(255), nullable=False)

    # ✅ 唯一约束：一个 uuid 最多一个转换记录
    __table_args__ = (UniqueConstraint("dataset_uuid", name="uix_dataset_uuid_convert"),)


class LeFormatConvertTestDB(Base):
    __tablename__ = "lerobot_format_convert_test"

    id = Column(Integer, primary_key=True, index=True)

    # ✅ 使用 dataset_uuid 作为关联字段
    dataset_uuid = Column(String(255), index=True, nullable=False)

    convert_status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=False)
    convert_path = Column(String(255), nullable=True)  # 移除 unique=True，允许多个不同路径

    updated_at = Column(
        DateTime(timezone=True),
        default=func.now(),  # 插入时默认时间
        onupdate=func.now(),  # 更新时自动更新为当前时间
        nullable=False,
    )

    # 📝 新增字段：最后更新信息（可用于记录状态变更详情、错误信息等）
    err_message = Column(
        Text,  # 使用 Text 类型支持较长内容
        nullable=True,  # 允许为空，初始无信息
    )

    version_uuid = Column(String(255), nullable=True)

    # ✅ 唯一约束：一个 uuid 最多一个转换记录
    __table_args__ = (UniqueConstraint("dataset_uuid", name="uix_dataset_uuid_convert"),)


class DmvAnnotationDB(Base):
    __tablename__ = "device_model_annotation"

    id = Column(Integer, primary_key=True, index=True)

    # 存储 dataset_uuid（字符串格式）
    dataset_uuid = Column(
        String(255), index=True, nullable=False, unique=True
    )  # 改为 255，与 datasets 表一致

    annotation_status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=False)

    annotatio_file_path = Column(String(255), nullable=True)

    device_model = Column(String(255), nullable=True)

    device_model_version = Column(String(255), nullable=True)


class LeformatEpisodeVideoHashDB(Base):
    __tablename__ = "leformat_episode_video_hash"
    id = Column(Integer, primary_key=True, index=True)
    dataset_uuid = Column(String(255), index=True, nullable=False)
    episode_idx = Column(Integer, index=True, nullable=False)
    video_path = Column(String(255), index=True, nullable=False)
    file_hash = Column(String(255), index=True, nullable=False)
    frame_num = Column(Integer, index=True, nullable=False)
    image_hashes = Column(LargeBinary, index=False, nullable=False)


class LeformatEpisodeVideoHashStatusDB(Base):
    __tablename__ = "leformat_episode_video_hash_status"
    id = Column(Integer, primary_key=True, index=True)
    dataset_uuid = Column(String(255), index=True, nullable=False, unique=True)
    convert_path = Column(String(255), index=True, nullable=False)
    status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=False, index=True)


class UrlVideoStAnnotationDB(Base):
    __tablename__ = "url_video_subtask_annotation"
    id = Column(Integer, primary_key=True, index=True)
    video_url = Column(String(255), index=True, nullable=False)
    start_frame_idx = Column(Integer, index=True, nullable=False)
    end_frame_idx = Column(Integer, index=True, nullable=False)
    annotation = Column(String(255), nullable=False)
    __table_args__ = (
        UniqueConstraint(
            "video_url",
            "start_frame_idx",
            "end_frame_idx",
            "annotation",
            name="uix_video_frame_range_annotation",
        ),
    )


class DlVideoDB(Base):
    __tablename__ = "download_videos"
    id = Column(Integer, primary_key=True, index=True)
    video_url = Column(String(255), index=True, nullable=False, unique=True)
    download_path = Column(String(255), index=True, nullable=True, unique=True)
    frame_num = Column(Integer, index=True, nullable=True)
    download_status = Column(
        Enum(TaskStatus), default=TaskStatus.PENDING, nullable=False, index=True
    )
    file_hash = Column(String(255), index=True, nullable=True, unique=True)
    file_hash_status = Column(
        Enum(TaskStatus), default=TaskStatus.PENDING, nullable=True, index=True
    )
    image_hashes = Column(LargeBinary, index=False, nullable=True)
    image_hash_status = Column(
        Enum(TaskStatus), default=TaskStatus.PENDING, nullable=True, index=True
    )


class LeformatEpisodeUrlVideoMatchDB(Base):
    __tablename__ = "leformat_episode_url_video_match"
    id = Column(Integer, primary_key=True, index=True)
    dataset_uuid = Column(String(255), index=True, nullable=False)
    episode_idx = Column(Integer, index=True, nullable=False)
    url_video_id = Column(Integer, ForeignKey("download_videos.id"), nullable=False)


class LeformatEpisodeUrlVideoMatchStatusDB(Base):
    __tablename__ = "leformat_episode_url_video_match_status"
    id = Column(Integer, primary_key=True, index=True)
    dataset_uuid = Column(String(255), index=True, nullable=False, unique=True)
    convert_path = Column(String(255), index=True, nullable=False)
    status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=True, index=True)
    unmatched_episode_indices = Column(Text, index=True, nullable=True)


class LeformatDatasetEpisodeOriginalSubtaskRangeAnnotationDB(Base):
    __tablename__ = "leformat_dataset_episode_original_subtask_annotation"
    id = Column(Integer, primary_key=True, index=True)
    dataset_uuid = Column(String(255), index=True, nullable=False)
    episode_idx = Column(Integer, index=True, nullable=False)
    start_frame_idx = Column(Integer, index=True, nullable=False)
    end_frame_idx = Column(Integer, index=True, nullable=False)
    annotation = Column(String(255), nullable=False)
    __table_args__ = (
        UniqueConstraint(
            "dataset_uuid",
            "episode_idx",
            "start_frame_idx",
            "end_frame_idx",
            "annotation",
            name="uix_dataset_episode_frame_range_annotation",
        ),
    )


class LeformatDatasetEpisodeOriginalSubtaskRangeAnnotationStatusDB(Base):
    __tablename__ = "leformat_dataset_episode_original_subtask_annotation_status"
    id = Column(Integer, primary_key=True, index=True)
    dataset_uuid = Column(String(255), index=True, nullable=False, unique=True)
    convert_path = Column(String(255), index=True, nullable=True)
    status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=False, index=True)
    err_msg = Column(Text, index=True, nullable=True)


class LeformatDatasetEpisodeOptimizedSubtaskRangeAnnotationDB(Base):
    __tablename__ = "leformat_dataset_episode_optimized_subtask_annotation"
    id = Column(Integer, primary_key=True, index=True)
    dataset_uuid = Column(String(255), index=True, nullable=False)
    episode_idx = Column(Integer, index=True, nullable=False)
    start_frame_idx = Column(Integer, index=True, nullable=False)
    end_frame_idx = Column(Integer, index=True, nullable=False)
    annotation = Column(String(255), nullable=False)
    __table_args__ = (
        UniqueConstraint(
            "dataset_uuid",
            "episode_idx",
            "start_frame_idx",
            "end_frame_idx",
            "annotation",
            name="uix_dataset_episode_frame_range_annotation",
        ),
    )


class LeformatDatasetEpisodeOptimizedSubtaskRangeAnnotationStatusDB(Base):
    __tablename__ = "leformat_dataset_episode_optimized_subtask_annotation_status"
    id = Column(Integer, primary_key=True, index=True)
    dataset_uuid = Column(String(255), index=True, nullable=False, unique=True)
    convert_path = Column(String(255), index=True, nullable=True)
    status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=False, index=True)
    err_msg = Column(Text, index=True, nullable=True)


class LeformatDatasetEpisodeSubtaskRangeAnnotationEmbeddingStatusDB(Base):
    __tablename__ = "leformat_dataset_episode_subtask_range_annotation_embedding_status"
    id = Column(Integer, primary_key=True, index=True)
    dataset_uuid = Column(String(255), index=True, nullable=False, unique=True)
    convert_path = Column(String(255), index=True, nullable=True)
    status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=False, index=True)
    err_msg = Column(Text, index=True, nullable=True)


class LeformatDatasetSimReplayStatusDB(Base):
    __tablename__ = "leformat_dataset_simulation_replay_status"
    id = Column(Integer, primary_key=True, index=True)
    dataset_uuid = Column(String(255), index=True, nullable=False, unique=True)
    convert_path = Column(String(255), index=True, nullable=True)
    status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=False, index=True)
    err_msg = Column(Text, index=True, nullable=True)
    version_uuid = Column(String(255), nullable=True)
    prestage_version_uuid = Column(String(255), nullable=True)
    device_model = Column(String(255), index=True, nullable=True)
    device_model_version = Column(String(255), index=True, nullable=True)


class LeformatDatasetMotionAnnotationStatusDB(Base):
    __tablename__ = "leformat_dataset_motion_annotation_status"
    id = Column(Integer, primary_key=True, index=True)
    dataset_uuid = Column(String(255), index=True, nullable=False, unique=True)
    convert_path = Column(String(255), index=True, nullable=True)
    status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=False, index=True)
    err_msg = Column(Text, index=True, nullable=True)


class LeformatDatasetFormatCheckStatusDB(Base):
    __tablename__ = "leformat_dataset_format_check_status"
    id = Column(Integer, primary_key=True, index=True)
    dataset_uuid = Column(String(255), index=True, nullable=False, unique=True)
    convert_path = Column(String(255), index=True, nullable=True)
    status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=False, index=True)
    err_msg = Column(Text, index=True, nullable=True)


class LeformatParquetPostProcessingStatusDB(Base):
    __tablename__ = "leformat_parquet_post_processing_status"
    id = Column(Integer, primary_key=True, index=True)
    dataset_uuid = Column(String(255), index=True, nullable=False, unique=True)
    convert_path = Column(String(255), index=True, nullable=True)
    status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=False, index=True)
    err_msg = Column(Text, index=True, nullable=True)
    prestage_version_uuid = Column(String(255), nullable=True, default="v0")
    version_uuid = Column(String(255), nullable=True, default="v0")
    device_model = Column(String(255), index=True, nullable=True)
    device_model_version = Column(String(255), index=True, nullable=True)

# 场景标注结果
class LeformatDatasetSceneAnnotationDB(Base):
    __tablename__ = "leformat_dataset_scene_annotation"
    id = Column(Integer, primary_key=True, index=True)
    dataset_uuid = Column(String(255), index=True, nullable=False)
    episode_idx = Column(Integer, index=True, nullable=False)
    description = Column(Text, nullable=False)
    object_name = Column(Text, nullable=False)
    box_x_center = Column(Numeric(12, 10), nullable=False)
    box_y_center = Column(Numeric(12, 10), nullable=False)
    box_width = Column(Numeric(12, 10), nullable=False)
    box_height = Column(Numeric(12, 10), nullable=False)
    object_logit = Column(Numeric(10, 8), nullable=False)

# 场景标注状态
class LeformatDatasetSceneAnnotationStatusDB(Base):
    __tablename__ = "leformat_dataset_scene_annotation_status"
    id = Column(Integer, primary_key=True, index=True)
    dataset_uuid = Column(String(255), index=True, nullable=False, unique=True)
    convert_path = Column(String(255), index=True, nullable=True)
    status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=False, index=True)
    err_msg = Column(Text, index=True, nullable=True)
    # 同步版本用，两个值相互独立
    version_uuid = Column(String(255), nullable=True)
    prestage_version_uuid = Column(String(255), nullable=True)
    # 下面两个属性方便查询
    device_model = Column(String(255), index=True, nullable=True)
    device_model_version = Column(String(255), index=True, nullable=True)

# 场景标注embedding状态
class LeformatDatasetSceneAnnotationEmbeddingStatusDB(Base):
    __tablename__ = "leformat_dataset_scene_annotation_embedding_status"
    id = Column(Integer, primary_key=True, index=True)
    dataset_uuid = Column(String(255), index=True, nullable=False, unique=True)
    convert_path = Column(String(255), index=True, nullable=True)
    status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=False, index=True)
    err_msg = Column(Text, index=True, nullable=True)