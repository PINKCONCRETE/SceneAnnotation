#!/usr/bin/env python3
"""
数据库连接测试脚本
用于诊断数据库操作卡住的问题
"""

import sys
import os
import time
import signal
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from database import DatasetDatabase
from models import LeformatDatasetSceneAnnotationEmbeddingStatusDB, TaskStatus

def timeout_handler(signum, frame):
    print("❌ 操作超时!")
    raise TimeoutError("数据库操作超时")

def test_database_connection():
    """测试数据库连接和基本操作"""
    print("🔍 开始测试数据库连接...")
    
    db_path = "/mnt/db/datasets.db"
    print(f"📁 数据库路径: {db_path}")
    
    try:
        # 设置超时处理
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30秒超时
        
        # 初始化数据库
        print("🔄 初始化数据库连接...")
        db = DatasetDatabase(Path(db_path))
        print("✅ 数据库初始化成功")
        
        # 测试会话创建
        print("🔄 测试会话创建...")
        with db.with_session() as session:
            print("✅ 会话创建成功")
            
            # 测试简单查询
            print("🔄 测试查询操作...")
            count = session.query(LeformatDatasetSceneAnnotationEmbeddingStatusDB).count()
            print(f"✅ 查询成功，当前记录数: {count}")
            
            # 测试插入操作
            print("🔄 测试插入操作...")
            test_record = LeformatDatasetSceneAnnotationEmbeddingStatusDB(
                dataset_uuid="test-uuid-" + str(int(time.time())),
                convert_path="/test/path",
                status=TaskStatus.COMPLETED,
                err_msg=None
            )
            session.add(test_record)
            print("✅ 记录添加成功")
            
            # 测试提交
            print("🔄 测试事务提交...")
            session.commit()
            print("✅ 事务提交成功")
            
        signal.alarm(0)  # 取消超时
        print("🎉 所有数据库操作测试通过!")
        
    except TimeoutError:
        print("❌ 数据库操作超时，可能存在锁定或连接问题")
        return False
    except Exception as e:
        print(f"❌ 数据库测试失败: {e}")
        return False
    finally:
        signal.alarm(0)  # 确保取消超时
    
    return True

def test_specific_uuid():
    """测试特定的UUID操作"""
    print("\n🔍 测试特定UUID的数据库操作...")
    
    db_path = "/mnt/db/datasets.db"
    test_uuid = "f217b0c1-9f1e-4598-9845-0a85906dadf6"
    
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(15)  # 15秒超时
        
        db = DatasetDatabase(Path(db_path))
        
        with db.with_session() as session:
            print(f"🔍 查询UUID: {test_uuid}")
            
            existing_record = (
                session.query(LeformatDatasetSceneAnnotationEmbeddingStatusDB)
                .filter(LeformatDatasetSceneAnnotationEmbeddingStatusDB.dataset_uuid == test_uuid)
                .first()
            )
            
            if existing_record:
                print(f"✅ 找到现有记录: {existing_record.convert_path}")
                print(f"📊 状态: {existing_record.status}")
            else:
                print("ℹ️  未找到现有记录，将创建新记录")
                
                new_record = LeformatDatasetSceneAnnotationEmbeddingStatusDB(
                    dataset_uuid=test_uuid,
                    convert_path="/mnt/nas/synnas/docker2/scene_annotation_parquet/realman_rmc_aidal_food_storage/parquet/scene_descriptions.parquet",
                    status=TaskStatus.COMPLETED,
                    err_msg=None
                )
                session.add(new_record)
                print("✅ 新记录已添加")
            
            print("🔄 提交事务...")
            session.commit()
            print("✅ 事务提交成功")
            
        signal.alarm(0)
        print("🎉 特定UUID测试通过!")
        return True
        
    except TimeoutError:
        print("❌ 特定UUID操作超时")
        return False
    except Exception as e:
        print(f"❌ 特定UUID测试失败: {e}")
        return False
    finally:
        signal.alarm(0)

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 数据库连接诊断测试")
    print("=" * 60)
    
    # 基本连接测试
    if test_database_connection():
        print("\n" + "=" * 60)
        # 特定UUID测试
        test_specific_uuid()
    
    print("\n" + "=" * 60)
    print("🏁 测试完成")
    print("=" * 60)