import json
import ollama
import argparse
import os
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional

class DatabaseManager:
    def __init__(self, db_path: str = "/mnt/nas/synnas/docker2/code_resource/fix.db"):
        self.db_path = db_path
        self.conn = None
        self.init_db()
    
    def init_db(self):
        """初始化数据库连接并创建表"""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL CHECK(
                    status IN ('pending', 'processing', 'failed', 'completed')
                ),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
    
    def get_file_status(self, file_path: str) -> Optional[str]:
        """获取文件的处理状态"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT status FROM file_tasks WHERE file_path = ?", (file_path,))
        result = cursor.fetchone()
        return result[0] if result else None
    
    def add_or_update_file(self, file_path: str, status: str):
        """添加或更新文件状态"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO file_tasks (file_path, status, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(file_path) 
            DO UPDATE SET status = ?, updated_at = CURRENT_TIMESTAMP
        """, (file_path, status, status))
        self.conn.commit()
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()

class TextFixer:
    def __init__(self, model_name: str = "deepseek-r1:14b"):
        self.model_name = model_name

    def fix_text(self, text: str) -> str:
        # 检查是否包含任何需要处理的特殊符号，只要有就送给大模型处理
        special_chars = [
            '\n',      # 换行符
            '\t',      # 制表符
            '\r',      # 回车符
            '```',     # 代码块
            '`',       # 反引号
            '**',      # 加粗
            '*',       # 星号（斜体/列表）
            '_',       # 下划线（斜体）
            '#',       # 井号（标题）
            '>',       # 引用
        ]
        
        # 检查是否以特殊字符开头或包含特殊字符
        if (text.startswith('#') or 
            text.startswith('-') or 
            text.startswith('>') or
            any(char in text for char in special_chars)):
            
            prompt = f"Please convert the following text to plain text format, remove all extra symbols and formatting requirements (including Markdown format), retain only the plain text content without any line breaks, and output the result as a single line with complete English sentences:\n\n {text}"
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return response['message']['content'].strip()
        
        return text
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="/mnt/nas/synnas/docker2/code_resource/fix.db",
        help="Path to the SQLite database file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocess files even if they are marked as completed",
    )
    args = parser.parse_args()
    
    # 初始化数据库管理器和文本修复器
    db_manager = DatabaseManager(args.db_path)
    text_fixer = TextFixer()
    print(f"Processing dataset: {args.path}")
    print(f"Database: {args.db_path}")
    
    try:
        # 只扫描args.path下一层的文件夹，查找annotations/scene_annotations.jsonl
        for subdir in os.listdir(args.path):
            subdir_path = Path(args.path) / subdir
            if subdir_path.is_dir():
                target_file = subdir_path / "annotations" / "scene_annotations.jsonl"
                if target_file.exists():
                    file_path_str = str(target_file)
                    
                    # 检查文件是否已经处理过
                    status = db_manager.get_file_status(file_path_str)
                    if status == 'completed' and not args.force:
                        print(f"\n✓ Skipping (already completed): {target_file}")
                        continue
                    elif status == 'failed' and not args.force:
                        print(f"\n⚠ Skipping (marked as failed): {target_file}")
                        continue
                    
                    print(f"\nProcessing file: {target_file}")
                    
                    # 更新状态为处理中
                    db_manager.add_or_update_file(file_path_str, 'processing')
                    
                    try:
                        # 读取所有条目
                        entries = []
                        fixed_count = 0
                        with open(target_file, "r", encoding="utf-8") as f:
                            for line_num, line in enumerate(f, 1):
                                try:
                                    entry = json.loads(line.strip())
                                    scene_text = entry.get("scene", "")
                                    
                                    # 去除首尾的引号（如果存在）
                                    original_text = scene_text.strip('"').strip("'")
                                    fixed_text = text_fixer.fix_text(original_text)
                                    
                                    # 如果文本被修改了，说明需要修复
                                    if fixed_text != original_text:
                                        print(f"  Line {line_num} (scene_index={entry.get('scene_index', 'N/A')}) needs fixing:")
                                        print(f"    Original: {repr(original_text[:100])}..." if len(original_text) > 100 else f"    Original: {repr(original_text)}")
                                        entry["scene"] = fixed_text
                                        print(f"    Fixed: {fixed_text}")
                                        fixed_count += 1
                                    
                                    entries.append(entry)
                                except json.JSONDecodeError as e:
                                    print(f"  Error parsing line {line_num}: {e}")
                                    continue
                        
                        # 写回文件
                        if fixed_count > 0:
                            print(f"\n  Fixed {fixed_count} entries, writing back to file...")
                            with open(target_file, "w", encoding="utf-8") as f:
                                for entry in entries:
                                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                            print(f"  Done!")
                        else:
                            print(f"  No entries need fixing.")
                        
                        # 标记为完成
                        db_manager.add_or_update_file(file_path_str, 'completed')
                        print(f"✓ Marked as completed in database")
                        
                    except Exception as e:
                        # 标记为失败
                        db_manager.add_or_update_file(file_path_str, 'failed')
                        print(f"✗ Error processing file: {e}")
                        print(f"✗ Marked as failed in database")
                        continue
    
    finally:
        # 关闭数据库连接
        db_manager.close()
        print("\n" + "="*60)
        print("Processing completed. Database connection closed.")