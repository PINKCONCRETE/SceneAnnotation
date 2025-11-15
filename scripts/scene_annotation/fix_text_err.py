import json
import ollama
import argparse
import os
from pathlib import Path

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
    args = parser.parse_args()
    
    text_fixer = TextFixer()
    print(f"Processing dataset: {args.path}")
    
    # 只扫描args.path下一层的文件夹，查找annotations/scene_annotations.jsonl
    for subdir in os.listdir(args.path):
        subdir_path = Path(args.path) / subdir
        if subdir_path.is_dir():
            target_file = subdir_path / "annotations" / "scene_annotations.jsonl"
            if target_file.exists():
                print(f"\nFound file: {target_file}")
                
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