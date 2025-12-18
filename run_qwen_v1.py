# run_qwen_vl.py
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch

# 模型 ID（Hugging Face）
model_id = "Qwen/Qwen2-VL-7B-Instruct"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_id)

print("Loading model (this may take a few minutes)...")
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,  # 或 torch.float16
    attn_implementation="flash_attention_2"  # 若未装 flash-attn，删掉此行
)

# 加载图像（替换为你自己的图片路径）
image_path = "test.jpg"
image = Image.open(image_path).convert("RGB")

# 构造多模态消息
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "详细描述这张图片的内容。"}
        ]
    }
]

# 应用聊天模板并处理输入
text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(
    images=image,
    text=text_prompt,
    return_tensors="pt"
).to(model.device)

# 生成回答
print("Generating response...")
with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,  # 可设为 True 并加 temperature=0.6 增加多样性
    )

# 解码输出
output = processor.decode(generated_ids[0], skip_special_tokens=True)
print("\nModel Output:\n")
print(output)