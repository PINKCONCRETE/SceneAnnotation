import ollama
from ollama import ResponseError  # 捕获模型相关错误

def test_ollama_chat(model_name="deepseek-r1:14b"):
    """测试 ollama.chat 方法的调用"""
    try:
        # 简单的测试提示词
        prompt = "请用一句话确认你能正常响应，并用数字1结尾。"
        
        # 调用 ollama.chat 方法（不含 think 参数，先确保基础功能可用）
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # 提取响应内容
        reply = response['message']['content']
        print(f"✅ 模型 {model_name} 调用成功！")
        print(f"响应内容：{reply}")
        
        # 二次测试：添加 think 参数（若模型支持）
        try:
            print("\n--- 测试带 think 参数的调用 ---")
            response_with_think = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": "2加3等于几？请说出思考过程"}],
                think=True  # 启用思考过程输出
            )
            print("带 think 的响应：", response_with_think['message']['content'])
        except ResponseError as e:
            if "unsupported parameter: think" in str(e).lower():
                print("ℹ️ 模型不支持 think 参数，忽略该功能即可正常使用。")
            else:
                print(f"⚠️ 带 think 参数的调用失败：{e}")
        
        return True

    except ResponseError as e:
        print(f"❌ 模型调用错误：{e}")
        if "model not found" in str(e).lower():
            print(f"提示：请先拉取模型 `ollama pull {model_name}`")
        return False
    except Exception as e:
        print(f"❌ 其他错误：{e}")
        print("提示：检查 Ollama 服务是否启动（运行 `ollama serve`）")
        return False

if __name__ == "__main__":
    # 替换为你已安装的模型名（如 "llama3:8b"、"deepseek-r1:14b"）
    test_ollama_chat(model_name="deepseek-r1:14b")
