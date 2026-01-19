#!/bin/bash
# export HTTP_PROXY="http://127.0.0.1:7897"
# export HTTPS_PROXY="http://127.0.0.1:7897"

echo "Starting vLLM OpenAI-compatible server..."

# 配置 - 使用完全开放的模型
MODEL_ID="OpenGVLab/InternVL2-2B"
UDS_PATH="/tmp/vllm-server.sock"

echo "Model: $MODEL_ID"
echo "UDS Socket: $UDS_PATH"
echo "Note: This server will run in the foreground. Press Ctrl+C to stop it."

# 清理旧socket
rm -f $UDS_PATH

# 启动服务器（注意：反斜杠后不能有空格或注释）
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_ID \
    --trust-remote-code \
    --limit-mm-per-prompt '{"image": 1}' \
    --gpu-memory-utilization 0.5 \
    --max-model-len 4096 \
    --max-num-seqs 32 \
    --uds $UDS_PATH