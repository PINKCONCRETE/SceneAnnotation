#!/bin/bash

# Configuration
MODEL_ID="Qwen/Qwen2-VL-7B-Instruct"
PORT=8000

echo "Starting vLLM OpenAI-compatible server..."
echo "Model: $MODEL_ID"
echo "Port: $PORT"
echo "Note: This server will run in the foreground. Press Ctrl+C to stop it."

# Start the server
# --trust-remote-code: Required for Qwen2-VL
# --limit-mm-per-prompt image=1: Optimization for single-image prompts
# --gpu-memory-utilization 0.9: Match the setting that worked in your script
# --max-model-len 4096: Prevent OOM on long sequences
# --max-num-seqs 64: Prevent OOM on concurrent requests
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_ID \
    --trust-remote-code \
    --limit-mm-per-prompt '{"image": 1}' \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --max-num-seqs 64 \
    --port $PORT
