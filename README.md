# Scene Annotation Pipeline

High-performance, parallelized video scene annotation system using **vLLM** and **Qwen2-VL**.

## 🚀 Key Features

*   **Client-Server Architecture**: Decoupled inference engine (vLLM) from data processing (Client).
*   **Extreme Performance**:
    *   **UDS (Unix Domain Socket)**: Zero-overhead communication between Client and Server (bypasses TCP/IP stack).
    *   **Local Image Streaming**: Zero-copy image transfer via local HTTP server (avoids Base64 CPU bottleneck).
    *   **Full Parallelism**: 
        *   Server: Continuous Batching & PagedAttention.
        *   Client: Multiprocess frame extraction + Async HTTP requests.
*   **Robustness**: Automatic retry on failures, conditional cleanup, and graceful shutdown.

## 🛠️ Architecture

```mermaid
graph TD
    A[Dataset Videos] -->|ProcessPool (10 workers)| B(Frame Extractor)
    B -->|Save JPEG| C[Local Storage /image]
    C -->|Serve via HTTP| D[Local HTTP Server :8081]
    
    E[Client Request Worker] -->|UDS /tmp/vllm.sock| F[vLLM Server]
    F -->|HTTP GET| D
    F -->|Inference Result| E
    
    E -->|Write JSONL| G[scene_annotations.jsonl]
    E -->|Delete Success| C
```

## 📦 Installation

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Model Preparation**:
    Ensure you have access to `Qwen/Qwen2-VL-7B-Instruct`.

## 🚦 Usage

### 1. Start vLLM Server
The server handles the heavy lifting (LLM inference). It listens on a Unix Socket for maximum speed.

```bash
./start_vllm_server.sh
```
*Wait until you see "Uvicorn running on..."*

### 2. Run Annotation Client
The client scans videos, extracts frames, and sends requests to the server in parallel.

```bash
python generate_scene_annotations_client.py
```

### 3. Configuration
Modify `generate_scene_annotations_client.py` -> `Config` class:
*   `NUM_REQUEST_WORKERS`: Concurrency level (default: 64).
*   `SHOW_HTTP_LOGS`: Toggle verbose HTTP logs.
*   `MAX_RETRIES`: Retry attempts for failed requests.

## 📝 Output Format
Results are saved to `scene_annotations_client.jsonl`:
```json
{"episode_idx": 123, "scene_annotation": "A robot arm picks up a red apple."}
```
