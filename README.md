# 🎙️ Chatterbox TTS with NVIDIA Triton Inference Server

This repository demonstrates how to deploy and run the [ResembleAI Chatterbox TTS model](https://huggingface.co/ResembleAI/chatterbox) using the NVIDIA Triton Inference Server and run real-time inference using the Triton HTTP client.

---

## 🚀 Features

- ✅ Serve Chatterbox TTS model locally with Triton Inference Server
- ✅ Load weights from `.safetensors` (no HuggingFace download)
- ✅ Send HTTP requests to Triton with Triton Python Client
- ✅ Convert and play model output as `.wav` audio
- ✅ Fully Docker-compatible setup

---

## 🧰 Folder Structure

model_repository/
└── src/
    └── chatterbox/
        ├── 1/
            │ └── model.py # Python backend model
        └── config.pbtxt
    └── weights/
    └── Dockerfile
    └── requirements.txt
    └── test/
        └── test_client.py
---

## ⚙️ Prerequisites

- Python 3.12+
- CUDA-enabled GPU
- Triton Inference Server
- Docker (optional but recommended)

### 🔧 Build and Deploy

```bash
docker build -t triton-chatterbox .
```

🧠 Run Triton Server

```bash
docker run --rm --gpus=1 \
  -v $(pwd)/src:/app/models \
  -p8000:8000 -p8001:8001 -p8002:8002 \
  triton-chatterbox
```

### 🔧 Install Dependencies and Testing

```bash
pip install tritonclient[http] soundfile numpy torch
python3 test_client.py
```







