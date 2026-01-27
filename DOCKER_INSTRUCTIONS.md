# Docker Instructions for Qwen3-TTS

This Dockerfile builds a container with the Qwen3-TTS demo pre-configured with the 0.6B CustomVoice model.

## Base Image

- **Base**: `nvcr.io/nvidia/pytorch:25.03-py3`
- **Reference**: [NVIDIA PyTorch Release 25.03](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-03.html)
- **Includes**: 
  - Ubuntu 24.04 with Python 3.12
  - CUDA 12.8.1
  - PyTorch 2.7.0a0
  - cuDNN 9.8.0

## Prerequisites

1. **NVIDIA Driver**: Release 570 or later (for CUDA 12.8)
   - For data center GPUs: Driver 470.57+ (R470), 525.85+ (R525), 535.86+ (R535), or 545.23+ (R545) also supported
2. **NVIDIA Container Toolkit**: [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
3. **Docker**: Version 20.10 or later

## Build the Docker Image

```bash
# Navigate to the Qwen3-TTS directory
cd /home/zenodia/Agents/Qwen3-TTS

# Build the Docker image (this will take 15-30 minutes)
docker build -t qwen3-tts:0.6b-customvoice .

# Optional: Build with specific build arguments
docker build \
  --build-arg HTTP_PROXY=http://your-proxy:port \
  --build-arg HTTPS_PROXY=http://your-proxy:port \
  -t qwen3-tts:0.6b-customvoice .
```

## Run the Container

### Option 1: Run the Demo (Default)

```bash
# Run the demo on port 8000
docker run --gpus all -p 8000:8000 qwen3-tts:0.6b-customvoice

# Access the demo at: http://localhost:8000
```

### Option 2: Interactive Mode

```bash
# Run in interactive mode
docker run --gpus all -it -p 8000:8000 qwen3-tts:0.6b-customvoice /bin/bash

# Inside the container, you can:
# 1. Run the demo
qwen-tts-demo /workspace/models/Qwen3-TTS-12Hz-0.6B-CustomVoice --ip 0.0.0.0 --port 8000

# 2. Run Python scripts
cd /workspace/Qwen3-TTS
python examples/test_model_12hz_customvoice.py

# 3. Use the Python API
python3 << EOF
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "/workspace/models/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

wavs, sr = model.generate_custom_voice(
    text="Hello, this is a test of Qwen3-TTS running in Docker!",
    language="English",
    speaker="Ryan",
)
sf.write("output.wav", wavs[0], sr)
print("Audio generated successfully!")
EOF
```

### Option 3: Run with Persistent Storage

```bash
# Create a volume for outputs
docker run --gpus all \
  -p 8000:8000 \
  -v $(pwd)/outputs:/workspace/outputs \
  qwen3-tts:0.6b-customvoice
```

## Docker Compose (Optional)

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  qwen3-tts:
    image: qwen3-tts:0.6b-customvoice
    ports:
      - "8000:8000"
    volumes:
      - ./outputs:/workspace/outputs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

Run with:
```bash
docker-compose up
```

## Included Components

The Docker image includes:

1. **Pre-installed Models**:
   - `Qwen3-TTS-Tokenizer-12Hz` (at `/workspace/models/Qwen3-TTS-Tokenizer-12Hz`)
   - `Qwen3-TTS-12Hz-0.6B-CustomVoice` (at `/workspace/models/Qwen3-TTS-12Hz-0.6B-CustomVoice`)

2. **Python Packages**:
   - qwen-tts (installed from source)
   - Flash Attention 2 (for optimal performance)
   - All dependencies from pyproject.toml

3. **System Dependencies**:
   - libsndfile, sox, ffmpeg (for audio processing)

## Supported Speakers (0.6B CustomVoice Model)

- Vivian (Chinese - young female)
- Serena (Chinese - young female)
- Uncle_Fu (Chinese - male)
- Dylan (Chinese Beijing Dialect - male)
- Eric (Chinese Sichuan Dialect - male)
- Ryan (English - male)
- Aiden (English - male)
- Ono_Anna (Japanese - female)
- Sohee (Korean - female)

## Troubleshooting

### GPU Not Detected
```bash
# Verify NVIDIA runtime is installed
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

### Out of Memory
```bash
# Use a smaller batch size or run on a GPU with more VRAM
# The 0.6B model requires approximately 8-12GB GPU memory with flash attention
```

### Permission Issues
```bash
# Run with user mapping
docker run --gpus all --user $(id -u):$(id -g) -p 8000:8000 qwen3-tts:0.6b-customvoice
```

## Building for Different Models

To build with different models, modify the download section in the Dockerfile:

```dockerfile
# For 1.7B CustomVoice model:
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='Qwen/Qwen3-TTS-Tokenizer-12Hz', local_dir='/workspace/models/Qwen3-TTS-Tokenizer-12Hz'); \
    snapshot_download(repo_id='Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', local_dir='/workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice')"

# For Base model:
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='Qwen/Qwen3-TTS-Tokenizer-12Hz', local_dir='/workspace/models/Qwen3-TTS-Tokenizer-12Hz'); \
    snapshot_download(repo_id='Qwen/Qwen3-TTS-12Hz-0.6B-Base', local_dir='/workspace/models/Qwen3-TTS-12Hz-0.6B-Base')"
```

## References

- [NVIDIA PyTorch Container Release Notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-03.html)
- [Qwen3-TTS GitHub Repository](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen3-TTS Paper](https://arxiv.org/abs/2601.15621)

