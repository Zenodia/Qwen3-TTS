# Use NVIDIA CUDA base image and install PyTorch manually for better compatibility
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface
ENV PATH="/opt/conda/bin:$PATH"

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    wget \
    curl \
    libsndfile1 \
    sox \
    libsox-fmt-all \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Copy the project files
COPY . /workspace/Qwen3-TTS/

# Install build dependencies for flash-attn and upgrade setuptools for PEP 660 support
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel packaging ninja

# Install Flash Attention 2 (recommended for efficiency)
# Using MAX_JOBS=4 to avoid memory issues during compilation
RUN MAX_JOBS=4 pip3 install --no-cache-dir flash-attn --no-build-isolation

# Install qwen-tts package from source
RUN cd /workspace/Qwen3-TTS && \
    pip install --no-cache-dir -e .

# Pre-download the 0.6B CustomVoice model and tokenizer
# This ensures the model is baked into the image
RUN pip install --no-cache-dir huggingface_hub && \
    python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='Qwen/Qwen3-TTS-Tokenizer-12Hz', local_dir='/workspace/models/Qwen3-TTS-Tokenizer-12Hz'); \
    snapshot_download(repo_id='Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice', local_dir='/workspace/models/Qwen3-TTS-12Hz-0.6B-CustomVoice')"

# Create a startup script for the demo
RUN printf '#!/bin/bash\ncd /workspace/Qwen3-TTS\nqwen-tts-demo /workspace/models/Qwen3-TTS-12Hz-0.6B-CustomVoice --ip 0.0.0.0 --port 8000\n' > /workspace/start-demo.sh && \
    chmod +x /workspace/start-demo.sh

# Expose port for Gradio demo
EXPOSE 8000

# Set the default command to run the demo
CMD ["/workspace/start-demo.sh"]

# Alternative: For interactive use, uncomment the line below
# CMD ["/bin/bash"]

