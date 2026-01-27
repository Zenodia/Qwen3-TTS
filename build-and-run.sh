#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Qwen3-TTS Docker Build and Run Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo

# Check if NVIDIA Docker runtime is available
echo -e "${YELLOW}Checking NVIDIA Docker runtime...${NC}"
if ! docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: NVIDIA Docker runtime not found or not working properly.${NC}"
    echo -e "${RED}Please install NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html${NC}"
    exit 1
fi
echo -e "${GREEN}✓ NVIDIA Docker runtime is available${NC}"
echo

# Check NVIDIA driver version
echo -e "${YELLOW}Checking NVIDIA driver version...${NC}"
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
echo -e "${GREEN}✓ NVIDIA Driver version: $DRIVER_VERSION${NC}"
echo -e "${YELLOW}Note: CUDA 12.8 requires driver 470.57+ or 525.85+ or 535.86+ or 545.23+ or 570+${NC}"
echo

# Build the Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
echo -e "${YELLOW}This may take 15-30 minutes on first build...${NC}"
docker build -t qwen3-tts:0.6b-customvoice .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Docker image built successfully!${NC}"
else
    echo -e "${RED}✗ Docker build failed!${NC}"
    exit 1
fi
echo

# Create outputs directory
mkdir -p outputs
echo -e "${GREEN}✓ Created outputs directory${NC}"

# Ask user what to do next
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}What would you like to do?${NC}"
echo -e "${YELLOW}========================================${NC}"
echo "1) Run the Gradio demo (default)"
echo "2) Run in interactive mode"
echo "3) Run with docker-compose"
echo "4) Exit"
echo

read -p "Enter your choice [1-4]: " choice
choice=${choice:-1}

case $choice in
    1)
        echo -e "${GREEN}Starting Gradio demo...${NC}"
        echo -e "${GREEN}Access the demo at: http://localhost:8000${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
        docker run --rm --gpus all -p 8000:8000 -v $(pwd)/outputs:/workspace/outputs qwen3-tts:0.6b-customvoice
        ;;
    2)
        echo -e "${GREEN}Starting interactive mode...${NC}"
        echo -e "${YELLOW}You can run: qwen-tts-demo /workspace/models/Qwen3-TTS-12Hz-0.6B-CustomVoice --ip 0.0.0.0 --port 8000${NC}"
        docker run --rm -it --gpus all -p 8000:8000 -v $(pwd)/outputs:/workspace/outputs qwen3-tts:0.6b-customvoice /bin/bash
        ;;
    3)
        echo -e "${GREEN}Starting with docker-compose...${NC}"
        echo -e "${GREEN}Access the demo at: http://localhost:8000${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
        docker-compose up qwen3-tts-demo
        ;;
    4)
        echo -e "${GREEN}Exiting...${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting...${NC}"
        exit 1
        ;;
esac

