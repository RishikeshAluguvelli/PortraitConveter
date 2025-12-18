# Installation Guide

## Prerequisites

- Python 3.8+
- pip package manager
- FFmpeg

## Quick Install (Recommended)

Use the automated setup script:

```bash
./setup_venv.sh
brew install ffmpeg  # macOS
sudo apt install ffmpeg  # Linux
```

## Manual Installation

### 1. Install Python Dependencies

**With virtual environment** (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

**System-wide**:
```bash
pip install -r requirements.txt
```

### 2. Install FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html and add to PATH.

### 3. GPU Support (Optional)

For NVIDIA GPU acceleration:

```bash
# Check CUDA
nvidia-smi

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. Verify Installation

```bash
python test_installation.py
```

## Troubleshooting

**ModuleNotFoundError**: Reinstall dependencies
```bash
pip install -r requirements.txt
```

**CUDA out of memory**: Use smaller resolution or CPU mode

**ffmpeg not found**: Install FFmpeg (see step 2)

**RetinaFace slow**: Normal - it runs on CPU only

See [requirements.txt](requirements.txt) for Python package details.
