# GPU Acceleration Setup for Steam Deck

## Current Status
Your Steam Deck has an **AMD Custom GPU 0405** but is currently using **CPU-only** processing.

## Why GPU isn't working:
1. **AMD GPU**: Steam Deck uses AMD, not NVIDIA (no CUDA support)
2. **Missing ROCm**: PyTorch doesn't have ROCm support for AMD GPUs
3. **faster-whisper**: Falls back to CPU when GPU acceleration unavailable

## Solutions (in order of recommendation):

### Option 1: ROCm PyTorch (Best for AMD GPU)
Install PyTorch with ROCm support for AMD GPUs:

```bash
# Activate your virtual environment first
source venv/bin/activate

# Install PyTorch with ROCm support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Verify installation
python -c "import torch; print('ROCm available:', torch.version.hip is not None)"
```

### Option 2: OpenVINO Backend (Intel/AMD Compatible)
Use Intel's OpenVINO for broad hardware support:

```bash
# Install OpenVINO toolkit
pip install openvino-dev

# Install faster-whisper with OpenVINO support
pip install faster-whisper[openvino]
```

### Option 3: ONNX Runtime (Cross-platform)
Use ONNX Runtime for better GPU utilization:

```bash
# Install ONNX Runtime
pip install onnxruntime
pip install optimum[onnxruntime]
```

### Option 4: Optimized CPU (Current Setup)
Your current setup is already optimized for CPU:
- Uses `int8` quantization for speed
- BatchedInferencePipeline for 3x speedup
- Optimized threading

## Testing GPU Detection
After installing ROCm PyTorch, test with:

```bash
cd "/home/deck/Documents/vibes only here/speech to text"
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('ROCm available:', hasattr(torch, 'hip') and torch.hip.is_available())
if hasattr(torch, 'hip') and torch.hip.is_available():
    print('HIP devices:', torch.hip.device_count())
"
```

## Expected Performance:
- **CPU (current)**: ~2-4x real-time speed
- **AMD GPU (with ROCm)**: ~5-8x real-time speed  
- **OpenVINO**: ~3-6x real-time speed

## Recommendation:
Try **Option 1 (ROCm)** first as it provides the best AMD GPU acceleration for your Steam Deck's hardware.