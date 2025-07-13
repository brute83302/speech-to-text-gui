# Speech to Text GUI

A modern, high-performance speech-to-text application with GPU acceleration support. Built with Python, faster-whisper, and tkinter for cross-platform compatibility.

![Speech to Text GUI](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20%7C%20AMD%20%7C%20Intel-green.svg)

## ✨ Features

- **🎤 Real-time Recording**: Record audio directly from your microphone
- **📁 File Support**: Process various audio formats (MP3, WAV, FLAC, M4A, OGG, WMA)
- **⚡ GPU Acceleration**: Automatic detection and optimization for NVIDIA, AMD, and Intel GPUs
- **🚀 High Performance**: Up to 50x real-time speed with GPU acceleration
- **🎯 Smart Fallback**: Optimized CPU processing when GPU unavailable
- **🌙 Modern UI**: Dark theme with smooth animations
- **💾 Export Options**: Save transcriptions and copy to clipboard
- **🔧 Multiple Models**: Support for tiny, base, small, medium, large, and turbo models

## 🎮 Hardware Support

### GPU Acceleration
- **NVIDIA GPUs**: Full CUDA acceleration (RTX series, GTX series)
- **AMD GPUs**: ROCm support for discrete graphics cards
- **Intel GPUs**: OpenVINO acceleration for Arc and integrated graphics
- **Steam Deck**: Optimized for AMD APU with OpenVINO backend

### Performance Expectations
| Hardware | Expected Speed | Example |
|----------|---------------|---------|
| NVIDIA RTX 4090 | 30-50x real-time | 1 hour audio → 1-2 minutes |
| AMD RX 7800 XT | 20-40x real-time | 1 hour audio → 2-3 minutes |
| Intel Arc A770 | 15-25x real-time | 1 hour audio → 3-4 minutes |
| Steam Deck APU | 9-15x real-time | 1 hour audio → 4-7 minutes |
| Modern CPU | 3-8x real-time | 1 hour audio → 8-20 minutes |

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Git (for installation)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/speech-to-text-gui.git
   cd speech-to-text-gui
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python speech_to_text_gui.py
   ```

### GPU Setup (Optional)

For maximum performance, install GPU-specific packages:

#### NVIDIA GPUs
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### AMD GPUs (Linux)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

#### Intel GPUs
```bash
pip install openvino-dev
```

## 📖 Usage

### Graphical Interface

1. **Launch the GUI**:
   ```bash
   python speech_to_text_gui.py
   ```

2. **Select Model**: Choose from tiny (fastest) to large (most accurate)
3. **Record Audio**: Click the microphone button to record
4. **Load File**: Use "Browse" to select an audio file
5. **Transcribe**: Click "Start Transcription" to process
6. **Export**: Save or copy the transcription results

### Command Line Interface

```bash
# Use default turbo model
python transcribe.py audio.mp3

# Use specific model
python transcribe.py audio.wav small

# Use tiny model for faster processing
python transcribe.py recording.flac tiny
```

### Model Recommendations

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| **turbo** | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐ | Real-time, quick notes |
| **small** | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | General purpose |
| **medium** | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | High accuracy needs |
| **large** | ⚡⚡ | ⭐⭐⭐⭐⭐ | Professional transcription |

## 🛠️ Technical Details

### Architecture
- **Frontend**: Python tkinter with modern dark theme
- **Backend**: faster-whisper with BatchedInferencePipeline
- **GPU Detection**: Automatic CUDA/ROCm/OpenVINO detection
- **Audio Processing**: sounddevice + soundfile with real-time monitoring

### Optimizations
- **Batch Processing**: 3x speedup with BatchedInferencePipeline
- **Smart Quantization**: int8/float16 based on hardware
- **Memory Management**: Automatic cleanup and garbage collection
- **Threading**: Non-blocking UI with background processing

## 🔧 Configuration

The application automatically detects and optimizes for your hardware. You can see the current configuration in the status bar:

- `"Turbo model ready (NVIDIA GPU, float16)"` - NVIDIA GPU acceleration
- `"Turbo model ready (AMD GPU, float16)"` - AMD GPU acceleration  
- `"Turbo model ready (Auto GPU, int8)"` - Intel/OpenVINO GPU acceleration
- `"Turbo model ready (CPU, int8)"` - Optimized CPU processing

## 🐛 Troubleshooting

### Common Issues

**"No audio recorded"**
- Check microphone permissions
- Test microphone with "Test Microphone" button
- Adjust sensitivity settings

**"Model loading failed"**
- Ensure stable internet connection for first-time model download
- Check available disk space (models are ~500MB-3GB)
- Try a smaller model first (e.g., "tiny")

**Slow transcription**
- Install GPU drivers for your hardware
- Use smaller models for faster processing
- Close other GPU-intensive applications

### Performance Tips

1. **Use turbo model** for best speed/accuracy balance
2. **Install GPU drivers** for hardware acceleration
3. **Close background apps** during transcription
4. **Use SSD storage** for faster model loading

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - High-performance inference library
- [OpenAI Whisper](https://github.com/openai/whisper) - Original speech recognition model
- [OpenVINO](https://github.com/openvinotoolkit/openvino) - Intel's optimization toolkit

## 📊 Changelog

### v1.0.0 (Latest)
- Initial release
- GPU acceleration support (NVIDIA, AMD, Intel)
- Modern dark theme UI
- Real-time recording with audio level monitoring
- Batch processing optimization
- Cross-platform compatibility
- Multiple audio format support

---

**Made with ❤️ for the open source community**