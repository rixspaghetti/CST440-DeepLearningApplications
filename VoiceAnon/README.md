# VoiceAnon: Real-Time Voice Anonymization

A deep learning-based system for real-time voice anonymization that preserves natural speech characteristics while protecting speaker identity.

## Project Overview

This project implements a real-time voice anonymization system using deep learning techniques. It aims to transform a speaker's voice in real-time while maintaining natural speech characteristics and content.

### Key Features
- Real-time voice processing (<100ms latency)
- Privacy-preserving voice transformation
- Natural speech preservation
- Support for multiple deep learning architectures
- Easy-to-use interface

## Setup Instructions

1. **Create a Python Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install PyTorch with CUDA Support (if using GPU)**
   ```bash
   # Visit https://pytorch.org/get-started/locally/ for the correct command
   # based on your system configuration
   ```

4. **Verify Installation**
   ```bash
   python -c "import torch; print(torch.__version__)"
   python -c "import torchaudio; print(torchaudio.__version__)"
   ```

## Project Structure

```
VoiceAnon/
├── main.ipynb           # Main Jupyter notebook for development
├── requirements.txt     # Project dependencies
├── need2know.md        # Detailed project documentation
└── README.md           # This file
```

## Usage

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open `main.ipynb` and follow the instructions within the notebook.

## Development

- The project uses Jupyter notebooks for development and experimentation
- All deep learning models are implemented using PyTorch
- Audio processing is handled using torchaudio and librosa
- Real-time processing capabilities are implemented using PyAudio

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

 