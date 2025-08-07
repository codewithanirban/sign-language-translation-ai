# Real-Time Sign Language to Speech Translation

A comprehensive deep learning project that implements and compares multiple neural network architectures for real-time sign language gesture classification using MediaPipe hand keypoints. The project includes a GUI-based real-time analysis tool for testing and evaluation.

## 🚀 Features

- **Multiple Model Architectures**: GRU+Attention, BiGRU+Attention, TCN, Lightweight Transformer
- **Real-Time Analysis**: GUI application with webcam integration for live testing
- **MediaPipe Integration**: Automatic hand keypoint extraction from video streams
- **Comprehensive Evaluation**: Training, validation, and testing with detailed metrics
- **ONNX Export**: Model optimization for deployment
- **Result Logging**: Save and analyze test results
- **Cross-Platform**: Works on Windows, macOS, and Linux

## 📊 Model Performance

Based on training results with 63,676 samples across 29 ASL classes:

| Model | Test Accuracy | Test F1 Score | Inference Time (ms) |
|-------|---------------|---------------|-------------------|
| BiGRU + Attention | 98.90% | 98.90% | 0.03 |
| GRU + Attention | 98.62% | 98.62% | 0.02 |
| TCN | 98.69% | 98.69% | 0.05 |
| Transformer | 98.48% | 98.48% | 0.04 |

## 🏗️ Project Structure

```
├── models/                     # Model architectures
│   ├── gru_attention.py       # GRU + Attention model
│   ├── bigru_attention.py     # BiGRU + Attention model
│   ├── tcn.py                # Temporal Convolutional Network
│   └── transformer.py        # Lightweight Transformer
├── data_utils.py              # Data loading and preprocessing
├── train_all_models.py        # Training and evaluation script
├── real_time_analysis.py      # GUI application for real-time testing
├── extract_keypoints.py       # MediaPipe keypoint extraction
├── deploy_onnx.py            # ONNX export and inference
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
└── README.md                 # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+ (recommended: Python 3.10)
- Webcam for real-time analysis
- CUDA-compatible GPU (optional, for faster training)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/real-time-sign-language-translation.git
   cd real-time-sign-language-translation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download ASL dataset** (optional - synthetic data is provided)
   ```python
   import kagglehub
   path = kagglehub.dataset_download("grassknoted/asl-alphabet")
   ```

## 📖 Usage

### 1. Training Models

**Option A: Use synthetic data (recommended for testing)**
```bash
python generate_synthetic_data.py
python train_all_models.py
```

**Option B: Use real ASL dataset**
```bash
python extract_keypoints.py
python train_all_models.py
```

### 2. Real-Time Analysis

Launch the GUI application for real-time testing:

```bash
python real_time_analysis.py
```

**GUI Features:**
- Model selection dropdown
- Live webcam feed with hand landmark visualization
- Real-time prediction display
- Recording functionality for result logging
- Save/load test results

### 3. ONNX Export

Export the best model for deployment:

```bash
python deploy_onnx.py --model bigru_attention --model_path outputs/best_bigru_attention.pth --onnx_path outputs/bigru_attention.onnx --data_type npy --data_path asl_keypoints.npy --labels_path asl_labels.npy --num_classes 29
```

## 🎯 Real-Time Analysis Guide

### Getting Started

1. **Launch the application**
   ```bash
   python real_time_analysis.py
   ```

2. **Select a model** from the dropdown menu
   - BiGRU + Attention (recommended for best accuracy)
   - GRU + Attention (fastest inference)
   - TCN (good balance)
   - Transformer (most complex)

3. **Start the camera** by clicking "Start Camera"

4. **Position your hand** in front of the camera
   - Ensure good lighting
   - Keep hand clearly visible
   - Maintain consistent distance

5. **Make ASL gestures** and observe real-time predictions

### Recording and Analysis

1. **Start recording** to log predictions
2. **Perform various gestures** for comprehensive testing
3. **Stop recording** when finished
4. **Save results** to JSON file for analysis
5. **Clear results** to start fresh

### Supported Gestures

The system recognizes 29 ASL classes:
- Letters A-Z
- Space
- Nothing (no gesture)
- Delete

## 🔧 Configuration

### Model Parameters

Edit model configurations in the respective model files:

```python
# Example: models/gru_attention.py
class GRUAttentionModel(nn.Module):
    def __init__(self, input_dim=63, hidden_dim=128, num_layers=2, num_classes=29, dropout=0.2):
        # Adjust parameters as needed
```

### Training Parameters

Modify training settings in `train_all_models.py`:

```python
# Training configuration
epochs = 20
batch_size = 64
learning_rate = 1e-3
```

## 📈 Performance Optimization

### For Real-Time Use

1. **Use CPU-optimized models**: All models are optimized for CPU inference
2. **Adjust camera resolution**: Lower resolution for faster processing
3. **Model selection**: Choose based on accuracy vs. speed requirements
4. **Batch processing**: Process multiple frames for better accuracy

### For Training

1. **GPU acceleration**: Use CUDA for faster training
2. **Data augmentation**: Increase dataset size with synthetic data
3. **Hyperparameter tuning**: Experiment with model architectures
4. **Transfer learning**: Use pre-trained models for better performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for hand landmark detection
- [PyTorch](https://pytorch.org/) for deep learning framework
- [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) for training data
- [OpenCV](https://opencv.org/) for computer vision utilities

## 📞 Support

For questions, issues, or contributions:

- Create an issue on GitHub
- Contact: chakrabortyanirban832@gmail.com
- Documentation: readme & documentation provided

## 🔄 Version History

- **v1.0.0**: Initial release with all model architectures and real-time GUI
- **v0.9.0**: Beta version with basic functionality
- **v0.8.0**: Alpha version with core models

---

**Note**: This project is designed for research and educational purposes. For production use, additional testing and validation is recommended.
