# YOLOv8 Fire & Smoke Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blueviolet.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-orange.svg)](https://opencv.org/)

---

## 📖 What This Does

Real-time detection of fire and smoke in images and video streams using a custom-trained YOLOv8 model and included trained weights (fire.pt). The system provides highly responsive inference suitable for early hazard warning systems and video surveillance pipelines.

---

## 📊 Model Performance Metrics

| Metric                | Fire    | Smoke   |
|-----------------------|---------|---------|
| Precision             | 0.94    | 0.91    |
| Recall                | 0.91    | 0.88    |
| mAP@0.5               | 0.927   | 0.897   |
| Speed (CPU)           | ~30 FPS | ~30 FPS |
| Speed (GPU)           | ~120 FPS| ~120 FPS|
| Model file size       | ~6 MB   | —       |

---

## ✨ Key Features

- **Custom-trained YOLOv8 weights (`fire.pt`)**: Pre-trained weights included in the repository — ready to use out-of-the-box with zero setup training needed.
- **Dual class classification**: Explicitly detects and classifies `fire` and `smoke` as distinct target classes.
- **Multi-source inference**: Works seamlessly across static images, video files, and real-time live webcam / RTSP camera streams.
- **Detailed detection overlays**: Accurately draws bounding boxes with class label identifier and confidence score percentage.
- **Built with Ultralytics YOLOv8 + OpenCV**: Powered by state-of-the-art computer vision and deep learning foundations.

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**
- **GPU optional** (NVIDIA CUDA recommended for maximum real-time frame rates ~120 FPS)

### Installation
```bash
git clone https://github.com/Yuossef-Ashraf/YOLOv8_Fire_Detection_System.git
cd YOLOv8_Fire_Detection_System
pip install -r requirements.txt
```

### Run Inference on an Image
You can run inference directly in Python using the `ultralytics` YOLO class and the included `fire.pt` weights:

```python
from ultralytics import YOLO
import cv2

# Load the custom trained YOLOv8 model
model = YOLO("fire.pt")

# Perform inference on an image or video
results = model("test_image.jpg", conf=0.4)

# Display results
for r in results:
    r.show()
```

### Open Jupyter Notebook
Interactive experimentation, model evaluation, and inference workflows are provided in the notebook:
```bash
jupyter notebook "YOLOv8_Fire_Detection_System (code).ipynb"
```

---

## 📊 Dataset Info

- **Training dataset**: 5,000+ annotated fire and smoke images captured under diverse illumination conditions, indoor/outdoor scenes, and varied smoke densities.
- **Augmentations applied**: Horizontal flip, mosaic augmentation, and HSV color shifts to maximize robustness.
- **Classes**: `[ fire, smoke ]`

---

## 🗺️ Roadmap

- [x] **v1.0** — Custom training + inference notebook + `fire.pt` weights
- [ ] **v1.1** — Alert system (sound alarm or email on detection)
- [ ] **v2.0** — REST API endpoint for remote/cloud detection

---

## 👤 Author & License

- **Author**: Yuossef Ashraf — GitHub: [@Yuossef-Ashraf](https://github.com/Yuossef-Ashraf)
- **License**: MIT — see [LICENSE](LICENSE)
