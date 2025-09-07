# VisionRT

GPU-accelerated computer vision pipeline for real-time inference.

## Same workload, 12x Faster

**Benchmark against the standard OpenCV + PyTorch pipeline.**

![rtcv](images/rtcv.png)

*Figure: VisionRT fits within the 90 FPS (11 ms) frame budget. The standard pipeline overruns, dropping to ~40 FPS.*

## Quick Start

```bash
# Build the module
pip install https://github.com/Abiel-Almonte/vision-rt/releases/download/v0.1.0/vision_rt-0.1.0-cp311-cp311-linux_x86_64.whl

# Test Installation
python -c "import vision_rt; print('VisionRT loaded')"
```


## Requirements

- CUDA 12.8+
- Python 3.11
- V4L2 camera device

>**Note:** Add your user to the video group for camera access:  
>`sudo usermod -aG video $USER`


## Usage

```python
import torch
import torch.nn as nn
import vision_rt as vrt

# Initialize CUDA
torch.cuda.init()

# Open camera
camera = vrt.Camera('/dev/video0')
camera.set_format(0)  # choose a capture format

# Define a PyTorch model
model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(64, 10),
).cuda()

# Wrap with VisionRT's CUDA Graph caching
rt_model = vrt.GraphCachedModule(model)

# Capture a CUDA graph with example input
example_input = torch.randn(1, 3, 224, 224).cuda()
rt_model.capture(example_input)

# Stream frames and run inference in real time
for frame in camera.stream():
    output = rt_model(frame)
    print("Output:", output.shape)
    break

camera.close()
```