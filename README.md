# Yolov5 Jetson

This is a fork of [yolov5](https://github.com/ultralytics/yolov5) at v3.1. Under active development and selectively including features from v4.0

## Why

1. YoloV5 Requires Python >= 3.8 and PyTorch >= 1.7.0. While NVIDIA makes PyTorch binaries available, these are for Python 3.6.9
   
    1. I've built a PyTorch Wheel for Python 3.8. Building on Jetson takes 12+ hours. Given it's size (250MB), it's available under Releases:
    
    2. See [PyTorch 1.7.0, Python 3.8 for Jetson Nano (aarch64)](https://github.com/estasney/yolov5Jetson/releases/tag/w1.7.0)
   
   
2. YoloV5 is not installable. This limited the useability for purposes outside of the source code. I.e. deployment with Flask

3. I found using pre-built models straightforward and wanted to add convenience methods for reusing models trained on custom data

4. The performance insights required additional dependencies.   

