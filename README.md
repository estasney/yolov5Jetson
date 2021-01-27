# Yolov5 Jetson

This is a fork of [yolov5](https://github.com/ultralytics/yolov5) at v3.1. Under active development.

## Why

1. Requires Python >= 3.8. However, PyTorch binaries for NVIDIA Jetson are only available for Python 3.6.9.

2. YoloV5 is not installable. This limited the useability for purposes outside of the source code. I.e. deployment with Flask

3. I found using pre-built models straightforward and wanted to add convenience methods for reusing models trained on custom data

4. The performance insights required additional dependencies.   

