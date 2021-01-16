# Yolov5 Jetson

This is a fork of [yolov5](https://github.com/ultralytics/yolov5) at v3.1.

## Why

1. v4.0 Requires torch >= 1.7.0. As of Jan 2021, torch binaries for `aarch64` are only available for 1.6.0.

2. YoloV5 was only available as a git repo. This limited the useability for purposes outside of the source code.

3. I found using pre-built models straightforward and wanted to add convenience methods for reusing models trained on custom data

