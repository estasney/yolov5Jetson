from typing import Union, Dict, List

import numpy as np

from yolov5 import PretrainedWeights
from yolov5.models.slim import SlimModelRunner
from yolov5.remote.npsock import NpSocketBase


class RemoteModelDetector(SlimModelRunner):
    def __init__(self, params: Union[Dict, str] = 'scratch',
                 weights: str = PretrainedWeights.SMALL,
                 device: str = 'cuda:0', img_size: int = 640):
        super().__init__(params, weights, device, img_size)

    def detect(self, source: List[np.array], img_size=640, conf=None, iou=None):
        return super().detect(source, img_size, conf, iou)

