from typing import Dict, Union, List, Tuple
import numpy as np
import torch
from utils.datasets import letterbox
from yolov5.utils.general import check_img_size, make_divisible, non_max_suppression_torch_ops, scale_coords
from yolov5 import PretrainedWeights
from yolov5.models.slim.detect import SlimModelDetector

class SlimModelRunner(SlimModelDetector):

    def __init__(self, params: Union[Dict, str] = 'scratch',
                 weights: str = PretrainedWeights.SMALL,
                 device: str = 'cuda:0'):
        super().__init__(params, weights, device)
        self.model = None
        self.half = False
        self._p = None
        self.names = []
        self.load_model()

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def load_model(self):
        super().load_model()
        self.model.eval()
        self.names = self.model.names
        self._p = next(self.model.parameters())















