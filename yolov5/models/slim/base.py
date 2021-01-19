from typing import Union, Dict

import torch

from yolov5.config import Params
from yolov5 import PretrainedWeights


class SlimModel:
    def __init__(self,  params: Union[Dict, str] = 'scratch',
                 weights: str = PretrainedWeights.SMALL,
                 device: str = 'cuda:0'):
        self.device = torch.device(device)
        self.cuda = self.device.type == 'cuda'
        self.weights_path = weights
        self.fliplr = 0.0
        self.accumulate = 32
        self.lr0 = 0.0
        self.lrf = 0.0
        self.momentum = 0.0
        self.weight_decay = 0.0
        self.warmup_epochs = 0.0
        self.warmup_momentum = 0.0
        self.warmup_bias_lr = 0.0
        self.box = 0.0
        self.cls = 0.0
        self.cls_pw = 0.0
        self.obj = 0.0
        self.obj_pw = 0.0
        self.iou_t = 0.0
        self.anchor_t = 0.0
        self.anchors = None
        self.fl_gamma = 0.0
        self.hsv_h = 0.0
        self.hsv_s = 0.0
        self.hsv_v = 0.0
        self.degrees = 0.0
        self.translate = 0.0
        self.scale = 0.0
        self.shear = 0.0
        self.perspective = 0.0
        self.flipud = 0.0
        self.mosaic = 0.0
        self.mixup = 0.0

        self.load_params(params)

    def load_params(self, params):
        if isinstance(params, str):
            params = getattr(Params, params.upper())
        for k, v in params.items():
            setattr(self, k, v)

    @property
    def params(self):
        param_names = ['lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs', 'warmup_momentum',
                       'warmup_bias_lr', 'box', 'cls', 'cls_pw', 'obj', 'obj_pw', 'iou_t', 'anchor_t',
                       'anchors', 'fl_gamma', 'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale',
                       'shear', 'perspective', 'flipud', 'fliplr', 'mosaic', 'mixup']
        return {k: getattr(self, k, None) for k in param_names}
