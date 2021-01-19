import math
from typing import Dict, Union, List, Tuple
import numpy as np
import torch

from models.common import Detections
from utils.datasets import letterbox
from yolov5.utils.general import check_img_size, make_divisible, non_max_suppression_torch_ops, scale_coords, xyxy2xywh
from yolov5 import PretrainedWeights
from yolov5.models.slim.detect import SlimModelDetector
from functools import lru_cache

from collections import namedtuple

ImageSize = namedtuple("ImageSize", "original, scaled, gain")


@lru_cache(maxsize=32)
def cached_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor


class SlimModelRunner(SlimModelDetector):

    def __init__(self, params: Union[Dict, str] = 'scratch',
                 weights: str = PretrainedWeights.SMALL,
                 device: str = 'cuda:0', img_size: int = 640):
        super().__init__(params, weights, device)
        self.model = None
        self.half = False
        self._p = None
        self._stride_max = 0
        self.names = []
        self.img_size = img_size
        self.load_model()

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def load_model(self):
        super().load_model()
        self.names = self.model.names
        self.model = self.model.model  # Removing autoshape
        self._p = next(self.model.parameters())
        self._stride_max = int(self.model.stride.max())

    def get_image_size(self, img: np.array) -> ImageSize:
        orig = img.shape[:2]
        gain = (self.img_size / max(orig))
        scaled = [y * gain for y in orig]

        return ImageSize(orig, scaled, gain)

    def preprocess(self, images: List[np.array]):
        sizes = [self.get_image_size(img) for img in images]
        div_sizes = np.array([x.scaled for x in sizes])
        # noinspection PyArgumentList
        div_sizes = [cached_divisible(x, self._stride_max) for x in div_sizes.max(axis=0)]
        # img_sized = [letterbox(img, new_shape=div_sizes)[0] for (i, img) in enumerate(images)]
        # Yolov5 sends list
        img_sized = [letterbox(img, new_shape=div_sizes)[0] for (i, img) in enumerate(images)]
        img_stacked = self.stack_to_torch(img_sized)
        return img_stacked, sizes, div_sizes

    def stack_to_torch(self, images: List[np.array]):
        img_stacked = np.stack(images, 0) if len(images) > 1 else images[0][np.newaxis]
        img_stacked = np.ascontiguousarray(img_stacked.transpose((0, 3, 1, 2)))
        img_stacked = torch.from_numpy(img_stacked).to(self.device).type_as(self._p) / 255
        return img_stacked

    def infer(self, x, conf=None, iou=None):
        if not conf:
            conf = getattr(self.model, 'conf', 0.25)
        if not iou:
            iou = getattr(self.model, 'iou', 0.45)
        with torch.no_grad():
            x = self.model(x)[0]
        x = non_max_suppression_torch_ops(x, conf_thres=conf, iou_thres=iou)
        return x

    def detect(self, source: List[np.array], img_size=640, conf=None, iou=None):
        stacked, sizes, div_sizes = self.preprocess(source)
        result = self.infer(stacked, conf, iou)

        detections = []
        for i, det in enumerate(result):
            scale_coords(div_sizes, result[i][:, :4], sizes[i].original)
            detection_result = {"entities": [], "detections": []}

            gn = torch.tensor(sizes[i].original)[[1, 0, 1, 0]]

            for c in det[:, -1].detach().unique():
                n = (det[:, -1] == c).sum()  # detections per class
                detection_result['entities'].append((self.names[int(c)], int(n)))

            for *xyxy, conf, cls in reversed(det):
                t_xyxy = torch.tensor(xyxy).view(1, 4)
                xywh = (xyxy2xywh(t_xyxy) / gn).view(-1).tolist()  # normalized xywh
                detection_result['detections'].append(dict(xyxy=t_xyxy.view(-1).tolist(), xywh=xywh,
                                                           cls=self.names[int(cls)],
                                                           confidence="{:.2%}".format(float(conf))))

            detections.append(detection_result)
        return detections
