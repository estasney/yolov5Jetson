
from typing import Union, Dict

import torch

from torch.backends import cudnn
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import check_img_size, non_max_suppression_torch_ops, scale_coords, xyxy2xywh
from yolov5.utils.torch_utils import intersect_dicts
from yolov5 import PretrainedWeights
from yolov5.models.slim.base import SlimModel
from yolov5.models.yolo import Model


class SlimModelDetector(SlimModel):

    def __init__(self, params: Union[Dict, str] = 'scratch',
                 weights: str = PretrainedWeights.SMALL,
                 device: str = 'cuda:0'):
        super().__init__(params, weights, device)
        self.model = None
        self.half = False
        self.load_model()

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def load_model(self):
        checkpoint = torch.load(self.weights_path, map_location=self.device)
        model = Model(checkpoint['model'].yaml, ch=3, nc=checkpoint['model'].yaml['nc']).to(self.device)
        state_dict = checkpoint['model'].float().state_dict()
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=[])  # intersect
        model.load_state_dict(state_dict, strict=False)
        model.names = checkpoint['model'].names

        model = model.fuse().eval().autoshape()
        if self.device.type != 'cpu':
            model = model.half()
            self.half = True

        self.model = model

    def detect(self, source, img_size=640, conf=None, iou=None):
        conf = self.model.conf if not conf else conf
        iou = self.model.iou if not iou else iou
        img_size = check_img_size(img_size, s=self.model.stride.max())  # check img_size

        # Set Dataloader
        cudnn.benchmark = True

        dataset = LoadImages(source, img_size=img_size)

        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        img = torch.zeros((1, 3, img_size, img_size), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once

        detections = []

        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = self.model(img, augment=False)[0]
            pred = non_max_suppression_torch_ops(pred, conf, iou, classes=None)

            # Process detections

            for i, det in enumerate(pred):  # detections per image

                p, s, im0 = path, '', im0s

                detection_result = {"entities": [], "detections": [], "src": path}

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    # Calling detach is necessary
                    for c in det[:, -1].detach().unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        detection_result['entities'].append((names[int(c)], int(n)))

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        t_xyxy = torch.tensor(xyxy).view(1, 4)
                        xywh = (xyxy2xywh(t_xyxy) / gn).view(-1).tolist()  # normalized xywh
                        detection_result['detections'].append(dict(xyxy=t_xyxy.view(-1).tolist(), xywh=xywh,
                                                                   cls=names[int(cls)],
                                                                   confidence="{:.2%}".format(float(conf))))

                detections.append(detection_result)

        return detections






