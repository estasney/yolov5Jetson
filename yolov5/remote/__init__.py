from typing import Union, Dict, List

import Pyro5.api
import numpy as np

from yolov5 import PretrainedWeights
from yolov5.models.slim import SlimModelRunner


@Pyro5.api.expose
class RemoteModelDetector(SlimModelRunner):
    def __init__(self, params: Union[Dict, str] = 'scratch',
                 weights: str = PretrainedWeights.SMALL,
                 device: str = 'cuda:0', img_size: int = 640):
        super().__init__(params, weights, device, img_size)

    def detect(self, source: List[np.array], img_size=640, conf=None, iou=None):
        # return super().detect(source, img_size, conf, iou)
        print("Got Called")


if __name__ == '__main__':
    daemon = Pyro5.api.Daemon()
    ns = Pyro5.api.locate_ns()
    uri = daemon.register(RemoteModelDetector)
    ns.register("detect", uri)
    print("Registed {}".format(uri))
    daemon.requestLoop()

