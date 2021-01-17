import pkg_resources


class PretrainedWeights:
    SMALL = pkg_resources.resource_filename('yolov5', 'weights/yolov5s.pt')
