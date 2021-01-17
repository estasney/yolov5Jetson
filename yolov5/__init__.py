import pkg_resources


class PretrainedWeights:
    SMALL = pkg_resources.resource_filename('weights', 'weights/yolov5s.pt')
