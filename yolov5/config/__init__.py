import yaml
import pkg_resources


class LazyLoader:

    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    def __get__(self, *args):
        if self.data:
            return self.data
        with open(self.filepath, "r") as fp:
            self.data = yaml.load(fp, Loader=yaml.FullLoader)
        return self.data


class ModelConfig:
    SMALL = LazyLoader(pkg_resources.resource_filename('yolov5', 'config/yolov5s.yaml'))
    MEDIUM = LazyLoader(pkg_resources.resource_filename('yolov5', 'config/yolov5m.yaml'))
    LARGE = LazyLoader(pkg_resources.resource_filename('yolov5', 'config/yolov5l.yaml'))
    XL = LazyLoader(pkg_resources.resource_filename('yolov5', 'config/yolov5x.yaml'))


class Params:
    SCRATCH = LazyLoader(pkg_resources.resource_filename('yolov5', 'config/scratch.yaml'))
    FINETUNE = LazyLoader(pkg_resources.resource_filename('yolov5', 'config/finetune.yaml'))
