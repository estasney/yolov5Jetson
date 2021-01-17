import logging
import json
import os

import click

from yolov5 import PretrainedWeights


@click.group()
def cli():
    """CLI Commands for Yolov5"""
    pass


@cli.command()
@click.option('-ds', '--dataset',
              type=click.Path(exists=True, dir_okay=False, file_okay=True, resolve_path=True), required=True)
@click.option('-l', '--log-dir', type=click.Path(dir_okay=True, file_okay=False, resolve_path=True),
              required=True)
@click.option("-e", "--epochs", type=int, default=30)
@click.option("-w", "--weights", default=PretrainedWeights.SMALL)
@click.option('-d', '--device', default='cuda:0')
@click.option('-p', '--params', type=click.Choice(['scratch', 'finetune']), default='scratch')
def train(dataset, log_dir, epochs, weights, device, params):
    """Train a model"""
    dataset = os.path.expanduser(click.format_filename(dataset))
    log_dir = os.path.expanduser(click.format_filename(log_dir))

    logging.basicConfig(format="%(message)s", level=logging.INFO)

    from yolov5.models.slim.train import SlimModelTrainer as Trainer

    model = Trainer(dataset=dataset, params=params, weights=weights, device=device)
    model.train(log_dir=log_dir, epochs=epochs)


@cli.command()
@click.option("-w", "--weights", default=PretrainedWeights.SMALL)
@click.option('-d', '--device', default='cuda:0')
@click.option('--src', type=click.Path(exists=True), required=True)
@click.option('-o', '--out-dir', type=click.Path(dir_okay=True, file_okay=False, resolve_path=True), required=True)
@click.option('--img-size', type=int, default=640)
@click.option('-c', '--confidence', type=float, default=0.25)
@click.option('--iou', type=float, default=0.45)
def detect(weights, device, src, out_dir, img_size, confidence, iou):
    """Run detections"""

    logging.basicConfig(format="%(message)s", level=logging.INFO)

    out_dir = os.path.expanduser(click.format_filename(out_dir))
    os.makedirs(out_dir, exist_ok=True)

    def name_result_file(folder):
        fc = set((f for f in os.listdir(folder) if f.startswith("result") and ".json" in f))
        counter = 0
        while True:
            fc_candidate = "result{}.json".format(counter)
            if fc_candidate in fc:
                counter += 1
            else:
                return fc_candidate

    result_path = os.path.join(out_dir, name_result_file(out_dir))

    print("Saving Results to {}".format(result_path))

    from yolov5.models.slim.detect import SlimModelDetector as Detector

    model = Detector(weights=weights, device=device)
    results = model.detect(src, img_size, conf=confidence, iou=iou)

    with open(result_path, "w") as fp:
        json.dump(results, fp)











