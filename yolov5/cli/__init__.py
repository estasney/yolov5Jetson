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
@click.option('-l', '--log_dir', type=click.Path(dir_okay=True, file_okay=False, resolve_path=True),
              required=True)
@click.option("-e", "--epochs", type=int, default=30)
@click.option("-w", "--weights", default=PretrainedWeights.SMALL)
@click.option('-d', '--device', default='cuda:0')
@click.option('-p', '--params', type=click.Choice(['scratch', 'finetune']), default='scratch')
def train(dataset, log_dir, epochs, weights, device, params):
    """Train a model"""
    dataset = os.path.expanduser(click.format_filename(dataset))
    log_dir = os.path.expanduser(click.format_filename(log_dir))

    from yolov5.models.slim import SlimModelTrainer as Trainer

    model = Trainer(dataset=dataset, params=params, weights=weights, device=device)
    model.train(log_dir=log_dir, epochs=epochs)


