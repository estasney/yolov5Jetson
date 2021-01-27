import logging
import math
import os
from typing import Union, Dict

import numpy as np
import torch
import yaml
from torch import optim as optim
from torch.cuda import amp
from torch.optim import lr_scheduler as lr_scheduler
from tqdm.auto import tqdm

import yolov5.test
from yolov5.models.slim.base import SlimModel
from yolov5.models.yolo import Model
from yolov5.utils.datasets import create_dataloader
from yolov5.utils.general import check_img_size, labels_to_class_weights, compute_loss, fitness
from yolov5.utils.torch_utils import intersect_dicts, ModelEMA
from yolov5 import PretrainedWeights

logger = logging.getLogger(__name__)


class OptShim:
    single_cls = False


class SlimModelTrainer(SlimModel):
    NBS = 64
    BATCH_SIZE = 2

    def __init__(self, dataset: str,
                 params: Union[Dict, str] = 'scratch',
                 weights: str = PretrainedWeights.SMALL,
                 device: str = 'cuda:0'):
        """

        Parameters
        ----------
        dataset : str
        params : Union[Dict, str] defaults to 'scratch'
            If string, must be one of `{'scratch', 'finetune'}`. Otherwise must be dict
        weights : str, defaults to PretrainedWeights.SMALL (yolov5s.pt)
            Path to weights file. Default path is pretrained for ultralytics
        device : str, defaults to `cuda:0`


        """
        super().__init__(params, weights, device)

        self.data_path = dataset

        self.model = None
        self.start_epoch = 0
        self.optimizer = None
        self.train_path = ''
        self.val_path = ''
        self.classes = []
        self.n_classes = 0
        self.best_fitness = 0.0

        self.load_params(params)
        self.load_dataset_meta(self.data_path)

    def load_dataset_meta(self, dataset):
        with open(dataset, "r") as fp:
            data_dict = yaml.load(fp, Loader=yaml.FullLoader)

        if os.path.isabs(data_dict['train']):
            self.train_path = data_dict['train']
        else:
            self.train_path = os.path.abspath(os.path.join(os.path.dirname(self.data_path), data_dict['train']))
        if os.path.isabs(data_dict['val']):
            self.val_path = data_dict['val']
        else:
            self.val_path = os.path.abspath(os.path.join(os.path.dirname(self.data_path), data_dict['val']))

        self.classes = data_dict['names']
        self.n_classes = len(self.classes)

    def load_model(self):
        checkpoint = torch.load(self.weights_path, map_location=self.device)
        model = Model(checkpoint['model'].yaml, ch=3, nc=self.n_classes).to(self.device)
        state_dict = checkpoint['model'].float().state_dict()
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=[])  # intersect
        model.load_state_dict(state_dict, strict=False)

        # Optimizer
        self.weight_decay = self.weight_decay * self.BATCH_SIZE * (self.accumulate / self.NBS)
        self.model = model
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.load_optimizer(checkpoint)

    def load_optimizer(self, checkpoint):

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in self.model.named_parameters():
            v.requires_grad = True
            if '.bias' in k:
                pg2.append(v)  # biases
            elif '.weight' in k and '.bn' not in k:
                pg1.append(v)  # apply weight decay
            else:
                pg0.append(v)  # all else

        optimizer = optim.SGD(pg0, lr=self.lr0,
                              momentum=self.momentum, nesterov=True)
        optimizer.add_param_group({'params': pg1, 'weight_decay': self.weight_decay})
        optimizer.add_param_group({'params': pg2})

        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])

        self.optimizer = optimizer

    def load_scheduler(self, epochs: int):
        lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - self.lrf) + self.lrf
        scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)
        return scheduler, lf

    def train(self, log_dir: str, epochs: int = 300,
              batch_size: int = 2, img_size=(640, 640)):
        """

        Parameters
        ----------
        log_dir : str
            Directory to store results to
        epochs : int
        batch_size : int
        img_size

        Returns
        -------
        """

        self.load_model()
        log_dir = os.path.expanduser(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        results_file = os.path.join(log_dir, 'results.txt')

        weights_dir = os.path.join(log_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        last_weight_fp = os.path.join(weights_dir, "last.pt")
        best_weight_fp = os.path.join(weights_dir, "best.pt")

        scheduler, lf = self.load_scheduler(epochs)

        gs = int(max(self.model.stride))
        imgsz, imgsz_test = [check_img_size(x, gs) for x in img_size]  # verify imgsz are gs-multiples

        ema = ModelEMA(self.model)

        dataloader, dataset = create_dataloader(self.train_path, imgsz, batch_size, gs,
                                                hyp=self.params, augment=True, opt=OptShim)
        n_batches = len(dataloader)

        ema.updates = self.start_epoch * n_batches // self.accumulate
        testloader = create_dataloader(self.val_path, imgsz_test, batch_size, gs,
                                       hyp=self.params, augment=False, rect=True, opt=OptShim)[0]

        self.cls = self.cls * (self.n_classes / 80)
        self.model.nc = self.n_classes
        self.model.hyp = self.params
        self.model.gr = 1.0
        self.model.class_weights = labels_to_class_weights(dataset.labels, self.n_classes).to(self.device)
        self.model.names = self.classes

        # Start Training

        nw = max(round(self.warmup_epochs * n_batches), 1000)  # number of warmup iterations

        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        scheduler.last_epoch = self.start_epoch - 1  # do not move
        scaler = amp.GradScaler(enabled=self.cuda)

        logger.info("Start Epoch : {}".format(self.start_epoch))
        logger.info("Running {} Epochs".format(epochs - self.start_epoch))

        s = ''
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'targets', 'img_size'))

        for epoch in range(self.start_epoch, epochs):
            self.model.train()

            mloss = torch.zeros(4, device=self.device)  # Mean losses
            self.optimizer.zero_grad()
            pbar = tqdm(enumerate(dataloader), total=n_batches)
            for i, (imgs, targets, paths, _) in pbar:
                ni = i + n_batches * epoch  # number integrated batches (since train start)
                imgs = imgs.to(self.device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
                # Warmup
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, np.interp(ni, xi, [1, self.NBS / batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi,
                                            [self.warmup_bias_lr if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.warmup_momentum, self.momentum])

                # Forward
                with amp.autocast(enabled=self.cuda):
                    pred = self.model(imgs)  # forward

                    # loss scaled by batch_size
                    loss, loss_items = compute_loss(pred, targets.to(self.device), self.model)

                    # Backward
                scaler.scale(loss).backward()

                # Optimize
                if ni % self.accumulate == 0:
                    scaler.step(self.optimizer)  # optimizer.step
                    scaler.update()
                    self.optimizer.zero_grad()
                    ema.update(self.model)

                # Print
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_allocated() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

            # Scheduler
            scheduler.step()

            # mAP
            ema.update_attr(self.model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride'])
            final_epoch = epoch + 1 == epochs

            if final_epoch:  # Calculate mAP
                results, maps, times = yolov5.test.test(self.data_path,
                                                        batch_size=batch_size,
                                                        imgsz=imgsz_test,
                                                        model=ema.ema,
                                                        single_cls=False,
                                                        dataloader=testloader,
                                                        save_dir=log_dir,
                                                        plots=False)  # plot first and last

            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > self.best_fitness:
                self.best_fitness = fi

            # Save model
            if final_epoch:
                with open(results_file, 'r') as f:  # create checkpoint
                    ckpt = {
                        'epoch':            epoch,
                        'best_fitness':     self.best_fitness,
                        'training_results': f.read(),
                        'model':            ema.ema,
                        'optimizer':        None if final_epoch else self.optimizer.state_dict()
                        }

                # Save last, best and delete
                torch.save(ckpt, last_weight_fp)
                if self.best_fitness == fi:
                    torch.save(ckpt, best_weight_fp)
                del ckpt

        # End Training
        torch.cuda.empty_cache()
        return results
