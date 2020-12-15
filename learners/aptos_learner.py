import time
import datetime
import numpy as np

from tqdm import tqdm
from pathlib import Path
from tensorboardX import SummaryWriter

from .base_learner import BaseLearner

import torch
import torch.nn

__all__ = ['APTOSLearner']

IS_REG = False
COEF = [0.57, 1.57, 2.57, 3.57]


def disc(X):
    X_p = np.copy(X)
    for i, pred in enumerate(X_p):
        if pred < COEF[0]:
            X_p[i] = 0
        elif pred >= COEF[0] and pred < COEF[1]:
            X_p[i] = 1
        elif pred >= COEF[1] and pred < COEF[2]:
            X_p[i] = 2
        elif pred >= COEF[2] and pred < COEF[3]:
            X_p[i] = 3
        else:
            X_p[i] = 4
    return X_p


class APTOSLearner(BaseLearner):
    fold_idx = 0

    def __init__(self,
                 model,
                 optimizer,
                 primary_dataloader,
                 valid_dataloader=None,
                 timestamp=datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M'),
                 metadata={}):

        super().__init__(model,
                         optimizer,
                         primary_dataloader,
                         valid_dataloader=valid_dataloader,
                         metadata=metadata)
        # TODO: use config for paths
        APTOSLearner.increment_fold_idx()
        self.writer = SummaryWriter('./experiments/logs/{}/'.format(timestamp))
        self.save_path = Path('./experiments/saved_models/{}/'.format(timestamp))

    @classmethod
    def increment_fold_idx(cls):
        cls.fold_idx += 1

    def validate(self):
        """Implemented the validation step in training."""

        # start validate
        self.model.eval()
        preds, labels = [], []
        for batch_idx, data in enumerate(self.valid_dataloader):
            # calculate and log losses
            losses_report, valid_preds, valid_labels = self.forward_one_batch(
                data)
            self._update_losses(losses_report, train=False)

            preds.append(valid_preds)
            labels.append(valid_labels)

        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)
        if IS_REG:
            preds = disc(preds)
        # calculate and log metrics
        metrics_report = self.evaluate_metrics(preds, labels)
        self._update_metrics(metrics_report, train=False)

        # TODO: lr scheduler step setting
        self.lr_scheduler.step(self.valid_loss_meters['CrossEntropyLoss'].avg)

        # end validate
        self.model.train()

    def fit_one_epoch(self):
        """Train the model for one epoch."""
        preds, labels = [], []
        for batch_idx, data in tqdm(enumerate(self.primary_dataloader)):
            losses_report, train_preds, train_labels = self.forward_one_batch(
                data)
            preds.append(train_preds)
            labels.append(train_labels)

            self._optimize(losses_report)
            self._update_losses(losses_report, train=True)

            self.iter += 1

            # log/check point
            with torch.no_grad():
                if self.iter % self.log_iter == 0:
                    # TODO: track train
                    preds = np.concatenate(preds, axis=0)
                    labels = np.concatenate(labels, axis=0)
                    if IS_REG:
                        preds = disc(preds)

                    metrics_report = self.evaluate_metrics(preds, labels)
                    self._update_metrics(metrics_report, train=True)
                    preds, labels = [], []

                    if self.valid_dataloader:
                        self.validate()

                    self.log_meters()
                    self.save_checkpoint()
                    self.reset_meters()

    def forward_one_batch(self, data, inference=False):
        """Forward pass one batch of the data with the model."""
        inputs = data['img']
        labels = data.get('label', None)
        inputs = inputs.cuda()
        outputs = self.model(inputs)
        losses_report = None
        if not inference:
            labels = labels.cuda()
            losses_report = self.compute_losses(outputs, labels)
        return losses_report, outputs.detach().cpu().numpy(), labels.detach(
        ).cpu().numpy() if labels is not None else labels

    def save_checkpoint(self):
        """Save the training checkpoint to the disk."""
        if not self.save_ckpt:
            return

        lookup = None
        is_best = False
        checkpoint = self.create_checkpoint()

        # save best only or not?
        if self.save_best_only:
            if self.valid_dataloader:
                for item in [self.valid_metric_meters, self.valid_loss_meters]:
                    if self.primary_indicator in item:
                        lookup = item
            else:
                for item in [self.train_metric_meters, self.train_loss_meters]:
                    if self.primary_indicator in item:
                        lookup = item
            if lookup:
                value = lookup[self.primary_indicator].avg
                if self.best_mode == 'min':
                    if value < self.best_indicator:
                        self.best_indicator = value
                        is_best = True
                else:
                    if value > self.best_indicator:
                        self.best_indicator = value
                        is_best = True

        # TODO: better naming convention
        if self.valid_dataloader:
            metric_string = '-'.join([
                f'{metric}-[{self.valid_metric_meters[metric].avg:.5f}]'
                for metric in self.valid_metric_meters
            ])
            loss_string = '-'.join([
                f'{loss}-[{self.valid_loss_meters[loss].avg:.5f}]'
                for loss in self.valid_loss_meters
            ])
        else:
            metric_string = '-'.join([
                f'{metric}-[{self.train_metric_meters[metric].avg:.5f}]'
                for metric in self.train_metric_meters
            ])
            loss_string = '-'.join([
                f'{loss}-[{self.train_loss_meters[loss].avg:.5f}]'
                for loss in self.train_loss_meters
            ])
        # TODO: use config for paths
        # make subdir
        folder = Path(self.save_path, str(self.fold_idx))
        folder.mkdir(parents=True, exist_ok=True)
        if not self.save_best_only or (self.save_best_only and is_best):
            torch.save(checkpoint,
                       f'{folder}/ep-[{self.epoch}]-iter-[{self.iter}]-{loss_string}-{metric_string}.pth')
