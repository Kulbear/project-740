from abc import ABC, abstractmethod
import numpy as np

import pipeline
import pipeline.losses as losses
import pipeline.metrics as metrics
import pipeline.lr_scheduler as schedulers
from pipeline.utils import AverageMeter

from torch.optim.lr_scheduler import _LRScheduler

__all__ = [
    'BaseLearner'
]


class BaseLearner(ABC):
    """ The base class for learners.

    We try to have the common and high-level code base related to training here.

    :param model: the model to train.
    :param optimizer: the optimizer we use to train the model
    :param primary_dataloader: the training dataloader
    :param valid_dataloader: the validation dataloader
    :param metadata: learner's metadata
    """

    def __init__(self,
                 model,
                 optimizer,
                 primary_dataloader,
                 valid_dataloader=None,
                 metadata={}):
        self.model = model
        self.optimizer = optimizer
        self.primary_dataloader = primary_dataloader
        self.valid_dataloader = valid_dataloader

        # TODO: messy
        self.losses = metadata.get('losses', {})
        self.metrics = metadata.get('metrics', {})
        self.log_epoch = metadata.get('log_iter', 1)
        self.log_iter = metadata.get('log_iter', 400)
        self.save_ckpt = metadata.get('save_ckpt', True)
        self.save_best_only = metadata.get('save_best_only', True)
        if self.save_best_only:
            self.primary_indicator = metadata.get('primary_indicator', None)
            self.best_mode = metadata.get('best_mode', None)
            self.best_indicator = float('inf') if self.best_mode == 'min' else -float('inf')
            # TODO: move assertions to parser stage
            assert self.primary_indicator is not None
            assert self.best_mode is not None

        self.start_epoch = metadata.get('start_epoch', 0)
        self.train_epoch = metadata.get('train_epoch', 10)

        if 'lr_scheduler' in metadata:
            lr_scheduler_class = getattr(schedulers, metadata['lr_scheduler']['type'], None)
            if lr_scheduler_class is not None:
                self.lr_scheduler = lr_scheduler_class(optimizer, metadata=metadata['lr_scheduler']['metadata'])
                self.lr_scheduler.use_default_step_function = issubclass(self.lr_scheduler.__class__, _LRScheduler)
                self.lr_scheduler.metric = metadata['lr_scheduler']['metadata'].get('metric', None)
                print('[LOG] Using default step function {} with metric {}'.format(
                    self.lr_scheduler.use_default_step_function, self.lr_scheduler.metric))
            else:
                raise NotImplementedError

        self.epoch = self.start_epoch
        self.iter = 0

    @abstractmethod
    def validate(self):
        """Implement the validation logic here."""
        pass

    @abstractmethod
    def fit_one_epoch(self):
        """Train the model for one epoch."""
        pass

    @abstractmethod
    def forward_one_batch(self, data, inference=False):
        """Forward pass one batch of the data with the model."""
        pass

    def update_components(model=None,
                          optimizer=None,
                          primary_dataloader=None,
                          valid_dataloader=None,
                          metadata={}):
        pass

    def _optimize(self, losses_report):
        """Backpropagate all the accmulated gradients with single or multiple weighted loss functions."""
        losses = [losses_report[loss['type']] * loss['loss_weight'] for loss in self.losses]
        backward_losses = losses[0]
        for i in range(1, len(losses)):
            backward_losses += losses[i]

        backward_losses.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _update_losses(self, losses_report, train=True):
        """Update all the losses to the log meters."""
        for loss in losses_report:
            if train:
                self.train_loss_meters[loss].update(losses_report[loss].detach().cpu().item())
            else:
                self.valid_loss_meters[loss].update(losses_report[loss].detach().cpu().item())

    def _update_metrics(self, metrics_report, train=True):
        """Update all the metrics to the log meters."""
        for metric in metrics_report:
            if train:
                self.train_metric_meters[metric].update(metrics_report[metric])
            else:
                self.valid_metric_meters[metric].update(metrics_report[metric])

    def train(self):
        """The main entry point for training the model with the specified metadata."""
        print('[LOG] Start training at epoch {}, until {}'.format(self.start_epoch, self.train_epoch))
        for epoch in range(self.start_epoch, self.train_epoch):
            self.model.train()
            self.reset_meters()
            self.fit_one_epoch()
            self.epoch += 1

    def reset_meters(self):
        """Reset all loss and metric meters."""
        self.train_loss_meters = {loss['type']: AverageMeter() for loss in self.losses}
        self.train_metric_meters = {metric['type']: AverageMeter() for metric in self.metrics}
        if self.valid_dataloader:
            self.valid_loss_meters = {loss['type']: AverageMeter() for loss in self.losses}
            self.valid_metric_meters = {metric['type']: AverageMeter() for metric in self.metrics}

    def log_meters(self):
        """Log all the readings in the meters to TensorBoard."""
        print('========================================')
        train_meters = [self.train_loss_meters, self.train_metric_meters]
        for meters_group in train_meters:
            for meter_name in meters_group:
                print(f'train/{meter_name}', meters_group[meter_name].avg, self.iter)
                self.writer.add_scalar(f'train/{meter_name}', meters_group[meter_name].avg, self.iter)

        if self.valid_dataloader:
            valid_meters = [self.valid_loss_meters, self.valid_metric_meters]
            for meters_group in valid_meters:
                for meter_name in meters_group:
                    print(f'valid/{meter_name}', meters_group[meter_name].avg, self.iter)
                    self.writer.add_scalar(f'valid/{meter_name}', meters_group[meter_name].avg, self.iter)

    def inference(self):
        """Produce the prediction result for the given dataloader.

        :param dataloader: the dataloader that provides data for inference
        :return: predicted result
        """
        self.model.train(False)
        preds = []
        for batch_idx, data in enumerate(self.primary_dataloader):
            _, pred, _ = self.forward_one_batch(data, inference=True)
            preds.append(pred)
        preds = np.concatenate(preds, axis=0)
        return preds

    def evaluate_metrics(self, predicted, target):
        """Evaluate the predicted result with all the metrics based on the given ground truth labels.

        :param predicted: predicted results in numpy ndarray format
        :param target: ground truth labels used for evaluation
        :return: a dict that contains evaluated metric values
        """
        metrics_report = {}
        for metric in self.metrics:
            metric_class = getattr(pipeline.metrics, metric['type'])
            metric_callable = metric_class(metadata=metric['metadata'])
            metrics_report[metric['type']] = metric_callable(predicted, target)
        return metrics_report

    def compute_losses(self, predicted, target):
        """Compute the predicted result with all the loss functions based on the given ground truth labels.

        :param predicted: predicted results in numpy ndarray format
        :param target: ground truth labels used for calculate the loss
        :return: a dict that contains computed loss values (used for backpropagation)
        """
        losses_report = {}
        for loss in self.losses:
            loss_class = getattr(pipeline.losses, loss['type'])
            loss_callable = loss_class(metadata=loss['metadata'])
            losses_report[loss['type']] = loss_callable(predicted, target)
        return losses_report

    def create_checkpoint(self, model_only=True):
        """Create a checkpoint used for persistent storage of the trained model.

        :return: a dict that contains all the information and state dicts we want to persistently save
        """

        if model_only:
            checkpoint = {
                'model_weights': self.model.state_dict(),
            }

            return checkpoint

        checkpoint = {
            'model_weights': self.model.state_dict(),
            'optimizers': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'save_epoch': self.epoch,
            'train_loss_meters': self.train_loss_meters,
            'train_metric_meters': self.train_metric_meters,
        }

        if self.valid_dataloader:
            checkpoint['valid_loss_meters'] = self.valid_loss_meters
            checkpoint['valid_metric_meters'] = self.valid_metric_meters

        return checkpoint
