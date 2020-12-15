from .base_loss import BaseLoss

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'MSELoss'
]


class MSELoss(BaseLoss):
    """The wrapper class for PyTorch's MSELoss.

    Creates a criterion that measures the mean squared error (squared L2 norm)
    between each element in the input x and target y.
    """

    def __init__(self, metadata):
        super().__init__(metadata=metadata)
        reduce = metadata.get('reduce', None)
        reduction = metadata.get('reduction', 'mean')
        size_average = metadata.get('size_average', None)
        self.loss_fn = nn.MSELoss(reduce=reduce,
                                  reduction=reduction,
                                  size_average=size_average)

    def __call__(self, prediction, target):
        return self.loss_fn(prediction, target)
