from .base_loss import BaseLoss

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'FBetaLoss',
    'BinaryFocalLoss',
    'BCEWithLogitsLoss',
    'CrossEntropyLoss'
]


class CrossEntropyLoss(BaseLoss):
    """The wrapper class for PyTorch's CrossEntropyLoss.

    Note that this criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.

    The input is expected to contain raw, unnormalized scores for each class.
    This criterion expects a class index in the range [0, C-1]
    as the target for each value of a 1D tensor of size minibatch.
    """

    def __init__(self, metadata):
        super().__init__(metadata=metadata)
        weight = metadata.get('weight', None)
        reduce = metadata.get('reduce', None)
        reduction = metadata.get('reduction', 'mean')
        ignore_index = metadata.get('ignore_index', -100)
        size_average = metadata.get('size_average', None)
        self.loss_fn = nn.CrossEntropyLoss(weight=weight,
                                           reduce=reduce,
                                           reduction=reduction,
                                           ignore_index=ignore_index,
                                           size_average=size_average)

    def __call__(self, prediction, target):
        return self.loss_fn(prediction, target)


class BCEWithLogitsLoss(BaseLoss):
    """The wrapper class for PyTorch's BCEWithLogitsLoss."""

    def __init__(self, metadata):
        super().__init__(metadata=metadata)
        weight = metadata.get('weight', None)
        reduce = metadata.get('reduce', None)
        reduction = metadata.get('reduction', 'mean')
        pos_weight = metadata.get('pos_weight', None)
        size_average = metadata.get('size_average', None)
        self.loss_fn = nn.BCEWithLogitsLoss(weight=weight,
                                            reduce=reduce,
                                            reduction=reduction,
                                            pos_weight=pos_weight,
                                            size_average=size_average)

    def __call__(self, prediction, target):
        return self.loss_fn(prediction, target)


class BinaryFocalLoss(BaseLoss):
    """Binary Focal Loss, for details, see

            https://arxiv.org/pdf/1708.02002.pdf
    """

    def __init__(self, metadata):
        super().__init__(metadata=metadata)
        self.gamma = metadata.get('gamma', 2)

    def __call__(self, prediction, target):
        assert target.size() == prediction.size()

        max_val = (-prediction).clamp(min=0)

        loss = prediction - prediction * target + max_val + ((-max_val).exp() + (-prediction - max_val).exp()).log()
        invprobs = F.logsigmoid(-prediction * (target * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.mean()


class FBetaLoss(BaseLoss):
    """Soft F-beta loss.

    The F score metric is not differentiable.
    The soft F-beta loss function enables the ability to do differentiation
    that helps to toward optimizing the F-beta metric.
    """

    def __init__(self, metadata):
        super().__init__(metadata=metadata)
        self.beta = metadata.get('beta', 2)
        self.epsilon = metadata.get('epsilon', 1e-7)

    def __call__(self, prediction, target):
        batch_size = prediction.size()[0]
        prediction = torch.sigmoid(prediction)
        num_pos = torch.sum(prediction, 1) + self.epsilon
        num_pos_hat = torch.sum(target, 1) + self.epsilon
        tp = torch.sum(target * prediction, 1)

        precision = tp / num_pos
        recall = tp / num_pos_hat

        fs = (1 + self.beta * self.beta) * precision * recall / (
                self.beta * self.beta * precision + recall + self.epsilon)
        loss = fs.sum() / batch_size

        return 1 - loss
