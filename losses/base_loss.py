from abc import ABC, abstractmethod


class BaseLoss(ABC):
    """Base loss function class"""
    def __init__(self, metadata={}):
        self.metadata = metadata

    @abstractmethod
    def __call__(self, prediction, target):
        """Implement the computation steps for the loss function."""
        pass
