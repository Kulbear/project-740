from abc import ABC, abstractmethod


class BaseMetric(ABC):
    """Base metric class"""
    def __init__(self, metadata={}):
        self.metadata = metadata

    @abstractmethod
    def __call__(self, prediction, target):
        """Implement the computation steps for the metric."""
        pass
