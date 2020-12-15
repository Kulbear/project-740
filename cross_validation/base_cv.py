from abc import ABC, abstractmethod


class BaseCrossValidation(ABC):
    def __init__(self, training_config):
        self.training_config = training_config

    @abstractmethod
    def _split(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass
