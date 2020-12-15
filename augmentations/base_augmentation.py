from abc import ABC, abstractmethod
from albumentations import NoOp


class BaseAugmentation(ABC):
    def __init__(self, metadata={}):
        self.metadata = metadata

    @abstractmethod
    def __call__(self, img, label=None):
        pass


class NoAugmentation(BaseAugmentation):
    def __init__(self, metadata={}):
        super().__init__(metadata=metadata)
        self.augmentation = NoOp()

    def __call__(self, img, label=None):
        data = {
            'image': img,
            'label': label
        }
        data = self.augmentation(**data)
        return data
