from albumentations import (
    Compose, ShiftScaleRotate,
    HorizontalFlip, VerticalFlip,
    RandomBrightnessContrast, Flip,
    IAAAdditiveGaussianNoise, OneOf
)

from .base_augmentation import BaseAugmentation


class APTOSAugmentation(BaseAugmentation):
    def __init__(self, metadata={}):
        super().__init__(metadata=metadata)
        prob_apply = self.metadata.get('prob_apply', 1.0)
        # TODO: use config to generate augmentations?
        # TODO: should be a must-have for searching augmentations
        self.augmentation = Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            OneOf([
                RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0, p=1),
                RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.1, p=1),
            ], p=0.3),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.05, rotate_limit=180, p=0.75),
            # IAAAdditiveGaussianNoise(p=0.2),
        ], p=prob_apply)

    def __call__(self, img, label=None):
        data = {
            'image': img,
            'label': label
        }
        augmented = self.augmentation(**data)
        # img = augmented['image']
        return augmented
