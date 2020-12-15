import cv2
import torch
import numpy as np

from pprint import pprint

from torch.utils.data.dataset import Dataset

import pipeline.augmentations as augmentations


def auto_crop(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


class APTOSDataset(Dataset):
    def __init__(self, data, label, metadata={}):
        print('[LOG] Init dataset with metadata ->')
        pprint(metadata)
        self.data = data
        self.label = label
        self.augmentation = getattr(augmentations, metadata.get('augmentations', 'NoAugmentation'))
        self.transform = self.augmentation() if self.augmentation else None
        self.img_size = metadata.get('img_size', 128)
        # this is the sigma value, Ben used 10
        self.augment_contrast = metadata.get('augment_contrast', 0)
        self.crop_method = metadata.get('crop_method', None)
        self.label_smoothing = metadata.get('label_smoothing', 0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = cv2.imread(self.data[idx])
        reshape_to = (self.img_size, self.img_size)

        # crop the black border of the images
        if self.crop_method:
            if self.crop_method == 'auto':
                img = auto_crop(img)
            else:
                # TODO: circle crop
                raise NotImplementedError('Selected crop method is not implemented')

        img = cv2.resize(img, reshape_to).transpose((2, 0, 1))

        # Ben's contrast augmentation
        # https://www.kaggle.com/ratthachat/aptos-updatedv14-preprocessing-ben-s-cropping
        if self.augment_contrast:
            assert isinstance(self.augment_contrast, int) or isinstance(self.augment_contrast, float)
            img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), self.augment_contrast), -4, 128)

        img /= 255.0

        label = self.label[idx]
        if self.transform is not None:
            data = self.transform(img, label=label)
            img, label = data['image'], data['label']

        if self.label is not None:
            # regression label smoothing
            if self.label_smoothing:
                # TODO: label smoothing function
                raise NotImplementedError('Label Smoothing is not implemented!')
            label = np.array(int(self.label[idx]))
            return {
                'img': torch.from_numpy(img).float(),
                'label': torch.from_numpy(label).float()
            }
        else:
            return {
                'img': torch.from_numpy(img).float()
            }
