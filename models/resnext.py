import torch.nn as nn

from pretrainedmodels import (
    resnext101_32x4d,
    resnext101_64x4d
)

from pipeline.models import BaseModel

__all__ = ['VisionResNeXt']

pretrained_settings = {
    'resnext101_32x4d': {
        'model': resnext101_32x4d,
        'pretrained': 'imagenet',
        'num_features': 2048
    },
    'resnext101_64x4d': {
        'model': resnext101_64x4d,
        'pretrained': 'imagenet',
        'num_features': 2048
    }
}


class ResNeXtFeatureExtractor(nn.Module):
    def __init__(self, arch, pretrained=True):
        super(ResNeXtFeatureExtractor, self).__init__()
        assert arch in pretrained_settings, ''
        self.model = pretrained_settings[arch]['model'](
            pretrained=pretrained_settings[arch]['pretrained'] if pretrained else False)

    def forward(self, x):
        x = self.model.features(x)
        return x


class VisionResNeXt(BaseModel):
    def __init__(self, metadata={}):
        arch = metadata.get('arch', '')
        assert arch in pretrained_settings
        self.last_linear_num_features = pretrained_settings[arch]['num_features']
        self.feature_extractor = ResNeXtFeatureExtractor
        super().__init__(metadata=metadata)


if __name__ == '__main__':
    from pipeline.models.net_tests import check_logit_net

    model = VisionResNeXt(metadata={'arch': 'resnext101_32x4d'})
    check_logit_net(model)

    model = VisionResNeXt(metadata={'arch': 'resnext101_64x4d'})
    check_logit_net(model)
