import torch.nn as nn

from pipeline.models import BaseModel

from pretrainedmodels import (
    se_resnext50_32x4d,
    se_resnext101_32x4d,
    senet154
)

__all__ = ['VisionSEResNeXt']

pretrained_settings = {
    'se_resnext50_32x4d': {
        'model': se_resnext50_32x4d,
        'num_features': 512 * 4,
        'pretrained': 'imagenet'
    },
    'se_resnext101_32x4d': {
        'model': se_resnext101_32x4d,
        'num_features': 512 * 4,
        'pretrained': 'imagenet'
    },
    'senet154': {
        'model': senet154,
        'num_features': 512 * 4,
        'pretrained': 'imagenet'
    }
}


class SEResNeXtFeatureExtractor(nn.Module):
    def __init__(self, arch, pretrained=None):
        super(SEResNeXtFeatureExtractor, self).__init__()
        assert arch in pretrained_settings, 'Only SEResNext50/101 and SENet154 are supported at this moment.'
        self.model = pretrained_settings[arch]['model'](
            pretrained=pretrained_settings[arch]['pretrained'] if pretrained else False)
        self.layers = []

    def get_pyramid_features(self):
        return self.layers

    def forward(self, x):
        self.layers = []
        x = self.model.layer0(x)
        self.layers.append(x)
        x = self.model.layer1(x)
        self.layers.append(x)
        x = self.model.layer2(x)
        self.layers.append(x)
        x = self.model.layer3(x)
        self.layers.append(x)
        x = self.model.layer4(x)
        self.layers.append(x)

        return x


class VisionSEResNeXt(BaseModel):
    def __init__(self, metadata={}):
        self.last_linear_num_features = 2048
        self.feature_extractor = SEResNeXtFeatureExtractor
        print(metadata)
        super().__init__(metadata=metadata)

    def _get_body_blocks(self):
        pass


if __name__ == '__main__':
    from pipeline.models.net_tests import check_logit_net

    model = VisionSEResNeXt(metadata={'arch': 'se_resnext50_32x4d'})
    check_logit_net(model)

    model = VisionSEResNeXt(metadata={'arch': 'se_resnext101_32x4d'})
    check_logit_net(model)

    model = VisionSEResNeXt(metadata={'arch': 'se154'})
    check_logit_net(model)
import torch.nn as nn

from pipeline.models import BaseModel

from pretrainedmodels import (
    se_resnext50_32x4d,
    se_resnext101_32x4d,
    senet154
)

__all__ = ['VisionSEResNeXt']

pretrained_settings = {
    'se_resnext50_32x4d': {
        'model': se_resnext50_32x4d,
        'num_features': 512 * 4,
        'pretrained': 'imagenet'
    },
    'se_resnext101_32x4d': {
        'model': se_resnext101_32x4d,
        'num_features': 512 * 4,
        'pretrained': 'imagenet'
    },
    'senet154': {
        'model': senet154,
        'num_features': 512 * 4,
        'pretrained': 'imagenet'
    }
}


class SEResNeXtFeatureExtractor(nn.Module):
    def __init__(self, arch, pretrained=None):
        super(SEResNeXtFeatureExtractor, self).__init__()
        assert arch in pretrained_settings, 'Only SEResNext50/101 and SENet154 are supported at this moment.'
        self.model = pretrained_settings[arch]['model'](
            pretrained=pretrained_settings[arch]['pretrained'] if pretrained else False)
        self.layers = []

    def get_pyramid_features(self):
        return self.layers

    def forward(self, x):
        self.layers = []
        x = self.model.layer0(x)
        self.layers.append(x)
        x = self.model.layer1(x)
        self.layers.append(x)
        x = self.model.layer2(x)
        self.layers.append(x)
        x = self.model.layer3(x)
        self.layers.append(x)
        x = self.model.layer4(x)
        self.layers.append(x)

        return x


class VisionSEResNeXt(BaseModel):
    def __init__(self, metadata={}):
        self.last_linear_num_features = 2048
        self.feature_extractor = SEResNeXtFeatureExtractor
        print(metadata)
        super().__init__(metadata=metadata)

    def _get_body_blocks(self):
        pass


if __name__ == '__main__':
    from pipeline.models.net_tests import check_logit_net

    model = VisionSEResNeXt(metadata={'arch': 'se_resnext50_32x4d'})
    check_logit_net(model)

    model = VisionSEResNeXt(metadata={'arch': 'se_resnext101_32x4d'})
    check_logit_net(model)

    model = VisionSEResNeXt(metadata={'arch': 'se154'})
    check_logit_net(model)
