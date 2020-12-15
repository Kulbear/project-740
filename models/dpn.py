import torch.nn as nn

from pipeline.models import BaseModel

from pretrainedmodels import (
    dpn68,
    dpn68b,
    dpn92,
    dpn98,
    dpn107,
    dpn131
)

__all__ = ['VisionDualPathNet']

pretrained_settings = {
    'dpn68': {
        'model': dpn68,
        'pretrained': 'imagenet',
        'num_features': 832
    },
    'dpn68b': {
        'model': dpn68b,
        'pretrained': 'imagenet+5k',
        'num_features': 832
    },
    'dpn92': {
        'model': dpn92,
        'pretrained': 'imagenet+5k',
        'num_features': 2688
    },
    'dpn98': {
        'model': dpn98,
        'pretrained': 'imagenet',
        'num_features': 2688
    },
    'dpn107': {
        'model': dpn107,
        'pretrained': 'imagenet+5k',
        'num_features': 2688
    },
    'dpn131': {
        'model': dpn131,
        'pretrained': 'imagenet',
        'num_features': 2688
    },
}


class DualPathNetFeatureExtractor(nn.Module):
    def __init__(self, arch, pretrained=True):
        super(DualPathNetFeatureExtractor, self).__init__()
        assert arch in pretrained_settings, ''
        self.model = pretrained_settings[arch]['model'](
            pretrained=pretrained_settings[arch]['pretrained'] if pretrained else False)

    def forward(self, x):
        x = self.model.features(x)
        return x


class VisionDualPathNet(BaseModel):
    def __init__(self, metadata={}):
        arch = metadata.get('arch', '')
        assert arch in pretrained_settings
        self.last_linear_num_features = pretrained_settings[arch]['num_features']
        self.feature_extractor = DualPathNetFeatureExtractor
        super().__init__(metadata=metadata)


if __name__ == '__main__':
    from pipeline.models.net_tests import check_logit_net

    model = VisionDualPathNet(metadata={'arch': 'dpn68'})
    check_logit_net(model)

    model = VisionDualPathNet(metadata={'arch': 'dpn68b'})
    check_logit_net(model)

    model = VisionDualPathNet(metadata={'arch': 'dpn92'})
    check_logit_net(model)

    model = VisionDualPathNet(metadata={'arch': 'dpn98'})
    check_logit_net(model)

    model = VisionDualPathNet(metadata={'arch': 'dpn107'})
    check_logit_net(model)

    model = VisionDualPathNet(metadata={'arch': 'dpn131'})
    check_logit_net(model)
