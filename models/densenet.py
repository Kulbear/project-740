import torch.nn as nn
import torchvision.models as models

from pipeline.models import BaseModel

__all__ = ['VisionDenseNet']

pretrained_settings = {
    'densenet121': {
        'model': models.densenet121,
        'num_features': 1024,
        'pretrained': 'imagenet'

    },
    'densenet161': {
        'model': models.densenet161,
        'num_features': 2208,
        'pretrained': 'imagenet'
    },
    'densenet169': {
        'model': models.densenet169,
        'num_features': 1664,
        'pretrained': 'imagenet'
    },
    'densenet201': {
        'model': models.densenet201,
        'num_features': 1920,
        'pretrained': 'imagenet'
    },
}


class DenseNetFeatureExtractor(nn.Module):
    def __init__(self, arch, pretrained=True, with_pooling=True):
        super(DenseNetFeatureExtractor, self).__init__()
        self.with_pooling = with_pooling
        assert arch in pretrained_settings, 'Only DenseNet101/161/169/201 are supported at this moment.'
        self.model = pretrained_settings[arch]['model'](
            pretrained=pretrained_settings[arch]['pretrained'] if pretrained else False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.model.features(x)
        x = self.relu(x)
        return x


class VisionDenseNet(BaseModel):
    def __init__(self, metadata={}):
        arch = metadata.get('arch', '')
        assert arch in pretrained_settings
        self.last_linear_num_features = pretrained_settings[arch]['num_features']
        self.feature_extractor = DenseNetFeatureExtractor
        super().__init__(metadata=metadata)


if __name__ == '__main__':
    from pipeline.models.net_tests import check_logit_net

    model = VisionDenseNet(metadata={'arch': 'densenet121'})
    check_logit_net(model)

    model = VisionDenseNet(metadata={'arch': 'densenet161'})
    check_logit_net(model)

    model = VisionDenseNet(metadata={'arch': 'densenet169'})
    check_logit_net(model)

    model = VisionDenseNet(metadata={'arch': 'densenet201'})
    check_logit_net(model)
