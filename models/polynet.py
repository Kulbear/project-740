import torch.nn as nn

from pretrainedmodels import polynet

from pipeline.models import BaseModel

__all__ = ['VisionPolyNet']

pretrained_settings = {
    'polynet': {
        'model': polynet,
        'pretrained': 'imagenet',
        'num_features': 2048  # see the pretrainedmodels implementation for details
    }
}


class PolyNetFeatureExtractor(nn.Module):
    def __init__(self, arch, pretrained=True):
        super(PolyNetFeatureExtractor, self).__init__()
        assert arch in pretrained_settings, ''
        self.model = pretrained_settings[arch]['model'](
            pretrained=pretrained_settings[arch]['pretrained'] if pretrained else False)

    def forward(self, x):
        x = self.model.features(x)
        return x


class VisionPolyNet(BaseModel):
    def __init__(self, metadata={}):
        arch = metadata.get('arch', '')
        assert arch in pretrained_settings
        self.last_linear_num_features = pretrained_settings[arch]['num_features']
        self.feature_extractor = PolyNetFeatureExtractor
        super().__init__(metadata=metadata)


if __name__ == '__main__':
    from pipeline.models.net_tests import check_logit_net

    model = VisionPolyNet(metadata={'arch': 'polynet'})
    check_logit_net(model)
