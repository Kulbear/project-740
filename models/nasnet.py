import torch.nn as nn

from pretrainedmodels import nasnetalarge

from pipeline.models import BaseModel

__all__ = ['VisionNASNetLarge']

pretrained_settings = {
    'nasnetalarge': {
        'model': nasnetalarge,
        'pretrained': 'imagenet+background',
        'num_features': 4032  # see the pretrainedmodels implementation for details
    }
}


class NASNetLargeFeatureExtractor(nn.Module):
    def __init__(self, arch, pretrained=True):
        super(NASNetLargeFeatureExtractor, self).__init__()
        assert arch in pretrained_settings, ''
        self.model = pretrained_settings[arch]['model'](
            pretrained=pretrained_settings[arch]['pretrained'] if pretrained else False)

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.relu(x)
        return x


class VisionNASNetLarge(BaseModel):
    def __init__(self, metadata={}):
        arch = metadata.get('arch', '')
        assert arch in pretrained_settings
        self.last_linear_num_features = pretrained_settings[arch]['num_features']
        self.feature_extractor = NASNetLargeFeatureExtractor
        super().__init__(metadata=metadata)


if __name__ == '__main__':
    from pipeline.models.net_tests import check_logit_net

    model = VisionNASNetLarge(metadata={'arch': 'nasnetalarge'})
    check_logit_net(model)
