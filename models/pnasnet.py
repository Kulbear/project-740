import torch.nn as nn

from pretrainedmodels import pnasnet5large

from pipeline.models import BaseModel

__all__ = ['VisionPNASNetLarge']

pretrained_settings = {
    'pnasnet5large': {
        'model': pnasnet5large,
        'pretrained': 'imagenet+background',
        'num_features': 4320  # see the pretrainedmodels implementation for details
    }
}


class PNASNetLargeFeatureExtractor(nn.Module):
    def __init__(self, arch, pretrained=True):
        super(PNASNetLargeFeatureExtractor, self).__init__()
        # TODO: handle both num_class = 1000 and 1001
        assert arch in pretrained_settings, ''
        self.model = pretrained_settings[arch]['model'](
            pretrained=pretrained_settings[arch]['pretrained'] if pretrained else False)

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.relu(x)
        return x


class VisionPNASNetLarge(BaseModel):
    def __init__(self, metadata={}):
        arch = metadata.get('arch', '')
        assert arch in pretrained_settings
        self.last_linear_num_features = pretrained_settings[arch]['num_features']
        self.feature_extractor = PNASNetLargeFeatureExtractor
        super().__init__(metadata=metadata)


if __name__ == '__main__':
    from pipeline.models.net_tests import check_logit_net

    model = VisionPNASNetLarge(metadata={'arch': 'pnasnet5large'})
    check_logit_net(model)
