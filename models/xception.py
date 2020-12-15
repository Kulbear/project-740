import torch.nn as nn

from pretrainedmodels import xception

from pipeline.models import BaseModel

__all__ = ['VisionXception']

pretrained_settings = {
    'xception': {
        'model': xception,
        'pretrained': 'imagenet',
        'num_features': 2048
    }
}


class XceptionFeatureExtractor(nn.Module):
    def __init__(self, arch, pretrained=True):
        super(XceptionFeatureExtractor, self).__init__()
        assert arch in pretrained_settings, ''
        self.model = pretrained_settings[arch]['model'](
            pretrained=pretrained_settings[arch]['pretrained'] if pretrained else False)

    def forward(self, x):
        x = self.model.features(x)
        # TODO: may need to set a fix version for `pretrainedmodels` to avoid compatibility issues
        # x = nn.ReLU(inplace=True)(x)
        return x


class VisionXception(BaseModel):
    def __init__(self, metadata={}):
        arch = metadata.get('arch', '')
        assert arch in pretrained_settings
        self.last_linear_num_features = pretrained_settings[arch]['num_features']
        self.feature_extractor = XceptionFeatureExtractor
        super().__init__(metadata=metadata)


if __name__ == '__main__':
    from pipeline.models.net_tests import check_logit_net

    model = VisionXception(metadata={'arch': 'xception'})
    check_logit_net(model)
