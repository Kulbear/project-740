import torch.nn as nn
import torchvision.models as models

from pipeline.models import BaseModel

from torchvision.models.resnet import (
    BasicBlock,
    Bottleneck
)

__all__ = ['VisionResNet']

pretrained_settings = {
    'resnet18': {
        'model': models.resnet18,
        'num_features': 512 * BasicBlock.expansion,
        'pretrained': 'imagenet'
    },
    'resnet34': {
        'model': models.resnet34,
        'num_features': 512 * BasicBlock.expansion,
        'pretrained': 'imagenet'
    },
    'resnet50': {
        'model': models.resnet50,
        'num_features': 512 * Bottleneck.expansion,
        'pretrained': 'imagenet'
    },
    'resnet101': {
        'model': models.resnet101,
        'num_features': 512 * Bottleneck.expansion,
        'pretrained': 'imagenet'
    },
    'resnet152': {
        'model': models.resnet152,
        'num_features': 512 * Bottleneck.expansion,
        'pretrained': 'imagenet'
    },
}


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, arch, pretrained=None, with_pooling=True):
        super(ResNetFeatureExtractor, self).__init__()
        self.with_pooling = with_pooling
        assert arch in pretrained_settings, 'Only ResNet18/34/50/101/152 are supported at this moment.'
        self.model = pretrained_settings[arch]['model'](
            pretrained=pretrained_settings[arch]['pretrained'] if pretrained else False)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        if self.with_pooling:
            x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        return x


class VisionResNet(BaseModel):
    def __init__(self, metadata={}):
        arch = metadata.get('arch', '')
        assert arch in pretrained_settings
        self.last_linear_num_features = pretrained_settings[arch]['num_features']
        self.feature_extractor = ResNetFeatureExtractor
        super().__init__(metadata=metadata)


if __name__ == '__main__':
    from pipeline.models.net_tests import check_logit_net

    model = VisionResNet(metadata={'arch': 'resnet18'})
    check_logit_net(model)

    model = VisionResNet(metadata={'arch': 'resnet34'})
    check_logit_net(model)

    model = VisionResNet(metadata={'arch': 'resnet50'})
    check_logit_net(model)

    model = VisionResNet(metadata={'arch': 'resnet101'})
    check_logit_net(model)

    model = VisionResNet(metadata={'arch': 'resnet152'})
    check_logit_net(model)
