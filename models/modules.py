import torch
import torch.nn as nn


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, output_size=None):
        super(AdaptiveConcatPool2d, self).__init__()
        self.output_size = output_size or 1
        self.avg_pooling = nn.AdaptiveAvgPool2d(self.output_size)
        self.max_pooling = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x):
        poolings = (self.avg_pooling(x), self.max_pooling(x))
        return torch.cat(poolings, 1)


class AdaptiveAvgMaxPool2d(nn.Module):
    def __init__(self, output_size=None):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.output_size = output_size or 1
        self.avg_pooling = nn.AdaptiveAvgPool2d(self.output_size)
        self.max_pooling = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x):
        avg = self.avg_pooling(x)
        max = self.max_pooling(x)
        return (max + avg) / 2


class Flatten(nn.Module):
    def __init__(self, full=False):
        super().__init__()
        self.full = full

    def forward(self, x):
        return x.view(-1) if self.full else x.view(x.size(0), -1)


class LogitsHead(nn.Module):

    def __init__(self, num_features=512, predefined_out_layer=None, num_classes=10, global_pooling_mode='concat',
                 dropout=0.):
        super(LogitsHead, self).__init__()

        layers = []

        assert global_pooling_mode in ['avg', 'max', 'concat'], 'Invalid pooling type for building the logits head.'
        if global_pooling_mode == 'avg':
            layers.append(nn.AdaptiveAvgPool2d(output_size=1))
        elif global_pooling_mode == 'max':
            layers.append(nn.AdaptiveMaxPool2d(output_size=1))
        elif global_pooling_mode == 'avgmax':
            layers.append(AdaptiveAvgMaxPool2d(output_size=1))
        elif global_pooling_mode == 'concat':
            num_features *= 2
            layers.append(AdaptiveConcatPool2d(output_size=1))

        layers.append(Flatten())

        if dropout > 0.:
            layers.append(torch.nn.Dropout(p=dropout))

        if not isinstance(predefined_out_layer, nn.Module) and predefined_out_layer:
            raise NotImplementedError('Predefined head component must be an object of nn.Module')
        layers.append(nn.Linear(num_features, num_classes) if not predefined_out_layer else predefined_out_layer)

        self.head = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.head(x)
        return logits
