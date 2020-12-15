from abc import abstractmethod

import torch.nn as nn

from pipeline.models.modules import LogitsHead


class BaseModel(nn.Module):
    def __init__(self, metadata={}):
        super(BaseModel, self).__init__()
        self.arch = metadata.get('arch', '')
        self.pretrained = metadata.get('pretrained', 'imagenet')
        self.global_pooling_mode = metadata.get('global_pooling_mode', 'avg')
        self.num_classes = metadata.get('num_classes', 10)
        self.dropout = metadata.get('dropout', 0)
        self.mode = metadata.get('mode', 'logits')
        self.with_pooling = metadata.get('with_pooling', True)
        self.global_pooling = metadata.get('global_pooling', None)
        self.predefined_out_layer = metadata.get('predefined_out_layer', None)
        print('[LOG] Model Metadata: {}'.format(metadata))
        self.body = self._create_body()
        self.head = self._create_head()

    @abstractmethod
    def _get_body_blocks(self):
        pass

    def _create_body(self):
        body = self.feature_extractor(
            self.arch,
            pretrained=self.pretrained,
        )
        return body

    def _create_head(self):
        # TODO: support other vision tasks
        if self.mode == 'logits':
            head = LogitsHead(
                num_features=self.last_linear_num_features,
                num_classes=self.num_classes,
                global_pooling_mode=self.global_pooling_mode,
                dropout=self.dropout,
                predefined_out_layer=self.predefined_out_layer
            )
        else:
            raise NotImplementedError('Currently only the MLP head is supported.')

        return head

    def forward(self, x):
        x = self.body(x)
        x = self.head(x)
        return x
