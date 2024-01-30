from typing import Tuple

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModel

class Bert(nn.Module):
    def __init__(
        self,
        pretrained: str,
        mode: str = 'cls',
        task: str = 'ranking'
    ) -> None:
        super(Bert, self).__init__()
        self._pretrained = pretrained
        self._mode = mode
        self._task = task

        self._config = AutoConfig.from_pretrained(self._pretrained)
        self._model = AutoModel.from_pretrained(self._pretrained, config=self._config)
        self.attribute_size = 100
        self.ranking_size = self._config.hidden_size - self.attribute_size

        if self._task == 'ranking':
            self.pre_ranking = nn.Linear(self.ranking_size, self.ranking_size)
            self._ranking = nn.Linear(self.ranking_size, 1)
            self.pre_attribute = nn.linear(self.attribute_size, self.attribute_size)
            self._attribute = nn.Linear(self.attribute_size, 1)
            self._adv_attribute = nn.Linear(self.ranking_size, 1)
        elif self._task == 'classification':
            self._ranking = nn.Linear(self._config.hidden_size, 2)
        else:
            raise ValueError('Task must be `ranking` or `classification`.')

    def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor = None, segment_ids: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self._model(input_ids, attention_mask = input_mask, token_type_ids = segment_ids)
        if self._mode == 'cls':
            ranking_logits = output[0][:, 0, :self.ranking_size]
            attribute_logits = output[0][:, 0, self.ranking_size:]
        elif self._mode == 'pooling':
            logits = output[1]
        else:
            raise ValueError('Mode must be `cls` or `pooling`.')
        pre_ranking_representation = self.pre_ranking(ranking_logits)
        ranking_score = self._ranking(pre_ranking_representation).squeeze(-1)
        pre_attribute_representation = self.pre_attribute(attribute_logits)
        attribute_score = self._attribute(pre_attribute_representation).squeeze(-1)
        adv_attribute_score = self._adv_attribute(ranking_logits).squeeze(-1)
        return ranking_score, attribute_score, adv_attribute_score
