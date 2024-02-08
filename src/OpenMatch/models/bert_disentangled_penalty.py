from typing import Tuple
import OpenMatch as om

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModel

class BertDisentangled(nn.Module):
    def __init__(
        self,
        pretrained: str,
        mode: str = 'cls',
        task: str = 'ranking',
        attribute_dim: int = 100,
        # hidden_dim: int = 512
    ) -> None:
        super(BertDisentangled, self).__init__()
        self._pretrained = pretrained
        self._mode = mode
        self._task = task

        # self._config = AutoConfig.from_pretrained(self._pretrained)
        self._model = om.models.Bert(pretrained=self._pretrained)
        state_dict = torch.load("../checkpoints/penalty_disentanglement/exp_4_penalty.bin")
        st = {}
        for k in state_dict:
            if k.startswith('bert'):
                st['_model'+k[len('bert'):]] = state_dict[k]
            elif k.startswith('classifier'):
                st['_dense'+k[len('classifier'):]] = state_dict[k]
            else:
                st[k] = state_dict[k]
        self._model.load_state_dict(st)
        self.attribute_size = attribute_dim
        self.ranking_size = 384 - self.attribute_size

        if self._task == 'ranking':
            self.pre_ranking = nn.Linear(self.ranking_size, self.ranking_size)
            self._ranking = nn.Linear(self.ranking_size, 1)
            self.pre_attribute = nn.Linear(self.attribute_size, self.attribute_size)
            self._attribute = nn.Linear(self.attribute_size, 1)
            self._adv_attribute = nn.Linear(self.ranking_size, 1)
            self.dropout_ranker = nn.Dropout(p=0.1)
            self.dropout_classifier = nn.Dropout(p=0.1)
        elif self._task == 'classification':
            self._ranking = nn.Linear(384, 2)
        else:
            raise ValueError('Task must be `ranking` or `classification`.')

    def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor = None, segment_ids: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        _, output = self._model(input_ids, input_mask, segment_ids)
        if self._mode == 'cls':
            ranking_logits = output[:, :self.ranking_size]
            attribute_logits = output[:, self.ranking_size:]
            # print(output.size())
            # print(ranking_logits.size())
            # print(attribute_logits.size())
        elif self._mode == 'pooling':
            logits = output[1]
        else:
            raise ValueError('Mode must be `cls` or `pooling`.')
        pre_ranking_representation = self.pre_ranking(ranking_logits)
        # pre_ranking_representation = nn.ReLU()(pre_ranking_representation)
        # pre_ranking_representation = self.dropout_ranker(pre_ranking_representation)
        ranking_score = self._ranking(pre_ranking_representation).squeeze(-1)
        pre_attribute_representation = self.pre_attribute(attribute_logits)
        # pre_attribute_representation = nn.ReLU()(pre_attribute_representation)
        # pre_attribute_representation = self.dropout_classifier(pre_attribute_representation)
        attribute_score = self._attribute(pre_attribute_representation).squeeze(-1)
        adv_attribute_score = self._adv_attribute(ranking_logits).squeeze(-1)
        return ranking_score, attribute_score, adv_attribute_score
