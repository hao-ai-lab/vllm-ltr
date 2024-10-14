# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#This file includes code from the allRank repository, licensed under the Apache License 2.0.

#Repository: https://github.com/allegro/allRank

import torch
from torch.nn import BCELoss

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.model_utils import get_torch_device


def bce(y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE):
    """
    Binary Cross-Entropy loss.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    device = get_torch_device()

    y_pred = y_pred.clone()
    y_true = y_true.clone()
    print("acc: ", ((y_pred > 0.5) == y_true).float().mean())
    mask = y_true == padded_value_indicator
    valid_mask = y_true != padded_value_indicator

    ls = BCELoss(reduction='none')(y_pred, y_true)
    ls[mask] = 0.0

    document_loss = torch.sum(ls, dim=-1)
    sum_valid = torch.sum(valid_mask, dim=-1).type(torch.float32) > torch.tensor(0.0, dtype=torch.float32, device=device)

    loss_output = torch.sum(document_loss) / torch.sum(sum_valid)

    return loss_output
