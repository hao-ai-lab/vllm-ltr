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

import os
from typing import Any, Dict, Tuple

from tensorboardX import SummaryWriter


class TensorboardSummaryWriter:
    def __init__(self, output_path: str) -> None:
        self.output_path = output_path
        self.writers = {}  # type: Dict[str, Any]

    def ensure_writer_exists(self, name: str) -> None:
        if name not in self.writers.keys():
            writer_path = os.path.join(self.output_path, name)
            self.writers[name] = SummaryWriter(writer_path)

    def save_to_tensorboard(self, results: Dict[Tuple[str, str], float], n_epoch: int) -> None:
        for (role, metric), value in results.items():
            metric_with_role = "_".join([metric, role])
            self.ensure_writer_exists(metric_with_role)
            self.writers[metric_with_role].add_scalar(metric, value, n_epoch)

    def close_all_writers(self) -> None:
        for writer in self.writers.values():
            writer.close()
