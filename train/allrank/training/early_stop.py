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

from allrank.utils.ltr_logging import get_logger

logger = get_logger()


class EarlyStop:
    def __init__(self, patience):
        self.patience = patience
        self.best_value = 0.0
        self.best_epoch = 0

    def step(self, current_value, current_epoch):
        logger.info("Current:{} Best:{}".format(current_value, self.best_value))
        if current_value > self.best_value:
            self.best_value = current_value
            self.best_epoch = current_epoch

    def stop_training(self, current_epoch) -> bool:
        return current_epoch - self.best_epoch > self.patience
