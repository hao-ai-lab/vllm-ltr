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

from allrank.utils.ltr_logging import get_logger

logger = get_logger()


def execute_command(command):
    logger.info("will execute {}".format(command))
    result = os.system(command)
    logger.info("exit_code = {}".format(result))
    if result != 0:
        raise RuntimeError("non-zero exit-code: {} from command '{}'".format(result, command))
