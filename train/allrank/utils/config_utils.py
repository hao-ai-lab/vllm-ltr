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

import importlib
from typing import Any

from allrank.config import NameArgsConfig


def instantiate_from_recursive_name_args(name_args: NameArgsConfig):
    def instantiate_if_name_args(o: Any):
        if isinstance(o, NameArgsConfig):
            return instantiate_from_recursive_name_args(o)
        elif isinstance(o, dict) and {"name", "args"} == o.keys():
            return instantiate_from_recursive_name_args(NameArgsConfig(**o))
        else:
            return o

    instantiated_args = dict([(k, instantiate_if_name_args(v)) for k, v in name_args.args.items()])
    return instantiate_class(name_args.name, **instantiated_args)


def instantiate_class(full_name: str, **kwargs):
    module_name, class_name = full_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_(**kwargs)
