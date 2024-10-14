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

import json
from collections import defaultdict
from typing import Dict, List, Optional

from attr import attrib, attrs


@attrs
class TransformerConfig:
    N = attrib(type=int)
    d_ff = attrib(type=int)
    h = attrib(type=int)
    positional_encoding = attrib(type=dict)
    dropout = attrib(type=float)


@attrs
class FCConfig:
    sizes = attrib(type=List[int])
    input_norm = attrib(type=bool)
    activation = attrib(type=str)
    dropout = attrib(type=float)


@attrs
class PostModelConfig:
    d_output = attrib(type=int)
    output_activation = attrib(type=str)


@attrs
class ModelConfig:
    fc_model = attrib(type=FCConfig)
    transformer = attrib(type=TransformerConfig)
    post_model = attrib(type=PostModelConfig)
    path = attrib(type=str, default="")
    n_features = attrib(type=int, default=4096)
    pred_layer_idx = attrib(type=int, default=-1)


@attrs
class PositionalEncoding:
    strategy = attrib(type=str)
    max_indices = attrib(type=int)


@attrs
class DataConfig:
    path = attrib(type=str)
    num_workers = attrib(type=int)
    batch_size = attrib(type=int)
    slate_length = attrib(type=int)
    validation_ds_role = attrib(type=str)


@attrs
class TrainingConfig:
    epochs = attrib(type=int)
    gradient_clipping_norm = attrib(type=float)
    batch_size = attrib(type=int, default=1)
    early_stopping_patience = attrib(type=int, default=0)
    


@attrs
class NameArgsConfig:
    name = attrib(type=str)
    args = attrib(type=dict)


@attrs
class Config:
    model = attrib(type=ModelConfig)
    data = attrib(type=DataConfig)
    optimizer = attrib(type=NameArgsConfig)
    training = attrib(type=TrainingConfig)
    loss = attrib(type=NameArgsConfig)
    metrics = attrib(type=Dict[str, List[int]])
    lr_scheduler = attrib(type=NameArgsConfig)
    val_metric = attrib(type=str, default=None)
    expected_metrics = attrib(type=Dict[str, Dict[str, float]], default={})
    detect_anomaly = attrib(type=bool, default=False)
    click_model = attrib(type=Optional[NameArgsConfig], default=None)

    @classmethod
    def from_json(cls, config_path):
        with open(config_path) as config_file:
            config = json.load(config_file)
            return Config.from_dict(config)

    @classmethod
    def from_dict(cls, config):
        config["model"] = ModelConfig(**config["model"])
        if config["model"].transformer:
            config["model"].transformer = TransformerConfig(**config["model"].transformer)
            if config["model"].transformer.positional_encoding:
                config["model"].transformer.positional_encoding = PositionalEncoding(
                    **config["model"].transformer.positional_encoding)
        config["data"] = DataConfig(**config["data"])
        config["optimizer"] = NameArgsConfig(**config["optimizer"])
        config["training"] = TrainingConfig(**config["training"])
        config["metrics"] = cls._parse_metrics(config["metrics"])
        config["lr_scheduler"] = NameArgsConfig(**config["lr_scheduler"])
        config["loss"] = NameArgsConfig(**config["loss"])
        if "click_model" in config.keys():
            config["click_model"] = NameArgsConfig(**config["click_model"])
        return cls(**config)

    @classmethod
    def to_json(cls, config, config_path):
        content = {}
        content["model"] = config.model.__dict__
        print("cont: ", content)
        with open(config_path, "w") as outfile: 
            json.dump(content, outfile)



    @staticmethod
    def _parse_metrics(metrics):
        metrics_dict = defaultdict(list)  # type: Dict[str, list]
        for metric_string in metrics:
            try:
                name, at = metric_string.split("_")
                metrics_dict[name].append(int(at))
            except (ValueError, TypeError):
                raise MetricConfigError(
                    metric_string,
                    "Wrong formatting of metric in config. Expected format: <name>_<at> where name is valid metric name and at is and int")
        return metrics_dict


class MetricConfigError(Exception):
    pass
