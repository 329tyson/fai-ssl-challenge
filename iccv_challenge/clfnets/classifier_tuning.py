import torch.nn as nn

from utils import ClfnetConfig


def _change_alexnet_last_layer(config, model):
    model.classifier[6] = nn.Linear(4096, config.num_class)


def _change_resnet_labst_layer(config, model):
    pass


def change_last_layer(config, model):
    if config.model_name == "alexnet":
        _change_alexnet_last_layer(config, model)

    if "resnet" in config.model_name:
        _change_resnet_labst_layer(config, model)
