import os

import torch.nn as nn
import torchvision.models as models

from utils import ClfnetConfig


def _build_model_from_pytorch(config: Clfnetconfig) -> nn.Module:
    # geneerate model from torchvision.models with given name
    model_name = config.model_name
    num_class = config.num_class
    pretrained = config.pretrained
    finetune = config.finetune

    model_name = model_name.lower()
    models_in_pytorch = models.__dict__.keys()

    assert model_name in models_in_pytorch

    model = models.__dict__[model_name](pretrained=pretrained)

    if finetune:
        # TODO implemenet robust reconfigure module changing last classifier
        pass

    return model.cuda()


def build_clfnet(config: ClfnetConfig):

    model_name = config.model_name
    models_in_pytorch = models.__dict__.keys()

    if model_name in models_in_pytorch:
        return _build_model_from_pytorch(config)

    else:
        # return your custom designeed network
        pass
