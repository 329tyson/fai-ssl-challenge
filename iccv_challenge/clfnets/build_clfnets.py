import os

import torch.nn as nn
import torchvision.models as models

from utils import ClfnetConfig
from clfnets.classifier_tuning import change_last_layer


def _build_model_from_pytorch(config: ClfnetConfig):
    # geneerate model from torchvision.models with given name
    model_name = config.model_name

    model_name = model_name.lower()
    models_in_pytorch = models.__dict__.keys()

    assert model_name in models_in_pytorch

    model = models.__dict__[model_name](pretrained=config.pretrained)

    if config.change_classifier:
        change_last_layer(config, model)

    return model.cuda()


def build_clfnet(config: ClfnetConfig):

    model_name = config.model_name
    models_in_pytorch = models.__dict__.keys()

    if model_name in models_in_pytorch:
        return _build_model_from_pytorch(config)

    else:
        # return your custom designeed network
        pass
