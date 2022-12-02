
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import re

def prune_bert_layers(model, ratio):
    '''
    Prune BERT model by ratio. If ratio is an int, that many weights will be pruned. If it
    is a float, that fraction of weights in each module will be pruned.
    '''
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=ratio)

def check_bert_layer_sparsity(model, layer_num):
    print(
    "Sparsity in layer {} ffn output: {:.2f}%".format(layer_num,
        100. * float(torch.sum(model.encoder.layer[layer_num].output.dense.weight == 0))
        / float(model.encoder.layer[layer_num].output.dense.weight.nelement())
    )
)