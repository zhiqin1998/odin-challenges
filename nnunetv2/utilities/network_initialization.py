import torch
from torch import nn


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


def init_weights_finetune(model, neg_slope=1e-2, start_idx=43, end_idx=46, freeze=False):
    with torch.no_grad():
        for conv in model.decoder.seg_layers:
            conv.weight[start_idx: end_idx] = nn.init.kaiming_normal_(conv.weight[start_idx: end_idx], a=neg_slope)
            if conv.bias is not None:
                conv.bias[start_idx: end_idx] = nn.init.constant_(conv.bias[start_idx: end_idx], 0)
    if freeze:
        for n, p in model.named_parameters():
            if 'decoder.seg_layers' not in n:
                p.requires_grad = False
                print('freezing', n)