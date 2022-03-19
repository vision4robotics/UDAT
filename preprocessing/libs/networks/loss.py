import sys
import torch
from torch import nn

def _pointwise_loss(lambd, input, target, size_average=True, reduce=True):
    d = lambd(input, target)
    if not reduce:
        return d
    return torch.mean(d) if size_average else torch.sum(d)

class KLDLoss(nn.Module):
    def __init__(self):
        super(KLDLoss, self).__init__()

    def KLD(self, inp, trg):
        inp = inp/torch.sum(inp)    # input
        trg = trg/torch.sum(trg)    # target
        eps = sys.float_info.epsilon  

        return torch.sum(trg*torch.log(eps+torch.div(trg,(inp+eps))))  

    def forward(self, inp, trg):
        return _pointwise_loss(lambda a, b: self.KLD(a, b), inp, trg)
