import torch
from torch import nn


class PreSBN(nn.Module):
    def __init__(self, eps=1e-12):
        super(PreSBN, self).__init__()
        self.eps = eps

    def forward(self, x):
        x = x - x.mean(-1, keepdim=True)
        x = x / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + self.eps)
        out = x / x.norm(dim=-1, keepdim=True)

        return out


class PostSBN(nn.Module):
    def __init__(self, d_model):
        super(PostSBN, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        out = self.gamma * x
        # optional
        # out = torch.pow(out,self.beta)
        return out
