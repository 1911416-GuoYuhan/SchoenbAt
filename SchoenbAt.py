import torch
import torch.nn as nn
from ppSBN import PreSBN, PostSBN
from rmfa import RMFA

# multi-head self-attention
class SchoenbAt(nn.Module):
    def __init__(self, dim, head_dim, num_head, dropout, rmfa_config):
        super().__init__()

        self.dim = dim # input_dim
        self.head_dim = head_dim
        self.num_head = num_head

        self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_k = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)

        self.attn = RMFA(head_dim, dropout, rmfa_config)

        self.ff = nn.Linear(self.num_head * self.head_dim, self.dim)
        
        self.pre_norm = PreSBN()
        self.post_norm = PostSBN(d_model = self.dim // self.num_head)

    def forward(self, X, mask):

        Q = self.split_heads(self.W_q(X))
        K = self.split_heads(self.W_k(X))
        V = self.split_heads(self.W_v(X))
        
        # pre-SBN
        Q, K = self.pre_norm(Q), self.pre_norm(K)
        with torch.amp.autocast('cuda', enabled = False):
            attn_out = self.attn(Q.float(), K.float(), V.float(), mask.float())
        # post-SBN
        attn_out = self.post_norm(attn_out)

        attn_out = self.combine_heads(attn_out)
        out = self.ff(attn_out)

        return out


    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X