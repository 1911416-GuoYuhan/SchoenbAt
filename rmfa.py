import torch
import torch.nn as nn

from rf_util.measures import P_Measure
from rf_util.coeffients import Maclaurin_Coefs
from random_features.maclaurin import Maclaurin

def rfa(q, k, v, eps = 1.0):
    """
    Args:
        q: [batch_size, head, length, proj_dim]
        k: [batch_size, head, length, proj_dim]
        v: [batch_size, head, length, d_tensor]
        
    Return:
        attn: [batch_size, head, length, d_tensor]
    """
    ZTkv = torch.einsum('bhlp,bhld->bhlpd',k,v)
    ZTkv = torch.cumsum(ZTkv,dim=2)
    ZQk = torch.einsum('bhlpd,bhlp->bhld',ZTkv,q)
    ZTk = torch.cumsum(k,dim=2)
    ZQZTk = torch.einsum('bhlp,bhlp->bhl',q,ZTk).clamp_min(eps)
    attn = ZQk/ZQZTk.unsqueeze(-1)
    return attn

class RMFA(nn.Module):
    def __init__(self, head_dim, dropout, config, device='cuda:0', max_val=20):
        super().__init__()
        self.drop_attn = torch.nn.Dropout(p = dropout)
        self.head_dim = head_dim
        self.nb_features = config["nb_features"] if "nb_features" in config else 128
        self.dotf = config["dotf"]
        self.device = device
        self.kernel_coefs = Maclaurin_Coefs(style = self.dotf, n=max_val).coefs
        # self.kernel_coefs = lambda x: Exponential_Measure.coefs(x)
        self.measure = P_Measure(2., False, max_val)
        
        self.Maclaurin_projector = Maclaurin(self.head_dim,self.nb_features,coef_fun=self.kernel_coefs,measure=self.measure,
                                        bias=0.,lengthscale=1,device=self.device)
        
        self.Maclaurin_projector.resample()
        

    def forward(self, Q, K, V, mask):
        # input [batch_size, nb_heads, seq_len, dim_head]
        b,h,l,d = Q.shape

        data_normalizer = (d ** -0.25)

        projection_q = self.Maclaurin_projector.proj(Q*data_normalizer)
        projection_k = self.Maclaurin_projector.proj(K*data_normalizer)

        if mask is not None:
            projection_k = projection_k.masked_fill(mask[:, None, :, None] == 0, 0)

        return rfa(projection_q,projection_k,V)
