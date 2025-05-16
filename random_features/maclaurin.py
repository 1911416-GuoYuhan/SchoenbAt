import torch
import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from rf_util.measures import P_Measure, Exponential_Measure

from random_features.polynomial_sketch import PolynomialSketch


class Maclaurin():

    def __init__(self, d_in, d_features, coef_fun,projection = 'rademacher',
                        measure=P_Measure(2, False, 20), bias=0., lengthscale='auto', var=1.0, device='cpu'):

        self.d_in = d_in
        self.d_features = d_features
        self.coef_fun = coef_fun
        self.projection = projection
        self.device = device
        self.measure = measure

        if measure.has_constant:
            # we add degree 0 (constant) of the ML expansion as a random feature
            self.d_features -= 1
        
        num_lengthscales = 1
        self.log_lengthscale = torch.ones(num_lengthscales, device=device).float() * np.log(lengthscale)
        
        self.log_var = torch.ones(1, device=device).float() * np.log(var)


    def resample(self):
        if self.d_features == 0:
            return

        degrees = self.measure.rvs(size=self.d_features)
        # degrees are sorted from highest to lowest
        degrees, proj_dims = np.unique(np.array(degrees), return_counts=True)
        self.coefs = self.coef_fun(degrees)
        if isinstance(self.measure, P_Measure):
            # ensures unbiasedness of maclaurin estimator
            self.coefs /= self.measure._pmf(degrees)

        self.modules = []

        for degree, dim in zip(degrees, proj_dims):
            # we skip the constant
            # the bias and lengthscales will already be included in the data
            proj = self.projection
            mod = PolynomialSketch(self.d_in, int(dim), degree=degree,lengthscale=1.0, var=1.0, projection_type=proj, device=self.device)
            mod.resample()
            self.modules.append(mod)


    def proj(self, x):
        # we first apply the lengthscale
        x = x / self.log_lengthscale.exp().to(x.device)

        if self.d_features > 0:
            # we need to adapt the scaling of the features per degree
            features = torch.cat([
                self.modules[i].forward(x) * np.sqrt(self.coefs[i]) * np.sqrt(self.modules[i].d_features) for i in range(len(self.modules))
            ], dim=-1)
            features = features / np.sqrt(self.d_features)

        # add degree 0 and 1 if desired
        add_features = None

        if self.measure.has_constant:
            add_features_size = torch.cat((torch.tensor(features.shape[:-1]),torch.tensor([1])),0)
            add_features = torch.tensor(self.coef_fun(0),dtype=torch.float).float().sqrt().repeat(tuple(add_features_size))

        if add_features is not None:
            # 放置在GPU上
            if x.is_cuda:
                # add_features = add_features.cuda()
                add_features = add_features.to(x.device)

            if self.measure.h01:
                # we need to append the linear features
                linear = torch.tensor(self.coef_fun(1),dtype=torch.float).float().sqrt() * x
                add_features = torch.cat([add_features, linear], dim=-1)
            
            features = torch.cat([add_features, features], dim=-1)
        
        features = features * torch.exp(self.log_var / 2.).to(features.device)

        return features


if __name__ == "__main__":
    def exponential_kernel(x,y):
        # exp(x*yT)
        return torch.exp(x @ y.transpose(-2,-1))
    
    def self_exponential_kernel(x):
        # exp(x*xT)
        return exponential_kernel(x,x)
    
    eps = 1e-12
    torch.manual_seed(0)
    data = torch.randn(100, 8)
    data = data - data.mean(-1, keepdim=True) 
    data = data / torch.sqrt(data.var(-1, unbiased=False, keepdim=True) + eps) 
    data = data / data.norm(dim=-1, keepdim=True) 

    data2 = torch.randn(100, 8)
    data2 = data2 - data2.mean(-1, keepdim=True) 
    data2 = data2 / torch.sqrt(data2.var(-1, unbiased=False, keepdim=True) + eps)
    data2 = data2 / data2.norm(dim=-1, keepdim=True) 
    
    kernel_coefs = lambda x: Exponential_Measure.coefs(x)
    measure = P_Measure(2., False, 20)

    exp_kernel = exponential_kernel(data,data2)

    dims = [100 * i for i in range(1, 10)]

    print('d =',data.shape[-1])
    for D in dims:
        scores = []
        for seed in np.arange(10):

            torch.manual_seed(seed)

            feature_encoder = Maclaurin(data.shape[-1], D, coef_fun=kernel_coefs, measure=measure,
                                projection = 'rademacher', bias=0., lengthscale=1)

            feature_encoder.resample()

            projection = feature_encoder.forward(data)
            projection2 = feature_encoder.forward(data2)

            
            approx_kernel = projection @ projection2.conj().transpose(-2,-1)

            if approx_kernel.dtype in [torch.complex32, torch.complex64, torch.complex128]:
                approx_kernel = approx_kernel.real

            score = (approx_kernel - exp_kernel).pow(2).sum().sqrt() / exp_kernel.pow(2).sum().sqrt()
            scores.append(score.item())
        
        print('D =',D,'Error =',np.array(scores).mean())

    print('Done!')
    