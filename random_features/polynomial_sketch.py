import torch
import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from random_features.projections import RademacherTransform

class PolynomialSketch():
    """
    The basic polynomial sketch (xTy/l^2 + b)^p with lengthscale l, bias b and degree p.
    """

    def __init__(self, d_in, d_features, degree=2, lengthscale='auto', var=1.0, ard=False, trainable_kernel=False,
                    device='cpu', projection_type='rademacher'):
        """
        d_in: Data input dimension
        d_features: Projection dimension
        degree: The degree of the polynomial kernel
        bias: The bias b (eventually added through input modifiction)
        lengthscale: Downscale of the inputs (1 if none)
        var: Scale of the final kernel
        ard: Automatic Relevance Determination = individual lengthscale per input dimension
        trainable_kernel: Learnable bias, lengthscales, kernel variance
        projection_type: rademacher/gaussian/srht/countsketch_sparse/countsketch_dense/countsketch_scatter
        ahle: Whether to use the construction by Ahle et al. (overcomes exponential variances w.r.t. p but is not always better)
        tree: Whether to use Ahle et al. with a tree construction, otherwise sequential
        complex_weights: Whether to use complex-valued weights (almost always lower variances but more expensive)
        complex_real: Whether to use Complex-to-Real (CtR) sketches, the same as complex_weights but with a real transformation in the end
        num_osnap_samples: Only for projection_type='osnap' - Number of times each input coordinate is allocated to a random index (out of d_features)
        device: cpu or cuda
        """
        self.d_in = d_in
        self.d_features = d_features
        self.degree = degree
        self.device = device
        self.projection_type = projection_type

        # we initialize the kernel hyperparameters
        self.log_bias = None

        num_lengthscales = d_in if ard else 1
        self.log_lengthscale = torch.ones(num_lengthscales, device=device).float() * np.log(lengthscale)
        self.log_var = torch.ones(1, device=device).float() * np.log(var)

        if projection_type == 'rademacher':
            node_projection = lambda d_in, d_out: RademacherTransform(d_in, d_out, device=device)

        # the number of leaf nodes is p
        self.sketch_list = [node_projection(self.d_in, self.d_features) for _ in range(degree)]

    def resample(self):
        # seeding is handled globally!
        for node in self.sketch_list:
            node.resample()

    def plain_forward(self, x):
        # non-hierarchical polynomial sketch

        output = None

        for i, ls in enumerate(self.sketch_list):
            current_output = ls.forward(x)
            
            if i == 0:
                output = current_output
            else:
                output = output * current_output

        else:
            output = output / np.sqrt(self.d_features)
        
        return output.to(x.device)

    def forward(self, x):
        # (hierarchical) random feature construction
        # we first apply the lengthscale
        x = x / self.log_lengthscale.exp().to(x.device)

        x = self.plain_forward(x)

        x = x * torch.exp(self.log_var / 2.).to(x.device)

        return x


if __name__ == '__main__':

    def reference_kernel(data, k, c, log_lengthscale='auto'):
        if isinstance(log_lengthscale, str) and log_lengthscale == 'auto':
            # lengthscale = sqrt(d_in)
            log_lengthscale = 0.5 * np.log(data.shape[1])

        data = data / np.exp(log_lengthscale)
        # implement polynomial kernel and compare!
        return (data.mm(data.t()) + c)**k

    torch.manual_seed(0)
