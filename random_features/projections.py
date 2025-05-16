from numpy.core.shape_base import block
import torch

def generate_rademacher_samples(shape, device='cpu'):
    """ Draws uniformly from the (complex) Rademacher distribution. """
    support = torch.tensor([1, -1], dtype=torch.float32, device=device)
    samples = torch.index_select(support, 0, torch.randint(len(support), shape, device=device).view(-1))
    return samples.reshape(shape)

class RademacherTransform():
    def __init__(self, d_in, d_features, device='cpu'):
        """
        d_in: Data input dimension
        d_features: Projection dimension
        complex_weights: Whether to use complex-valued projections
        """
        self.d_in = d_in
        self.d_features = d_features
        self.weights = None
        self.device = device

    def resample(self):
        self.weights = generate_rademacher_samples(
            (self.d_in, self.d_features), device=self.device)

    def forward(self, x):
        return torch.matmul(x, self.weights.to(x.device))

