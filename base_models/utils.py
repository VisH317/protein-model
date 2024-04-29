import torch
from torch import nn, Tensor

class RMSNorm(nn.Module):
    def __init__(self, affine_transform: bool = True, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.affine_transform = affine_transform

        if affine_transform:
            self.beta = nn.Parameter(torch.zeros(1))
            self.gamma = nn.Parameter(torch.ones(1))
    
    def forward(self, input: Tensor) -> Tensor:
        size = input.size()[-1]
        var = torch.rsqrt(torch.sum(input ** 2, dim=-1) / size) + self.eps
        # print(input.size(), var.size())
        out = input * var.unsqueeze(-1)
        if self.affine_transform: out = out * self.gamma + self.beta
        return out

