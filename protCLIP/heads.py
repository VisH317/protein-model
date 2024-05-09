import torch
from torch import nn, Tensor

class LMHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()

        self.dense = nn.Linear(d_model, d_model)
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x: Tensor) -> Tensor:
        y = self.norm(self.act(self.dense(x)))
        return self.out(y)

