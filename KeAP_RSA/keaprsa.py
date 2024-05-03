import torch
from torch import nn
from base_models.keap_trainer import KeAPL
from typing import Any
from base_models.keap import CrossAttention, MLP



class RetrivalTransformer(nn.Module):
    def __init__(self, d_hidden: int, d_memory: int, d_in: int, d_attn: int):
        super().__init__()
        


class KeAPRSA(nn.Module):
    def __init__(self, KeAPL_path: str) -> None:
        super().__init__()
