import torch
from torch import nn, Tensor
import torch.nn.functional as F
from protCLIP.utils import RMSNorm


class MLPTransform(nn.Module):
    def __init__(self, d_model: int, d_clip: int, d_inter: int | None = None) -> None:
        super().__init__()
        self.d_inter = d_inter if d_inter is not None else d_model * 4
        self.d_model = d_model
        self.d_clip = d_clip

        self.mlp = nn.Sequential(
            nn.Linear(d_model, self.d_inter),
            nn.SiLU(),
            nn.Linear(self.d_inter, d_clip),
            RMSNorm(d_model)
        )

    def forward(self, x: Tensor) -> Tensor: return self.mlp(x)


class SelfAttentionTransform(nn.Module):
    def __init__(self, d_model: int, d_out: int, d_inter: int | None = None) -> None:
        super().__init__()
        self.d_inter = d_inter if d_inter is not None else d_model * 4
        self.d_model = d_model
        self.d_out = d_out
        self.d_attn = 128
        self.n_heads = 8

        # attention stuff
        self.qkv = nn.Linear(d_model, 3 * self.d_attn * self.n_heads)
        self.O = nn.Linear(self.d_attn, d_model)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, self.d_inter),
            nn.SiLU,
            nn.Linear(self.d_inter, d_model)
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, seqlen, d = x.size()

        qkv = self.qkv(x)
        Q, K, V = qkv.reshape(batch, seqlen, self.n_heads, self.d_attn).split(self.d_attn, dim=-1)
        O = self.O(F.scaled_dot_product_attention(Q, K, V, dropout_p=0.05))
        out = self.norm1(O + x)

        O = self.mlp(out)
        return self.norm2(O + out)


class CrossGatedSelfAttentionTransform(nn.Module):
    def __init__(self, d_model: int, d_other: int, d_inter: int | None = None, use_gate_act: bool = False) -> None:
        super().__init__()
        self.d_inter = d_inter if d_inter is not None else d_model * 4
        self.d_model = d_model
        self.d_other = d_other
        self.d_attn = 128
        self.n_heads = 8
        self.use_gate_act = use_gate_act

        # attention stuff
        self.qkv = nn.Linear(d_model, 3 * self.d_attn * self.n_heads)
        self.O = nn.Linear(self.d_attn * self.n_heads, d_model)

        self.gate = nn.Linear(self.d_other, self.d_attn * self.n_heads)
        self.gate_act = nn.SiLU()

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, self.d_inter),
            nn.SiLU,
            nn.Linear(self.d_inter, d_model)
        )

    def forward(self, x: Tensor, x1: Tensor) -> Tensor:
        batch, seqlen, d = x.size()

        qkv = self.qkv(x)
        Q, K, V = qkv.reshape(batch, seqlen, self.n_heads, self.d_attn).split(self.d_attn, dim=-1)
        O_att = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.05)

        gate = self.gate(x1)
        O_gated = self.O(O_att * gate)

        out = self.norm1(O_gated + x)

        O = self.mlp(out)
        return self.norm2(O + out)
        

class CrossGatedTransformer(nn.Module):
    def __init__(self, n_enc: int, d_model: int, d_other: int, d_out: int, d_inter: int | None = None, use_gate_act: bool = False) -> None:
        super().__init__()

        self.d_inter = d_inter if d_inter is not None else d_model * 4
        self.d_model = d_model
        self.n_enc = n_enc
        self.d_out = d_out
        self.d_other = d_other
        self.d_attn = 128
        self.n_heads = 8
        self.use_gate_act = use_gate_act

        self.encoders = nn.ModuleList([CrossGatedSelfAttentionTransform(d_model, d_other, d_inter, use_gate_act) for _ in range(self.n_enc)])
        self.out_proj = nn.Linear(d_model, d_out)

    def forward(self, x: Tensor, x1: Tensor) -> Tensor:
        for enc in self.encoders:
            x = enc(x, x1)
        
        return self.out_proj(x)
