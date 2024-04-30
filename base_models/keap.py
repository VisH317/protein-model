import torch
from torch import nn, Tensor
import torch.nn.functional as F
from base_models.utils import RMSNorm


# modules: cross attention block, keap encoder, keap, LMhead

class CrossAttention(nn.Module):
    def __init__(self, d_prot: int, d_text: int, d_attn: int, d_out: int = None, dropout_p: float = 0.05):
        super().__init__()

        self.w_Q = nn.Linear(d_prot, d_attn)
        self.w_KV = nn.Linear(d_text, 2 * d_attn)
        self.w_O = nn.Linear(d_attn, d_out if d_out is not None else d_prot)

        self.norm1 = RMSNorm()
        self.norm2 = RMSNorm()

        self.d_attn = d_attn
        self.dropout_p = dropout_p


    # prot_input: b x l_prot x d_prot
    # text_input: b x l_text x d_text
    # returns: b x l_prot x d_out
    def forward(self, prot_input: Tensor, text_input) -> Tensor:
        prot_norm = self.norm1(prot_input)
        Q = self.w_Q(prot_norm)
        KV = self.w_KV(self.norm2(text_input))
        K, V = torch.tensor_split(KV, (self.d_attn, self.d_attn), dim=-1)

        O = F.scaled_dot_product_attention(Q, K, V, dropout_p=self.dropout_p) # TODO: setup attention mask
        
        return self.w_O(O) + prot_norm
    

class MLP(nn.Module):
    def __init__(self, d_hidden: int, d_inner: int | None = None) -> None:
        super().__init__()

        self.d_inner = d_inner if d_inner is not None else 4 * d_hidden
        self.lin = nn.Sequential(
            nn.Linear(d_hidden, self.d_inner),
            nn.Linear(self.d_inner, d_hidden)
        )

    # input: b x l x d_hidden
    def forward(self, input: Tensor) -> Tensor:
        return self.lin(input) + input
        

class KeAPEncoder(nn.Module):
    def __init__(self, d_hidden: int, d_attribute: int, d_relation: int, d_attn: int, d_ff: int | None = None, dropout_p: float = 0.05) -> None:
        super().__init__()
        
        self.ca1 = CrossAttention(d_hidden, d_attribute, d_attn, d_hidden, dropout_p=dropout_p)
        self.ca2 = CrossAttention(d_hidden, d_relation, d_attn, d_hidden, dropout_p=dropout_p)
        self.mlp = MLP(d_hidden, d_ff)
    

    def forward(self, prot_input: Tensor, relation_input: Tensor, text_input: Tensor) -> Tensor:
        prot = self.ca1(prot_input, text_input)
        prot = self.ca2(prot, relation_input)
        return self.mlp(prot)


class KeAP(nn.Module):
    def __init__(self, n_enc: int, d_prot: int, d_hidden: int, d_attribute: int, d_relation: int, d_attn: int, d_ff: int | None = None, dropout_p: float = 0.05) -> None:
        super().__init__()

        self.embed = nn.Linear(d_prot, d_hidden)
        self.enc = nn.ModuleList([KeAPEncoder(d_hidden, d_attribute, d_relation, d_attn, d_ff, dropout_p) for _ in range(n_enc)])

    def forward(self, input: Tensor) -> Tensor:
        x = self.embed(input)
        for enc in self.enc:
            x = enc(x)
        
        return x
    


