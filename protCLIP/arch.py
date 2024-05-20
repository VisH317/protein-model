import torch
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F
from base_models.transformer import BertPooler
import lightning as L
from base_models.transformer import BertModel, text_model_id, prot_model_id
from typing import Mapping, Any
from protCLIP.transforms import MLPTransform, SelfAttentionTransform, CrossGatedSelfAttentionTransform, CrossGatedTransformer


class ProtCLIP(nn.Module):
    def __init__(self, d_prot: int, d_text: int, d_clip: int, d_inter: int = None):
        super().__init__()

        self.d_prot = d_prot
        self.d_clip = d_clip
        self.d_inter = d_inter if d_inter is not None else 4 * d_clip

        self.prot_pooler = BertPooler(self.d_prot)
        self.prot = MLPTransform(self.d_prot, self.d_inter, self.d_clip)

        self.text_pooler = BertPooler(d_text)
        self.text = MLPTransform(d_text, self.d_inter, self.d_clip)

        # self.norm = nn.LayerNorm(d_clip)
        self.act = nn.Sigmoid()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, input_prot: Tensor, input_text: Tensor, prot_ends: list, text_ends: list) -> Tensor:
        prot = self.prot_pooler(input_prot, prot_ends)
        text = self.text_pooler(input_text, text_ends)

        print("protbuhr: ", prot.size())
        print("textbruh: ", text.size())

        prot = self.prot(prot)
        text = self.text(text)

        prot_o = prot.norm(dim=-1, keepdim=True)
        text_o = text.norm(dim=-1, keepdim=True)

        print("prots: ", prot_o.size())
        print("texts: ", text_o.size())

        logit_scale = self.logit_scale.exp()
        out = logit_scale * (prot_o @ text_o.t())

        return out, out.t()

class GatedProtCLIP(nn.Module):
    def __init__(self, d_prot: int, d_text: int, d_clip: int, n_enc: int, d_inter: int | None = None, use_gate_act: bool = False) -> None:
        super().__init__()

        self.prot = CrossGatedTransformer(n_enc, d_prot, d_text, d_clip, d_inter, use_gate_act)
        self.text = CrossGatedTransformer(n_enc, d_text, d_prot, d_clip, d_inter, use_gate_act)
        self.prot_pooler = BertPooler(d_clip, use_norm=True)
        self.text_pooler = BertPooler(d_clip, use_norm=True)

        self.norm = nn.LayerNorm(d_clip)
        self.act = nn.Sigmoid()

    def forward(self, input_prot: Tensor, input_text: Tensor) -> Tensor:
        prot = self.prot_pooler(self.prot(input_prot, input_text))
        text = self.text_pooler(self.text(input_text, input_prot))

        return self.act(prot @ text.t())


class ProtCLIPLit(L.LightningModule):
    def __init__(self, clip_transform, lr: float = 3e-4):
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.prot_model = BertModel(prot_model_id)
        self.text_model = BertModel(text_model_id)
        self.clip_transform = clip_transform.to(device=device)
        self.lr = lr


    def forward(self, input) -> Tensor:
        prot, text = input

        prot = self.prot_model(prot)
        text = self.text_model(text)

        clip_out = self.clip_transform(prot, text)
        return clip_out


    def training_step(self, batch, batch_idx):
        prot, text, target = batch

        prot = self.prot_model(prot)
        text = self.text_model(text)
        clip_out = self.clip_transform(prot, text)
        loss = F.mse_loss(clip_out, target)
        self.log("train loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    
    
    def validation_step(self, batch) -> Tensor | Mapping[str, Any] | None:
        prot, text, target = batch
        prot = self.prot_model(prot)
        text = self.text_model(text)
        clip_out = self.clip_transform(prot, text)

        loss = F.mse_loss(clip_out, target)
        self.log("val loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

