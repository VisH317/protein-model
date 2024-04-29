import torch
from torch import nn, Tensor
import torch.nn.functional as F
import lightning as L


class Transform(nn.Module):
    def __init__(self, d_model: int, d_text: int, d_clip: int, d_inter: int = None):
        super().__init__()

        self.d_model = d_model
        self.d_clip = d_clip
        self.d_inter = d_inter if d_inter is not None else 4 * d_clip

        self.lin = nn.Sequential(
            nn.Linear(d_model, self.d_inter),
            nn.Linear(self.d_inter, d_clip)
        )

        self.lin2 = nn.Sequential(
            nn.Linear(d_text, self.d_inter),
            nn.Linear(self.d_inter, d_clip)
        )

    def forward(self, input_prot: Tensor, input_text: Tensor) -> Tensor:
        prot = self.lin(input_prot)
        text = self.lin2(input_text)

        return F.sigmoid(prot @ text)


class FullModel(L.LightningModule):
    def __init__(self, prot_model, text_model, clip_transform):
        self.prot_model = prot_model
        self.text_model = text_model
        self.clip_transform = clip_transform

    def training_step(self, batch, batch_idx):
        prot, text, target = batch
        prot = self.prot_model(prot)
        text = self.text_model(text)
        clip_out = self.clip_transform(prot, text)
        loss = F.mse_loss(clip_out, target)
        self.log("train loss: ", loss)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
        return optimizer

