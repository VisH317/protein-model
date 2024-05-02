import torch
from typing import Any
from torch import nn, Tensor
from base_models.keap import KeAP
from base_models.transformer import BertModel, text_model_id, prot_model_id
from base_models.utils import LMHead
import lightning as L
from lightning import Trainer
from data.proteinkg import ProteinKG25
from typing import Mapping, Tuple, Dict, Any


class KeAPL(L.LightningModule):
    def __init__(self, keap: KeAP, lmhead: LMHead | None = None, lr: float = 3e-4):
        super().__init__()

        self.prot = BertModel(prot_model_id)
        self.text = BertModel(text_model_id)

        self.keap = keap
        self.lmhead = lmhead
        self.criterion = nn.CrossEntropyLoss()

        self.lr = lr

        self.train_losses = []
        self.val_losses = []
    
    def forward(self, data: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        prot, rel, att = data
        prot = self.prot(prot)
        rel = self.text(rel)
        att = self.text(att)
    
        y = self.keap(prot, rel, att)
        return y        

    # prot: b x l_prot x d_prot
    # att: b x l_att x d_att
    # rel: b x l_rel x d_rel
    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor | Mapping[str, Any] | None:
        prot, rel, att, target, indices = batch
        with torch.no_grad():
            prot = self.prot(prot)
            rel = self.text(rel)
            att = self.text(att)
        y = self.keap(prot, rel, att)
        y = self.lmhead(y)

        loss = self.criterion(y[:, indices, :], target[:, indices, :])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.train_losses.append(loss.item())
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor | Mapping[text_model_id, Any] | None:
        prot, rel, att, target, indices = batch

        prot = self.prot(prot)
        rel = self.text(rel)
        att = self.text(att)
        
        y = self.keap(prot, rel, att)
        y = self.lmhead(y)

        loss = self.criterion(y[:, indices, :], target[:, indices, :])
        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.val_losses.append(loss.item())


