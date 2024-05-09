import lightning as pl
from torch import nn, Tensor
from protCLIP.transforms import CrossGatedTransformer
from protCLIP.heads import LMHead
from base_models.transformer import BertModel, text_model_id

class CrossGate(pl.LightningModule):
    def __init__(self, transformer: CrossGatedTransformer, head: LMHead) -> None:
        super().__init__()
        self.transformer = transformer
        self.head = head

        self.text = BertModel(text_model_id)
        self.criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)

    
    def forward(self, batch: Tensor) -> Tensor:
        pass
