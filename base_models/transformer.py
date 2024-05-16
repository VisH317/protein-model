import torch
from torch import nn, Tensor
from protCLIP.utils import RMSNorm
from transformers import AutoModel, AutoTokenizer
from typing import List


# these are base BERT models to test with at first. These are being used for their lower compuational complexity and matching with original KeAP architecture
text_model_id = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
prot_model_id = "yarongef/DistilProtBert"


class BertPooler(nn.Module):
    def __init__(self, d_model: int, use_norm: bool = False) -> None:
        super().__init__()

        self.lin = nn.Linear(d_model, d_model)
        self.act = nn.Tanh()
        self.use_norm = use_norm

        if self.use_norm: self.norm = RMSNorm(d_model)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.use_norm: hidden_states = self.norm(hidden_states)
        return self.act(self.lin(hidden_states[:, -1]))


class BertModel:
    def __init__(self, model_id, max_len: int = 2048, needs_custom_pooler: bool = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.prot_tokenizer = AutoTokenizer.from_pretrained(model_id, output_hidden_states=True)
        self.prot_model = AutoModel.from_pretrained(model_id, output_hidden_states=True).to(device=self.device)
        self.prot_model.eval()

        self.needs_custom_pooler = needs_custom_pooler
        self.max_len = max_len
    
    def __call__(self, input: str):
        tokens = self.prot_tokenizer(input, return_tensors="pt", max_length=self.max_len, padding=True, truncation=True).to(device=self.device) # TODO: Set up attention mask here
        output = self.prot_model(**tokens)
        return output.hidden_states[0]
    
