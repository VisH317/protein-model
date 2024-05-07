import torch
from transformers import AutoModel, AutoTokenizer
from typing import List

# these are base BERT models to test with at first. These are being used for their lower compuational complexity and matching with original KeAP architecture
text_model_id = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
prot_model_id = "Rostlab/prot_bert"

class BertModel:
    def __init__(self, model_id, max_len: int = 2048):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.prot_tokenizer = AutoTokenizer.from_pretrained(model_id, output_hidden_states=True)
        self.prot_model = AutoModel.from_pretrained(model_id, output_hidden_states=True).to(device=self.device)
        self.max_len = max_len
    
    def __call__(self, input: str):
        tokens = self.prot_tokenizer(input, return_tensors="pt", max_length=self.max_len, padding=True, truncation=True).to(device=self.device) # TODO: Set up attention mask here
        output = self.prot_model(**tokens)
        return output.pooler_output
    
