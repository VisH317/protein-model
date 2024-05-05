from transformers import AutoModel, AutoTokenizer
from typing import List

# these are base BERT models to test with at first. These are being used for their lower compuational complexity and matching with original KeAP architecture
text_model_id = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
prot_model_id = "Rostlab/prot_bert"

class BertModel:
    def __init__(self, model_id, max_length: int = 16384):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(model_id, output_hidden_states=True)
        self.max_length = max_length
    
    def __call__(self, input: List[str]):
        tokens = self.tokenizer.batch_encode_plus(input, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True) # TODO: Set up attention mask here
        output = self.model(**tokens)
        return output.pooler_output
    
