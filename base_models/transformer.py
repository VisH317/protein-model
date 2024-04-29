from transformers import AutoModel, AutoTokenizer

# these are base BERT models to test with at first. These are being used for their lower compuational complexity and matching with original KeAP architecture
text_model_id = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
prot_model_id = "Rostlab/prot_bert"

class BertModel:
    def __init__(self, model_id):
        self.prot_tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.prot_model = AutoModel.from_pretrained(model_id)
    
    def forward(self, input: str):
        tokens = self.prot_tokenizer(input)
        output = self.prot_model(**tokens)
        return output.logits
    
