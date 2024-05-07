import os
import pathlib
from src.models import KeAP
from transformers import AutoTokenizer
from data.proteinkg import ProteinKG25

PROTEIN_MODEL_PATH = "Rostlab/prot_bert"
TEXT_MODEL_PATH = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
DECODER_MODEL_PATH = pathlib.Path("../keap_data")

keap = KeAP.from_pretrained(protein_model_path=PROTEIN_MODEL_PATH, text_model_path=TEXT_MODEL_PATH, decoder_model_path=DECODER_MODEL_PATH)

prot_tok = AutoTokenizer.from_pretrained(PROTEIN_MODEL_PATH)
text_tok = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)


# test sample
protein = "MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSE"
rel = "is a"
attribute = "mitochondrion inheritance: The distribution of mitochondria, including the mitochondrial genome, into daughter cells after mitosis or meiosis, mediated by interactions between mitochondria and the cytoskeleton."

def run_pred(protein, rel, attribute):
    protein_tok = prot_tok(protein, return_tensors="pt")
    rel_tok = text_tok(rel, return_tensors="pt")
    att_tok = text_tok(attribute, return_tensors="pt")

    prot_input = (protein_tok.input_ids, protein_tok.attention_mask, protein_tok.token_type_ids)
    rel_input = (rel_tok.input_ids, rel_tok.attention_mask, rel_tok.token_type_ids)
    att_input = (att_tok.input_ids, att_tok.attention_mask, att_tok.token_type_ids)

    out = keap(prot_input, rel_input, att_input)

    return out.pooler_output

# run_pred(protein, rel, attribute)

dataset = ProteinKG25()
