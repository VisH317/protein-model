import torch
import pathlib
from src.models import KeAP
from transformers import AutoTokenizer
from data.proteinkg import ProteinKG25, collate_clip
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

PROTEIN_MODEL_PATH = "Rostlab/prot_bert"
TEXT_MODEL_PATH = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
DECODER_MODEL_PATH = pathlib.Path("../keap_data")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

keap = KeAP.from_pretrained(protein_model_path=PROTEIN_MODEL_PATH, text_model_path=TEXT_MODEL_PATH, decoder_model_path=DECODER_MODEL_PATH).to(device=device)

prot_tok = AutoTokenizer.from_pretrained(PROTEIN_MODEL_PATH)
text_tok = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)


# test sample
protein = "MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSE"
rel = "is a"
attribute = "mitochondrion inheritance: The distribution of mitochondria, including the mitochondrial genome, into daughter cells after mitosis or meiosis, mediated by interactions between mitochondria and the cytoskeleton."

def run_pred(protein, rel, attribute):
    max_prot_len = max([len(prot) for prot in protein])
    max_rel_len = max([len(r) for r in rel])
    max_att_len = max([len(att) for att in attribute])

    protein_tok = prot_tok(protein, return_tensors="pt", max_length=max_prot_len, truncation=True, padding=True).to(device=device)
    rel_tok = text_tok(rel, return_tensors="pt", max_length=max_rel_len, truncation=True, padding=True).to(device=device)
    att_tok = text_tok(attribute, return_tensors="pt", max_length=max_att_len, truncation=True, padding=True).to(device=device)

    prot_input = (protein_tok.input_ids, protein_tok.attention_mask, protein_tok.token_type_ids)
    rel_input = (rel_tok.input_ids, rel_tok.attention_mask, rel_tok.token_type_ids)
    att_input = (att_tok.input_ids, att_tok.attention_mask, att_tok.token_type_ids)

    out = keap(prot_input, rel_input, att_input)

    return out.pooler_output

# run_pred(protein, rel, attribute)

def create_store():
    dataset = ProteinKG25("data/proteinkg25_parsed_train.pkl")
    loader = DataLoader(dataset, batch_size=8, num_workers=8, collate_fn=collate_clip)

    store = []

    for i, x in tqdm(enumerate(loader), desc="test", total=len(dataset)//8):
        prot, rel, att, _t = x
        out = run_pred(prot, rel, att)
        store += [out[ix] for ix in out.size()[0]]
    
    with open("store.pkl") as f:
        pickle.dump(store, f)

