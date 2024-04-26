import requests
from tqdm import tqdm
import pickle
import argparse


BASE_URL = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/"

def get_go_def(go_id: str):
    res = requests.get(BASE_URL + go_id)
    data = res.json()
    return data["results"][0]["name"], data["results"][0]["definition"]["text"]


def create_annotations(split: str):
    relations = []
    with open("ProteinKG25/relation2id.txt", "r") as f:
        for line in tqdm(f.readlines(), desc="reading relations"):
            relations.append(line.split("\t")[0].strip())

    proteins = []
    with open("ProteinKG25/protein_seq.txt", "r") as f:
        for line in tqdm(f.readlines(), desc="reading proteins"):
            proteins.append(line.strip())
    
    go = []
    with open("ProteinKG25/go_def.txt", "r") as f:
        for line in tqdm(f.readlines(), desc="reading GO"):
            go.append(line.strip())
    
    data = []
    with open(f"ProteinKG25/protein_go_{split}_triplet.txt", "r") as f:
        for line in tqdm(f.readlines(), desc="filling protein-go triplets"):
            index_li = line.split(" ")
            data.append((proteins[int(index_li[0])], relations[int(index_li[1])], go[int(index_li[2])]))
    
    with open(f"proteinkg25_parsed_{split}.pkl", "wb") as f:
        pickle.dump(data, f)

parser = argparse.ArgumentParser("parse_proteinkg")
parser.add_argument("split", help="train, test, valid split to parse")

if __name__ == "__main__":
    args = parser.parse_args()
    print(f"creating annotations for {args.split}")
    create_annotations(args.split)
