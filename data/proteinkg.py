import os
import torch
from torch import Tensor
from torch.utils.data import Dataset
import pickle
from typing import List, Tuple


def mask(prot: List[str]):
    t = torch.randint(5, tuple([len(prot)]))
    for i in range(len(prot)):
        if t[i] == 0: prot[i] = "[MASK]"
    
    indices = (t==0).nonzero()
    return "".join(prot), indices.tolist()


def collate(li: List[Tuple[List[str], str, str, List[str], Tensor]]) -> Tuple[List[str], List[str], List[str], List[List[str]], List[Tensor]]:
    ret = ([li[0]], [li[1]], [li[2]], [li[3]], [li[4]])
    for i in range(1, len(li)):
        ret[0].append(li[i][0])
        ret[1].append(li[i][1])
        ret[2].append(li[i][2])
        ret[3].append(li[i][3])
        ret[4].append(li[i][4])

    return ret


def collate_clip(li: List[Tuple[List[str], str, str, List[str], Tensor]]) -> Tuple[List[List[str]], List[str], Tensor]:
    prot_li = [item[0] for item in li]
    rel_li = [item[1] for item in li]
    att_li = [item[2] for item in li]
    t = torch.eye(len(prot_li))

    return prot_li, rel_li, att_li, t

def collate_clip_combine_text(li: List[Tuple[List[str], str, str, List[str], Tensor]]) -> Tuple[List[List[str]], List[str], Tensor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prot_li = [item[3] for item in li]
    rel_li = [" ".join(item[1].split("_")) + " " + item[2] for item in li]
    t = torch.eye(len(prot_li)).to(device=device)

    return prot_li, rel_li, t


class ProteinKG25(Dataset):
    def __init__(self, file_path: str):
        with open(file_path, "rb") as f:
            self.data = pickle.load(f)
    
    # returns: masked protein sequence, relation text, attribute text, target protein sequence, mask indices
    def __getitem__(self, idx: int) -> Tuple[List[str], str, str, List[str], Tensor]:
        item = self.data[idx]
        prot, indices = mask(list(item[0]))
        return (prot, item[1], item[2], item[0], indices)

    def __len__(self):
        return len(self.data)

