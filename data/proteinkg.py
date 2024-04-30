import torch
from torch import Tensor
from torch.utils.data import Dataset
import pickle
from typing import List, Tuple


def mask(prot: List[str]):
    t = torch.randint(0, 5, (len(prot)))
    for i in range(len(prot)):
        if t[i] == 0: prot[i] = "[MASK]"
    
    indices = (t==0).nonzero()
    return prot, indices.tolist()


def collate(li: List[Tuple[List[str], str, str, List[str], Tensor]]) -> Tuple[List[List[str]], List[str], List[str], List[List[str]], List[Tensor]]:
    ret = ([li[0]], [li[1]], [li[2]], [li[3]], [li[4]])
    for i in range(1, len(li)):
        ret[0].append(li[i][0])
        ret[1].append(li[i][1])
        ret[2].append(li[i][2])
        ret[3].append(li[i][3])
        ret[4].append(li[i][4])

    return ret


class ProteinKG25(Dataset):
    def __init__(self, file_path: str):
        with open(file_path, "rb") as f:
            self.data = pickle.load(f)
    
    # returns: masked protein sequence, relation text, attribute text, target protein sequence, mask indices
    def __getitem__(self, idx: int) -> Tuple[List[str], str, str, List[str], Tensor]:
        item = self.data[idx]
        prot, indices = mask(list(item[0]))
        return (prot, item[1], item[2], list(item[0]), indices)

    def __len__(self):
        return len(self.data)

