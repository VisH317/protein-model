from torch.utils.data import Dataset
import pickle

class ProteinKG25(Dataset):
    def __init__(self, file_path: str):
        with open(file_path, "rb") as f:
            self.data = pickle.load(f)
    
    def __getitem__(self, idx: int):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

