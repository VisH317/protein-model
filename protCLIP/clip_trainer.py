import torch
from protCLIP.arch import ProtCLIPLit, ProtCLIP
from torch.utils.data import DataLoader
from data.proteinkg import ProteinKG25, collate_clip
from lightning import Trainer
from typing import Dict, Any

default_config = {
    "d_model": 1024,
    "d_text": 768,
    "d_clip": 512,
    "d_inter": None,
    "lr": 3e-4,
    "batch_size": 16,
    "val_batch_size": 4
}

default_data_config = {
    "train_data_file": "./data/proteinkg25_parsed_train.pkl",
    "val_data_file": "./data/proteinkg25_parsed_valid.pkl",
}


def train_clip(config: Dict[str, Any] = default_config, data_config: Dict[str, Any] = default_data_config):
    clip = ProtCLIP(config["d_model"], config["d_text"], config["d_clip"], config["d_inter"])
    model = ProtCLIPLit(clip, lr=config["lr"])

    # set up data
    train_data = ProteinKG25(data_config["train_data_file"])
    val_data = ProteinKG25(data_config["val_data_file"])

    # create loaders
    train_loader = DataLoader(train_data, shuffle=False, num_workers=8, batch_size=config["batch_size"], collate_fn=collate_clip)
    val_loader = DataLoader(val_data, shuffle=False, num_workers=8, batch_size=config["val_batch_size"], collate_fn=collate_clip)

    # fit model
    trainer = Trainer(max_epochs=4, profiler="simple", accumulate_grad_batches=4)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # save parameters
    torch.save(clip.state_dict(), "keap.pth")

