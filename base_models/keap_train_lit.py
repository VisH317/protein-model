import torch
from torch.utils.data import DataLoader
from data.proteinkg import ProteinKG25, collate
from base_models.keap_trainer import KeAPL, LMHead, KeAP
from typing import Dict, Any
from lightning import Trainer


default_config = {
    "d_model": 512,
    "d_prot": 1024,
    "d_rel": 768,
    "d_att": 768,
    "d_attn": 128,
    "n_enc": 4,
    "vocab_size": 30,
    "d_ff": None,
    "dropout_p": 0.05,
    "lr": 3e-4,
    "batch_size": 16,
    "val_batch_size": 4,
}

default_data_config = {
    "train_data_file": "data/proteinkg25_parsed_train.pkl",
    "val_data_file": "data/proteinkg25_parsed_val.pkl",
}

def train_keap(config: Dict[str, Any] = default_config, data_config: Dict[str, Any] = default_data_config):

    # setup lightning module
    keap = KeAP(config["n_enc"], config["d_prot"], config["d_model"], config["d_att"], config["d_rel"], config["d_attn"], d_ff=config["d_ff"], dropout_p=config["dropout_p"])
    lmhead = LMHead(config["d_model"], config["vocab_size"])

    keap_lit = KeAPL(keap, lmhead, lr=config["lr"])

    # set up data
    train_data = ProteinKG25(data_config["train_data_file"])
    val_data = ProteinKG25(data_config["val_data_file"])

    # create loaders
    train_loader = DataLoader(train_data, shuffle=True, batch_size=config["batch_size"], collate_fn=collate)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=config["val_batch_size"], collate_fn=collate)

    # fit model
    trainer = Trainer(max_epochs=4, profiler="simple", accumulate_grad_batches=4)
    trainer.fit(keap_lit, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # save parameters
    torch.save(keap.state_dict(), "keap.pth")
    torch.save(lmhead.state_dict(), "lmhead.pth")
