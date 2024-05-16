import torch
from torch import nn, Tensor
from protCLIP.arch import ProtCLIPLit, ProtCLIP
from torch.utils.data import DataLoader
from data.proteinkg import ProteinKG25, collate_clip_combine_text
from lightning import Trainer
from typing import Dict, Any
from base_models.transformer import BertModel, prot_model_id, text_model_id
import wandb
from lightning.pytorch.loggers import WandbLogger
from tqdm import tqdm
import pickle
import time

# yo if you see this I'm doing this because I dont feel like setting up an env file in kaggle ok so please don't copy it :)
WANDB_KEY = "b2e79ea06ca3e1963c1b930a9944bce6938bbb59"

default_config = {
    "d_model": 1024,
    "d_text": 768,
    "d_clip": 512,
    "d_inter": None,
    "n_epochs": 3,
    "max_epoch_len": 10000,
    "lr": 9e-5,
    "batch_size": 32,
    "grad_accum": 8,
    "val_batch_size": 4
}

# NOTE: I NEED TO CHECK HOW BCE IS COMPUTED, BECUASE THE COMPUTATION HAS TO BE 1-to-1

default_data_config = {
    "train_data_file": "./data/proteinkg25_parsed_train.pkl",
    "val_data_file": "./data/proteinkg25_parsed_valid.pkl",
}


def train_clip_old(config: Dict[str, Any] = default_config, data_config: Dict[str, Any] = default_data_config):

    wandb.login(key=WANDB_KEY)

    logger = WandbLogger(project="protCLIP")

    clip = ProtCLIP(config["d_model"], config["d_text"], config["d_clip"], config["d_inter"])
    model = ProtCLIPLit(clip, lr=config["lr"])

    # set up data
    train_data = ProteinKG25(data_config["train_data_file"])
    val_data = ProteinKG25(data_config["val_data_file"])

    # create loaders
    train_loader = DataLoader(train_data, shuffle=False, num_workers=8, batch_size=config["batch_size"], collate_fn=collate_clip_combine_text)
    val_loader = DataLoader(val_data, shuffle=False, num_workers=8, batch_size=config["val_batch_size"], collate_fn=collate_clip_combine_text)

    # fit model
    trainer = Trainer(max_epochs=4, profiler="simple", accumulate_grad_batches=4, logger=logger)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # save parameters
    torch.save(clip.state_dict(), "keap.pth")


def train_clip(config: Dict[str, Any] = default_config, data_config: Dict[str, Any] = default_data_config):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.login(key=WANDB_KEY)
    wandb.init(project="protCLIP")

    prot_model = BertModel(prot_model_id)
    text_model = BertModel(text_model_id)
    clip = ProtCLIP(config["d_model"], config["d_text"], config["d_clip"], config["d_inter"]).to(device=device)

    # set up data
    train_data = ProteinKG25(data_config["train_data_file"])
    val_data = ProteinKG25(data_config["val_data_file"])

    # train + optim
    crit_text = nn.CrossEntropyLoss()
    crit_prot = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(clip.parameters(), config["lr"], betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
    scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.5, total_iters=config["batch_size"]*config["max_epoch_len"] // (15 * config["grad_accum"]))
    exp_sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)

    val_losses = [4]
    train_losses = []

    for epoch in range(config["n_epochs"]):

        # setup loaders
        train_loader = DataLoader(train_data, shuffle=False, batch_size=config["batch_size"], collate_fn=collate_clip_combine_text)
        val_loader = DataLoader(val_data, shuffle=False, batch_size=config["val_batch_size"], collate_fn=collate_clip_combine_text)
        val_loader_iter = iter(val_loader)

        opt.zero_grad()
        for ix, data in (bar := tqdm(enumerate(train_loader), total=config["max_epoch_len"], desc=f"Epoch: {epoch+1}")):
            prot, rel, _target = data

            # print("prot: ", prot)
            # print("text: ", rel)
            # print("target: ", target)

            with torch.no_grad():
                prot_emb = prot_model(prot)
                text_emb = text_model(rel)
                # print("prot_emb: ", prot_emb)
                # print("text_emb: ", text_emb)
            out_prot, out_text = clip(prot_emb, text_emb)

            target = torch.arange(out_prot.size()[0], dtype=torch.long, device=device)
            loss = (crit_prot(out_prot, target) + crit_text(out_text, target.t()))/2
            print("loss: ", loss)
            loss.backward()


            if (ix+1) % config["grad_accum"] == 0:
                opt.step()
                opt.zero_grad()
                scheduler.step()
            
            train_losses.append(loss.item())
            wandb.log({"train_loss": loss.item()})
            # wandb.log
            bar.set_description(f"epoch: {epoch + 1}, Loss: {round(train_losses[-1], 4)}, Val Loss: {round(val_losses[-1], 4)}")

            if ix >= config["max_epoch_len"]: break

            time.sleep(3)

            # validation loop
            if ix % 32 == 0:
                with torch.no_grad():
                    try:
                        prot, rel, _target = next(val_loader_iter)
                    except:
                        val_loader = DataLoader(val_data, config["val_batch_size"], shuffle=True)
                        val_loader_iter = iter(val_loader)
                        prot, rel, _target = next(val_loader_iter)
                    
                    prot_emb = prot_model(prot)
                    text_emb = text_model(rel)
                    target = torch.arange(prot_emb.size()[0], dtype=torch.long, device=device)
                    out_prot, out_text = clip(prot_emb, text_emb)
                    loss = (crit_prot(out_prot, target) + crit_text(out_text, target.t()))/2

                    # val_losses.append(loss.item())
                    wandb.log({"val_loss": loss.item()})
                    # bar.set_description(f"Epoch: {epoch+1}, Loss: {round(train_losses[-1], 4)}, Val loss: {round(val_losses[-1], 4)}")
        
        exp_sched.step()
    
    torch.save(clip.state_dict(), "clip.pt")
    with open("loss.pkl", "wb") as f:
        pickle.dump([train_losses, val_losses], f)
    
    return clip