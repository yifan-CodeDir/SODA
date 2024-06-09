import json
import os, sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from models.baselines.PMTNLC import *
from preprocessing.pmt_baseline.PMTDataset import *

from preprocessing.pmt_baseline.PMTDataLoader import *
from runtime.pmt_baseline_cross_version.Trainer import Trainer
from Macros import Macros
from preprocessing import utils

def train(config, model_config, project, device):
    train_dataset = PMTDataset(config.train_path + f"/{project}/train",
                                    config.sent_vocab_path + f"/{project}/vocab_method_name.pkl", config.body_vocab_path + f"/{project}/vocab_body.pkl", max_sent_length=150)
    train_dataloader = PMTDataLoader(train_dataset, model_config["batch_size"])

    val_dataset = PMTDataset(config.validation_path + f"/{project}/val",
                                    config.sent_vocab_path + f"/{project}/vocab_method_name.pkl", config.body_vocab_path + f"/{project}/vocab_body.pkl", max_sent_length=150)
    val_dataloader = PMTDataLoader(val_dataset, model_config["batch_size"])

    # num_mutator = 8  

    start_epoch = 0
    experiment_id = ""
    if len(config.model_path) == 0:
            model = PMTNLC(
                num_classes=train_dataset.num_classes,
                name_vocab_size=train_dataset.name_vocab_size,
                body_vocab_size=train_dataset.body_vocab_size,
                name_embed_dim=model_config["name_embed_dim"],
                body_embed_dim=model_config["body_embed_dim"],
                gru_hidden_dim=model_config["gru_hidden_dim"],
                gru_num_layers=model_config["gru_num_layers"],
                att_dim=model_config["att_dim"],
                num_mutator=train_dataset.num_mutator,
                dropout=model_config["dropout"]
            ).to(device)
            optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=model_config["lr"])
    else:
        model_dict = torch.load(config.model_path)
        model = model_dict["model"]
        optimizer = model_dict["optimizer"]
        experiment_id = config.experiment_id
        start_epoch = int(config.model_path.rsplit("_", 1)[1].split(".pth.tar")[0]) + 1

    # criterion = nn.NLLLoss(reduction='sum').to(device) # option 1
    criterion = nn.CrossEntropyLoss().to(device)  # option 2
    trainer = Trainer(config, model_config, model, optimizer, criterion, train_dataloader, val_dataloader, start_epoch, experiment_id, project)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--project", type=str, default="")

    parser.add_argument("--model_config", type=str, default=Macros.model_configs_dir / "pmt_baseline.json")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--experiment_id", type=str, default="")

    parser.add_argument("--experiment", type=str, default="pmt_baseline")
    parser.add_argument("--train_path", type=str, default=Macros.defects4j_root_dir / "pmt_baseline_ordered" / "train")
    parser.add_argument("--validation_path", type=str, default=Macros.defects4j_root_dir / "pmt_baseline_ordered" / "val")
    parser.add_argument("--sent_vocab_path", type=str, default=Macros.defects4j_root_dir / "pmt_baseline_ordered" / "vocab_method_name.pkl")
    parser.add_argument("--body_vocab_path", type=str, default=Macros.defects4j_root_dir / "pmt_baseline_ordered" / "vocab_body.pkl")

    args = parser.parse_args()

    with open(args.model_config, "r") as f:
        model_config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.set_seed(Macros.random_seed)

    # projects = ["Lang", "Chart", "Gson", "Cli", "JacksonCore", "Csv"]
    # projects = ["Csv"]

    # if not os.path.exists(os.path.dirname(model_save_dir)):
    # for project in projects:
    train(args, model_config, args.project, device)
