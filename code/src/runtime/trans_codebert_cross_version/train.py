import os, sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import json
from preprocessing.codebert.CodebertDataset import CodebertDataset
from preprocessing.Defects4JLoader import Defects4JLoader
from models.trans.PretainedTrans import PretrainedTrans
from runtime.trans_codebert_cross_version.Trainer import Trainer
from Macros import Macros
from preprocessing import utils
import pytorch_warmup as warmup

def init_model(model_config, train_dataset, device):
    pretrained_model_dict = Macros.MODEL_DICT[args.pretrained_trans]
    
    # pretrained_model = pretrained_model_dict["model"].from_pretrained(pretrained_model_dict["pretrained"])
    pretrained_model = pretrained_model_dict["model"].from_pretrained("/xxx/huggingface/models--microsoft--codebert-base", local_files_only=True)  # locally load codet5
    model = PretrainedTrans(
        trans=pretrained_model,
        embedding_dim=model_config["embedding_dim"],
        num_classes=train_dataset.num_classes,
    ).to(device)

    return model

def train(config, model_config, project, device):
    train_dataset = CodebertDataset(config.train_path + f"/{project}/train")
    validation_dataset = CodebertDataset(config.validation_path + f"/{project}/val")

    train_dataloader = Defects4JLoader(train_dataset, model_config["batch_size"])

    validation_dataloader = Defects4JLoader(validation_dataset, model_config["val_batch_size"])

    start_epoch = 0
    experiment_id = ""
    if len(config.model_path) == 0:
        model = init_model(model_config, train_dataset, device)
        optimizer = optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()), lr=model_config["lr"])
        num_steps = len(train_dataloader) * model_config["num_epochs"]
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
        warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    else:
        model_dict = torch.load(config.model_path)
        model = model_dict["model"]
        optimizer = model_dict["optimizer"]
        lr_scheduler = model_dict["lr_scheduler"]
        warmup_scheduler = model_dict["warmup_scheduler"]
        experiment_id = config.experiment_id
        start_epoch = int(config.model_path.rsplit("_", 1)[-1].split(".pth.tar")[0]) + 1   # extract the epoch number from a string like "epoch_x.pth.tar"
    class_weights = torch.FloatTensor(Macros.class_weights_diff_cross_version[project]) if config.is_diff else torch.FloatTensor(Macros.class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)

    trainer = Trainer(config, model_config, model, optimizer, criterion, lr_scheduler, warmup_scheduler, train_dataloader, validation_dataloader, start_epoch, experiment_id, project)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, default=Macros.model_configs_dir / "codebert_trans_test.json")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--experiment_id", type=str, default="")
    parser.add_argument("--is_diff", action="store_true")

    parser.add_argument("--experiment", type=str, default="codebert_trans")
    parser.add_argument("--train_path", type=str, default=Macros.defects4j_root_dir / "codebert" / "train")
    parser.add_argument("--validation_path", type=str, default=Macros.defects4j_root_dir / "codebert" / "val")
    parser.add_argument("--pretrained_trans", type=str, choices=["codebert", "codet5"], default="codebert")

    args = parser.parse_args()

    projects = ["Lang", "Chart", "Gson", "JacksonCore", "Csv", "Cli"]

    with open(args.model_config, "r") as f:
        model_config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.set_seed(Macros.random_seed)

    # if not os.path.exists(os.path.dirname(model_save_dir)):
    for project in projects:
        train(args, model_config, project, device)
