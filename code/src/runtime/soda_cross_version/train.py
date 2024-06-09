import os, sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" 

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import json
from preprocessing.codet5.CodeT5Dataset import CodeT5Dataset
from preprocessing.Defects4JLoader import Defects4JLoader
from runtime.soda_cross_version.Trainer import Trainer
from Macros import Macros
from preprocessing import utils
# import pytorch_warmup as warmup
from runtime.soda_cross_version.clip_mlm import CLIP

# use accelerate to train on multiple GPUs
from accelerate import Accelerator
from accelerate.state import AcceleratorState


def init_model(model_config, train_dataset, device):
    clip = CLIP(
        args=model_config,
        dim_text=768,
        model_path=model_config['pretrained_model_path'],
        num_classes=train_dataset.num_classes
    ).to(device)

    return clip


def load_data(config, project):
    train_dataset = CodeT5Dataset(config.train_path + f"/{project}/train")
    dataset_pos = CodeT5Dataset(config.pos_path + f"/{project}/train_pos")
    validation_dataset = CodeT5Dataset(config.validation_path + f"/{project}/val")

    return train_dataset, dataset_pos, validation_dataset 

def train(config, train_dataset, dataset_pos, validation_dataset, model_config, project, device):

    train_dataloader = Defects4JLoader(train_dataset, model_config["batch_size"])
    train_cla_dataloader = Defects4JLoader(train_dataset, model_config["batch_size"])
    pos_dataloader = Defects4JLoader(dataset_pos, model_config["batch_size"])
    validation_dataloader = Defects4JLoader(validation_dataset, model_config["batch_size"])

    rep_start_epoch = 0
    cla_start_epoch = 0
    experiment_id = ""
    if len(config.model_path) == 0:
        model = init_model(model_config, train_dataset, device)
        rep_optimizer = optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()), lr=model_config["lr"])
        cla_optimizer = optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()), lr=model_config["lr_2"])
        # num_steps = len(train_dataloader) * model_config["num_rep_epochs"]
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(rep_optimizer, T_max=num_steps)  # only for representation training
        # warmup_scheduler = warmup.UntunedLinearWarmup(rep_optimizer)                         # only for representation training
    else:
        model_dict = torch.load(config.model_path)
        model = model_dict["model"]
        # rep_optimizer = optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()), lr=model_config["lr"])
        # cla_optimizer = optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()), lr=model_config["lr_2"])
        rep_optimizer = model_dict["rep_optimizer"]
        cla_optimizer = model_dict["cla_optimizer"]
        # lr_scheduler = model_dict["lr_scheduler"]
        experiment_id = config.experiment_id
        rep_start_epoch = model_dict["rep_epoch"] + 1  # epochs to be trained 
        cla_start_epoch = model_dict["cla_epoch"] + 1
        # warmup_scheduler = warmup.UntunedLinearWarmup(rep_optimizer)

    class_weights = torch.FloatTensor(Macros.class_weights_diff_cross_version[project]) if config.is_diff else torch.FloatTensor(Macros.class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)

    # accelerate
    model, rep_optimizer, cla_optimizer, train_dataloader, train_cla_dataloader, pos_dataloader, validation_dataloader = config.accelerator.prepare(
            model, rep_optimizer, cla_optimizer, train_dataloader, train_cla_dataloader, pos_dataloader, validation_dataloader
        )

    trainer = Trainer(config, model_config, model, rep_optimizer, cla_optimizer, criterion, train_dataloader, train_cla_dataloader, pos_dataloader, validation_dataloader, rep_start_epoch, cla_start_epoch, experiment_id, project)
    trainer.train_rep()
    trainer.train_cla()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, default=Macros.model_configs_dir / "soda_trans_test.json")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--experiment_id", type=str, default="")
    parser.add_argument("--is_diff", action="store_true")

    parser.add_argument("--experiment", type=str, default="codet5_token_diff")
    parser.add_argument("--train_path", type=str, default=Macros.defects4j_root_dir / "codet5_token_diff" / "train")
    parser.add_argument("--pos_path", type=str, default=Macros.defects4j_root_dir / "codet5_token_diff" / "train_pos")
    parser.add_argument("--validation_path", type=str, default=Macros.defects4j_root_dir / "codet5_token_diff" / "val")
    parser.add_argument("--pretrained_trans", type=str, choices=["codebert", "codet5"], default="codet5")

    args = parser.parse_args()

    projects = ["Lang", "Chart", "JacksonCore", "Csv", "Cli", "Gson"]
    # projects = ["Lang"]

    with open(args.model_config, "r") as f:
        model_config = json.load(f)

    # use accelerator
    accelerator = Accelerator()
    AcceleratorState().deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"]=model_config['batch_size']
    # AcceleratorState().deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"]=args.grad_acc_steps
    args.accelerator = accelerator

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = accelerator.device
    utils.set_seed(Macros.random_seed)


    # if not os.path.exists(os.path.dirname(model_save_dir)):
    for project in projects:
        # get train, valid, positive data
        train_dataset, dataset_pos, validation_dataset = load_data(args, project)
        train(args, train_dataset, dataset_pos, validation_dataset, model_config, project, device)


