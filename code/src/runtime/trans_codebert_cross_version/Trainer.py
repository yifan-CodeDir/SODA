import random

import string
import torch
import sys, os
from sklearn.metrics import f1_score

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from Macros import Macros
import numpy as np
from runtime.shared.Logger import Logger
from tqdm import tqdm
# import neptune.new as neptune

class Trainer:
    def __init__(self, config, model_config, model, optimizer, criterion, lr_scheduler, warmup_scheduler, train_dataloader, validation_dataloader, start_epoch, experiment_id, project):
        self.config = config
        self.model_config = model_config
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.warmup_scheduler = warmup_scheduler
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.device = next(self.model.parameters()).device
        self.start_epoch = start_epoch
        self.experiment_id = experiment_id
        self.project = project
        (Macros.defects4j_model_dir / config.experiment / self.project).mkdir(parents=True, exist_ok=True)
        (Macros.log_dir / config.experiment / self.project).mkdir(parents=True, exist_ok=True)   # make dirs for log 

    def train(self):
        logger = Logger(Macros.log_dir / self.config.experiment / self.project)
        # if self.experiment_id != "":
        #     run = neptune.init_run(project="<PROJECT>", api_token=os.getenv('NEPTUNE_API_TOKEN'), with_id=self.experiment_id)
        # else:
        #     run = neptune.init_run(project="<PROJECT>", api_token=os.getenv('NEPTUNE_API_TOKEN'))
        # run["model/parameters"]=self.model_config
        # run["algorithm"] = self.config.experiment
        
        best_val_f1 = 0
        stop_counter = 0
        for epoch in range(self.start_epoch, self.start_epoch + 1000):   # train until converge
            val_f1 = self._train_epoch(epoch, logger)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                stop_counter = 0
            else:
                stop_counter += 1
            
            if stop_counter == 3:  # if continous three epochs do not reach higher val_f1, we think the training is converged 
                break
        
        # torch.save({
        #     'epoch': epoch,
        #     'model': self.model,
        #     'optimizer': self.optimizer,
        # }, os.path.join(Macros.defects4j_model_dir / self.config.experiment / self.project, f'epoch_{epoch}_final.pth.tar'))
        # run.stop()

    def eval_dataset(self, dataloader):
        loss_vals = []
        self.model.eval()
        with torch.no_grad():
            labels_data, preds_data = torch.Tensor([]), torch.Tensor([])
            for batch_idx, (ids, mask, idx, labels) in enumerate(dataloader):
                ids = ids.to(self.device)
                mask = mask.to(self.device)
                labels = labels.to(self.device)
                idx = idx.to(self.device)

                labels_data = torch.cat([labels_data, labels.cpu()])

                scores = self.model(ids, mask)
                predictions = scores.max(dim=1)[1]
                preds_data = torch.cat([preds_data, predictions.cpu()])

                loss = self.criterion(scores, labels)
                loss_vals.append(loss.item())
        self.model.train()
        return np.mean(loss_vals), labels_data, preds_data


    def _train_epoch(self, epoch_idx, logger):
        self.model.train()

        train_loss_vals = []
        curr_loss_vals = []
        labels_train, preds_train = torch.Tensor([]), torch.Tensor([])
        # curr_i = 0
        with tqdm(self.train_dataloader, unit="batch") as tepoch:
            for batch_idx, (ids, mask, idx, labels) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch_idx}")
                ids = ids.to(self.device)
                mask = mask.to(self.device)
                idx = idx.to(self.device)
                labels = labels.to(self.device)
                labels_train = torch.cat([labels_train, labels.cpu()])
                
                scores = self.model(ids, mask)
                predictions = scores.max(dim=1)[1]
                preds_train = torch.cat([preds_train, predictions.cpu()])
                loss = self.criterion(scores, labels)
                curr_loss_vals.append(loss.item())
                tepoch.set_postfix(loss=loss.item())
                (loss/self.model_config["accum_iter"]).backward()

                if ((batch_idx + 1) % self.model_config["accum_iter"] == 0) or (batch_idx + 1 == len(tepoch)):
                    mean_loss = np.mean(curr_loss_vals)
                    # run["train/loss"].log(mean_loss)
                    # run["train/last_lr"].log(self.lr_scheduler.get_last_lr()[0])
                    # run["train/lr"].log(self.lr_scheduler.get_lr()[0])
                    # run["train/warm_up_step"].log(self.warmup_scheduler.last_step)
                    # run["train/warm_up_lr"].log(self.optimizer.param_groups[0]['lr'])                    
                    train_loss_vals.append(mean_loss)
                    curr_loss_vals = []
                    self.optimizer.step()
                    with self.warmup_scheduler.dampening():
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()

        validation_loss, labels_val, preds_val = self.eval_dataset(self.validation_dataloader)

        logger.add_log(epoch_idx, preds_train, labels_train, preds_val, labels_val, np.mean(train_loss_vals), validation_loss, print_log=(batch_idx + 1 == len(tepoch)))
        torch.save({
            'epoch': epoch_idx,
            'model': self.model,
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler,
            'warmup_scheduler': self.warmup_scheduler
        }, os.path.join(Macros.defects4j_model_dir / self.config.experiment / self.project, f'epoch_{epoch_idx}.pth.tar'))

        return f1_score(labels_val, preds_val, pos_label=1)
