import string
import numpy as np
import torch
import random
import os, sys
from sklearn.metrics import f1_score

from tqdm import tqdm

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

from preprocessing.pmt_baseline.PMTDataset import *
from runtime.shared.Logger import Logger

from Macros import Macros
# import neptune.new as neptune

class Trainer:
    def __init__(self, config, model_config, model, optimizer, criterion, dataloader, val_dataloader, start_epoch, experiment_id):
        self.config = config
        self.model_config = model_config
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.device = next(self.model.parameters()).device
        self.start_epoch = start_epoch
        self.experiment_id = experiment_id
        # self.project = project

        (Macros.defects4j_model_dir / config.experiment).mkdir(parents=True, exist_ok=True)
        (Macros.log_dir / config.experiment).mkdir(parents=True, exist_ok=True)   # make dirs for log 

    def train(self):
        # log_file = Macros.defects4j_model_dir / self.config.experiment / "log.txt"
        # plt_file = Macros.defects4j_model_dir / self.config.experiment / "plt.eps"
        logger = Logger(Macros.log_dir / self.config.experiment)
        
        # if self.experiment_id != "":
        #     run = neptune.init_run(project="<PROJECT>", api_token=os.getenv('NEPTUNE_API_TOKEN'), with_id=self.experiment_id)
        # else:
        #     run = neptune.init_run(project="<PROJECT>", api_token=os.getenv('NEPTUNE_API_TOKEN'))
        # run["model/parameters"]=self.model_config
        # run["algorithm"] = self.config.experiment

        best_val_f1 = 0
        stop_counter = 0
        for epoch in range(self.start_epoch, self.start_epoch + 10000):
            val_f1 = self._train_epoch(epoch, logger)
            torch.save({
                'epoch': epoch,
                'model': self.model,
                'optimizer': self.optimizer,
            }, os.path.join(Macros.defects4j_model_dir / self.config.experiment, f'epoch_{epoch}.pth.tar'))

            if val_f1 >= best_val_f1:
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
        # }, os.path.join(Macros.defects4j_model_dir / self.config.experiment, f'epoch_{epoch}.pth.tar'))
        # run.stop()

    def eval_dataset(self, dataloader):
        loss_vals = []
        self.model.eval()
        with torch.no_grad():
            labels_data, preds_data = torch.Tensor([]), torch.Tensor([])
            with tqdm(dataloader, unit="batch") as tepoch:
                for _, sents1, sents2, body, before, after, mutator, labels in tepoch:
                    sents1 = sents1.to(self.device)
                    sents2 = sents2.to(self.device)
                    body = body.to(self.device)
                    before = before.to(self.device)
                    after = after.to(self.device)
                    mutator = mutator.to(self.device)
                    labels = labels.to(self.device)

                    labels_data = torch.cat([labels_data, labels.cpu()])

                    scores, _, _, _ = self.model(sents1, sents2, body, before, after, mutator)
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
        with tqdm(self.dataloader, unit="batch") as tepoch:
            for batch_idx, (_, sents1, sents2, body, before, after, mutator, labels) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch_idx}")
                sents1 = sents1.to(self.device)
                sents2 = sents2.to(self.device)
                body = body.to(self.device)
                before = before.to(self.device)
                after = after.to(self.device)
                mutator = mutator.to(self.device)
                labels = labels.to(self.device)
                labels_train = torch.cat([labels_train, labels.cpu()])

                scores, _, _, _ = self.model(sents1, sents2, body, before, after, mutator)
                predictions = scores.max(dim=1)[1]
                preds_train = torch.cat([preds_train, predictions.cpu()])

                loss = self.criterion(scores, labels)
                tepoch.set_postfix(loss=loss.item())
                curr_loss_vals.append(loss.item())
                (loss/self.model_config["accum_iter"]).backward()

                if self.model_config["max_grad_norm"] is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model_config["max_grad_norm"])

                if ((batch_idx + 1) % self.model_config["accum_iter"] == 0) or (batch_idx + 1 == len(tepoch)):
                    mean_loss = np.mean(curr_loss_vals)
                    # run["train/loss"].log(mean_loss)
                    train_loss_vals.append(mean_loss)
                    curr_loss_vals = []
                    self.optimizer.step()
                    self.optimizer.zero_grad()

        validation_loss, labels_val, preds_val = self.eval_dataset(self.val_dataloader)

        logger.add_log(epoch_idx, preds_train, labels_train, preds_val, labels_val, np.mean(train_loss_vals), validation_loss, print_log=(batch_idx + 1 == len(tepoch)))

        return f1_score(labels_val, preds_val, pos_label=1)