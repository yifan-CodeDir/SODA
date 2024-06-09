import random
from sklearn.metrics import f1_score
import string
import torch
import sys, os
from itertools import cycle
from Logger import RepLogger, ClaLogger

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from Macros import Macros
import numpy as np
from tqdm import tqdm, trange
import torch.nn.functional as F
# import neptune.new as neptune
from accelerate import Accelerator


def cosine_similarity_matrix(features1, features2):
    features1_normalized = F.normalize(features1, p=2, dim=-1)
    features2_normalized = F.normalize(features2, p=2, dim=-1)
    similarity_matrix = torch.matmul(features1_normalized, features2_normalized.transpose(0, 1))
    return similarity_matrix

def sup_contrastive_loss(temp, embedding, label, device):
    """calculate the contrastive loss
    """
    # cosine similarity between embeddings
    cosine_sim = cosine_similarity_matrix(embedding, embedding)
    # print(cosine_sim.size())
    # remove diagonal elements from matrix
    n = cosine_sim.shape[0]
    dis = cosine_sim[~torch.eye(n, dtype=bool)].reshape(n, -1)
    # apply temprature to elements
    dis = dis / temp
    cosine_sim = cosine_sim / temp
    # apply exp to elements
    dis = torch.exp(dis)
    cosine_sim = torch.exp(cosine_sim)

    # calculate row sum
    row_sum = torch.sum(dis, dim=1)

    # calculate outer sum
    contrastive_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
    for i in range(n):
        n_i = (label == label[i]).sum().item() - 1
        inner_sum = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        # calculate inner sum
        for j in range(n):
            if label[i] == label[j] and i != j:
                inner_sum = inner_sum + torch.log(cosine_sim[i][j] / row_sum[i])
        if n_i != 0:
            contrastive_loss += (inner_sum / (-n_i))
    # return (contrastive_loss / n)  # I add mean here to normalize 
    return contrastive_loss  # don't normalize


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        return torch.mean((label) * torch.pow(euclidean_distance, 2) +
                          (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0),
                                                  2))


class Trainer:
    def __init__(self, config, model_config, model, rep_optimizer, cla_optimizer, criterion, train_dataloader, train_cla_dataloader, pos_dataloader, validation_dataloader, rep_start_epoch, cla_start_epoch, experiment_id):
        self.config = config
        self.model_config = model_config
        self.model = model
        self.rep_optimizer = rep_optimizer
        self.criterion = criterion
        # self.project = project
        # self.lr_scheduler = lr_scheduler
        # self.warmup_scheduler = warmup_scheduler

        self.cla_optimizer = cla_optimizer

        self.train_dataloader = train_dataloader
        self.train_cla_dataloader = train_cla_dataloader
        self.pos_dataloader = pos_dataloader
        self.validation_dataloader = validation_dataloader

        self.device = next(self.model.parameters()).device
        self.rep_start_epoch = rep_start_epoch
        self.cla_start_epoch = cla_start_epoch
        self.experiment_id = experiment_id
        self.c_loss = ContrastiveLoss()
        (Macros.defects4j_model_dir / config.experiment).mkdir(parents=True, exist_ok=True)
        (Macros.log_dir / config.experiment).mkdir(parents=True, exist_ok=True)   # make dirs for log 

    def train_rep(self):
        logger = RepLogger(Macros.log_dir / self.config.experiment)
        # if self.experiment_id != "":
        #     run = neptune.init_run(project="<PROJECT>", api_token=os.getenv('NEPTUNE_API_TOKEN'), with_id=self.experiment_id)
        # else:
        #     run = neptune.init_run(project="<PROJECT>", api_token=os.getenv('NEPTUNE_API_TOKEN'))
        # run["model/parameters"]=self.model_config
        # run["algorithm"] = self.config.experiment
        
        for epoch in range(self.rep_start_epoch, self.rep_start_epoch + 40):
            train_loss_val = self.train_rep_epoch(epoch, logger)
            if train_loss_val < 0.5:  # train util the representation has been learned (0.1 for fold4, 0.3 for other projects)
                break

        self.rep_start_epoch = epoch  # revise to record in train_cla_epoch (epochs to be trained)
    
    def train_rep_epoch(self, epoch_idx, logger):
        self.model.train()

        plot_data = []
        plot_label = []

        train_loss_vals = []
        curr_loss_vals = []
        # labels_train, preds_train = torch.Tensor([]), torch.Tensor([])
        with tqdm(zip(self.train_dataloader, cycle(self.pos_dataloader)), total=len(self.train_dataloader), unit="batch") as tepoch:
            for batch_idx, (data, data_p) in enumerate(tepoch):
                ids, mask, idx, labels = data[0], data[1], data[2], data[3]
                ids_p, mask_p, idx_p, labels_p = data_p[0], data_p[1], data_p[2], data_p[3]

                # to device
                tepoch.set_description(f"Epoch {epoch_idx}")
                ids = ids.to(self.device)
                mask = mask.to(self.device)
                idx = idx.to(self.device)
                labels = labels.to(self.device)
                ids_p = ids_p.to(self.device)
                mask_p = mask_p.to(self.device)
                idx_p = idx_p.to(self.device)
                labels_p = labels_p.to(self.device)
                # labels_train = torch.cat([labels_train, labels.cpu()]) # calculate statistics
                
                # calculate loss and perform backward
                CLS1, CLS2 = self.model(text1=ids,
                                    mask1=mask,
                                    text2=ids_p,
                                    mask2=mask_p,
                                    training_classifier=False)
                loss = self.c_loss(CLS1, CLS2, labels & labels_p).sum()
                
                plot_data.append(CLS1)
                plot_label.append(labels)

                # gather for metrics
                batch_loss = self.config.accelerator.gather_for_metrics(loss)
                curr_loss_vals.extend(batch_loss.cpu().tolist())

                tepoch.set_postfix(loss=loss.item())
                # (loss/self.model_config["accum_iter"]).backward()
                self.config.accelerator.backward(loss/self.model_config["accum_iter"])

                if ((batch_idx + 1) % self.model_config["accum_iter"] == 0) or (batch_idx + 1 == len(tepoch)):   # do backward propagation and update model
                    mean_loss = np.mean(curr_loss_vals)
                    # run["train/loss"].log(mean_loss)
                    # run["train/last_lr"].log(self.lr_scheduler.get_last_lr()[0])
                    # run["train/lr"].log(self.lr_scheduler.get_lr()[0])
                    # run["train/warm_up_step"].log(self.warmup_scheduler.last_step)
                    # run["train/warm_up_lr"].log(self.optimizer.param_groups[0]['lr'])                    
                    train_loss_vals.append(mean_loss)
                    curr_loss_vals = []
                    self.rep_optimizer.step()
                    # with self.warmup_scheduler.dampening():
                    #     self.lr_scheduler.step()
                    self.rep_optimizer.zero_grad()

        # plot loss and representaion vector to check if the model get fully trained
        # if ((batch_idx + 1) % self.model_config["log_iter"] == 0) or (batch_idx + 1 == len(tepoch)):
        logger.add_log(epoch_idx, np.mean(train_loss_vals), plot_data, plot_label)
        # save model
        if self.config.accelerator.is_main_process:
            model_to_save = self.model.module if hasattr(self.model,'module') else self.model
            torch.save({
                'rep_epoch': epoch_idx,  # epochs have been trained
                'cla_epoch': self.cla_start_epoch-1,
                'model': model_to_save,
            }, os.path.join(Macros.defects4j_model_dir / self.config.experiment, f'rep_epoch_{epoch_idx}_cla_epoch_{self.cla_start_epoch-1}.pth.tar'))
        
        return np.mean(train_loss_vals)

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

                # labels_data = torch.cat([labels_data, labels.cpu()])

                scores, _ = self.model(ids, mask, training_classifier=True)
                predictions = scores.max(dim=1)[1]
                # preds_data = torch.cat([preds_data, predictions.cpu()])

                loss = self.criterion(scores, labels)
                loss_vals.append(loss.item())

                # collect labels and batch_predictions on one machine to evaluate
                batch_labels, batch_predictions = self.config.accelerator.gather_for_metrics((labels, predictions))
                labels_data = torch.cat([labels_data, batch_labels.cpu()])
                preds_data = torch.cat([preds_data, batch_predictions.cpu()])

        self.model.train()
        return np.mean(loss_vals), labels_data, preds_data

    def train_cla(self):
        logger = ClaLogger(Macros.log_dir / self.config.experiment)
        
        best_val_f1 = 0
        stop_counter = 0
        for epoch in range(self.cla_start_epoch, self.cla_start_epoch + 1000):
            val_f1 = self.train_cla_epoch(epoch, logger)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                stop_counter = 0
            else:
                stop_counter += 1
            
            if stop_counter == 3:  # if continous three epochs do not reach higher val_f1, we think the training is converged 
                break
        
    def train_cla_epoch(self, epoch_idx, logger):
        self.model.train()

        train_loss_vals = []
        curr_loss_vals = []
        labels_train, preds_train = torch.Tensor([]), torch.Tensor([])
        with tqdm(self.train_cla_dataloader, unit="batch") as tepoch:
            for batch_idx, (ids, mask, idx, labels) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch_idx}")
                ids = ids.to(self.device)
                mask = mask.to(self.device)
                idx = idx.to(self.device)
                labels = labels.to(self.device)
                labels_train = torch.cat([labels_train, labels.cpu()])
                
                scores, pooled_hidden_state = self.model(ids, mask, training_classifier=True)  # train classifier
                predictions = scores.max(dim=1)[1]
                preds_train = torch.cat([preds_train, predictions.cpu()])

                # note that the loss consists of two parts: cross entropy loss and supervised contrastive loss
                lam = self.model_config["lambda"]
                tem = self.model_config["temperature"]
                cross_loss = self.criterion(scores, labels)

                # contrastive_l = sup_contrastive_loss(tem, pooled_hidden_state, labels, self.device) 
                # loss = (lam * contrastive_l) + (1 - lam) * (cross_loss)
                loss = cross_loss
                
                # loss = self.criterion(scores, labels)
                curr_loss_vals.append(loss.item())
                tepoch.set_postfix(loss=loss.item())
                # (loss/self.model_config["accum_iter"]).backward()
                self.config.accelerator.backward(loss/self.model_config["accum_iter"])

                if ((batch_idx + 1) % self.model_config["accum_iter"] == 0) or (batch_idx + 1 == len(tepoch)):
                    mean_loss = np.mean(curr_loss_vals)
                    # run["train/loss"].log(mean_loss)
                    # run["train/last_lr"].log(self.lr_scheduler.get_last_lr()[0])
                    # run["train/lr"].log(self.lr_scheduler.get_lr()[0])
                    # run["train/warm_up_step"].log(self.warmup_scheduler.last_step)
                    # run["train/warm_up_lr"].log(self.optimizer.param_groups[0]['lr'])                    
                    train_loss_vals.append(mean_loss)
                    curr_loss_vals = []
                    self.cla_optimizer.step()
                    self.cla_optimizer.zero_grad()

        validation_loss, labels_val, preds_val = self.eval_dataset(self.validation_dataloader)

        logger.add_log(epoch_idx, preds_train, labels_train, preds_val, labels_val, np.mean(train_loss_vals), validation_loss, print_log=True)
        if self.config.accelerator.is_main_process:
            model_to_save = self.model.module if hasattr(self.model,'module') else self.model
            torch.save({
                'rep_epoch': self.rep_start_epoch,  # epochs have been trained
                'cla_epoch': epoch_idx,
                'model': model_to_save,
            }, os.path.join(Macros.defects4j_model_dir / self.config.experiment, f'rep_epoch_{self.rep_start_epoch}_cla_epoch_{epoch_idx}.pth.tar'))
        
        return f1_score(labels_val, preds_val, pos_label=1)