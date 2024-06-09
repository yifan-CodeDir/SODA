import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
import json
import torch
import os

class RepLogger:
    def __init__(self, log_dir):
        self.log_dir = str(log_dir)

    def add_log(self, epoch, train_loss, plot_data, plot_label, pos_label=1):
        temp_dict = {}
        temp_dict["val/epoch"] = epoch
        temp_dict["val/train_loss"] = train_loss

        with open(self.log_dir + "/rep_log.txt", "a") as file:  # record loss
            file.write(str(temp_dict) + "\n")

        torch.save(torch.cat(plot_data, dim=0), self.log_dir + f'/rep_epoch_{epoch}_data.pt')
        torch.save(torch.cat(plot_label), self.log_dir + f'/rep_epoch_{epoch}_label.pt')

        plot_data = torch.cat(plot_data, dim=0).to('cpu').detach().numpy()
        plot_label = torch.cat(plot_label).to('cpu').detach().numpy()
        plt.scatter(plot_data[:, 0], plot_data[:, 1], c=plot_label)
        if not os.path.exists(os.path.join(self.log_dir, "figure")):
            os.mkdir(os.path.join(self.log_dir, "figure"))
        plt.savefig(
            os.path.join(self.log_dir, "figure", f'figure_epoch_{epoch}.png'))

class ClaLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir

    def compute_metrics(self, preds, labels, pos_label):
        precision = precision_score(labels, preds, pos_label=pos_label)
        recall = recall_score(labels, preds, pos_label=pos_label)
        f1 = f1_score(labels, preds, pos_label=pos_label)
        accuracy = accuracy_score(labels, preds)
        return precision, recall, f1, accuracy

    def add_log(self, epoch, preds_train, labels_train, preds_val, labels_val, train_loss, validation_loss, pos_label=1, print_log=False):
        val_report = classification_report(preds_val, labels_val, output_dict=True, labels=[0, 1])
        train_precision, train_recall, train_f1, train_accuracy = self.compute_metrics(preds_train, labels_train, pos_label)
        validation_precision, validation_recall, validation_f1, validation_accuracy = self.compute_metrics(preds_val, labels_val, pos_label)
        temp_dict = {}
        temp_dict["val/epoch"] = epoch
        temp_dict["val/train_loss"] = train_loss
        temp_dict["val/train_accuracy"] = train_accuracy
        temp_dict["val/train_precision"] = train_precision
        temp_dict["val/train_recall"] = train_recall
        temp_dict["val/train_f1"] = train_f1
        temp_dict["val/validation_loss"] = validation_loss
        temp_dict["val/validation_accuracy"] = validation_accuracy
        temp_dict["val/validation_precision"] = validation_precision
        temp_dict["val/validation_recall"] = validation_recall
        temp_dict["val/validation_f1"] = validation_f1

        with open(str(self.log_dir) + "/cla_log.txt", "a") as file:
            file.write(str(temp_dict) + "\n")

        # run["val/epoch"].log(epoch)
        # run["val/train_loss"].log(train_loss)
        # run["val/train_accuracy"].log(train_accuracy)
        # run["val/train_precision"].log(train_precision)
        # run["val/train_recall"].log(train_recall)
        # run["val/train_f1"].log(train_f1)
        # run["val/validation_loss"].log(validation_loss)
        # run["val/validation_accuracy"].log(validation_accuracy)
        # run["val/validation_precision"].log(validation_precision)
        # run["val/validation_recall"].log(validation_recall)
        # run["val/validation_f1"].log(validation_f1)

        with open(str(self.log_dir) + "/cla_log.txt", "a") as file:
            file.write(f"Val Prec / Recall (Count) {val_report['0']['precision']:.3f} / {val_report['0']['recall']:.3f}({val_report['0']['support']}), {val_report['1']['precision']:.3f} / {val_report['1']['recall']:.3f}({val_report['1']['support']})" + "\n")  
