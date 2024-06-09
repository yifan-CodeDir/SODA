import json
import pickle
import os, sys
from statistics import mean
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from tqdm import tqdm

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

import argparse

import torch
import torch.nn as nn

from models.baselines.PMTNLC import *
from preprocessing.pmt_baseline_suite.PMTDatasetSuite import *
from preprocessing.codebert_token_diff_suite.CodebertDatasetSuite import *
from preprocessing.codebert_token_diff_suite.Defects4JLoaderSuite import Defects4JLoaderSuite
from preprocessing.pmt_baseline_suite.PMTDataLoaderSuite import *
from Macros import Macros
# import neptune.new as neptune

from sklearn import metrics
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  

def get_file_scores(config, device):
    if config.is_trans:
        val_dataset = CodebertDatasetSuite(config.validation_dir)
        val_dataloader = Defects4JLoaderSuite(val_dataset, 1)
        # files = [f"epoch_{epoch}.pth.tar" for epoch in range(model_config["num_epochs"])]
        files = ["best_model.pth.tar"]
    else:
        val_dataset = PMTDatasetSuite(config.validation_dir,
                                    config.sent_vocab_path, config.body_vocab_path, max_sent_length=150)
        val_dataloader = PMTDataLoaderSuite(val_dataset, 1)
        # files = [f"epoch_{epoch}.pth.tar" for epoch in range(model_config["num_epochs"])]
        files = ["best_model.pth.tar"]

    files_list = []
    for filename in files:
        # epoch = int(filename.split("_")[1].split(".")[0])
        # curr_file = {"epoch":epoch, "name":filename, "data": []}
        curr_file = {"name":filename, "data": []}
        model_dict = torch.load(f"{config.model_dir}/{filename}")
        model = model_dict["model"]
        model.eval()
        with torch.no_grad():
            labels_data, preds_data, curr_preds = torch.Tensor([]), torch.Tensor([]), torch.Tensor([])
            with tqdm(val_dataloader, unit="batch") as tepoch:
                for suite_batch in tepoch:
                    for suite in suite_batch:
                        if config.is_trans:
                            if config.is_cl:
                                curr_suite = {"scores": [], "label": suite["label"]}
                                for mutant in suite["mutants"]:
                                    ids = torch.LongTensor(mutant["embed"]).to(device)
                                    mask = torch.LongTensor(mutant["mask"]).to(device)
                                    scores, _ = model(ids.unsqueeze(0), mask.unsqueeze(0), training_classifier=True)
                                    # scores = model(ids.unsqueeze(0), mask.unsqueeze(0))
                                    curr_suite["scores"].append(scores)
                                curr_file["data"].append(curr_suite)
                            else:
                                curr_suite = {"scores": [], "label": suite["label"]}
                                for mutant in suite["mutants"]:
                                    ids = torch.LongTensor(mutant["embed"]).to(device)
                                    mask = torch.LongTensor(mutant["mask"]).to(device)
                                    scores = model(ids.unsqueeze(0), mask.unsqueeze(0))
                                    curr_suite["scores"].append(scores)
                                curr_file["data"].append(curr_suite)

                        else:
                            curr_suite = {"scores": [], "label": suite["label"]}
                            for mutant in suite["mutants"]:
                                _, sents1, sents2, body, before, after, mutator, labels = mutant                      
                                sents1 = sents1.to(device)
                                sents2 = sents2.to(device)
                                before = before.to(device)
                                after = after.to(device)
                                body = body.to(device)
                                mutator = mutator.to(device)
                                labels = labels.to(device)
                                scores, _, _, _ = model(sents1, sents2, body, before, after, mutator)
                                curr_suite["scores"].append(scores)
                            curr_file["data"].append(curr_suite)
        files_list.append(curr_file)
    return files_list

def preds_to_log(files_list, config, threshold):
    # run = neptune.init_run(project="<PROJECT>", api_token=os.getenv('NEPTUNE_API_TOKEN'))
    # run["model/parameters"] = model_config
    # run["algorithm"] = config.experiment
    # run["threshold"] = threshold

    for file_dict in files_list:
        # run["val/epoch"].log(file_dict["epoch"])
        labels_data, preds_data, curr_preds = torch.Tensor([]), torch.Tensor([]), torch.Tensor([])
        for suite in file_dict["data"]:
            for scores in suite["scores"]:
                predictions = (scores[:,1] > threshold).long()
                curr_preds = torch.cat([curr_preds, predictions.cpu()])
            
            label = torch.Tensor([suite["label"]])
            pred = torch.Tensor([1]) if torch.sum(curr_preds) > 0 else torch.Tensor([0])
            curr_preds = torch.Tensor([])
            labels_data = torch.cat([labels_data, label])
            preds_data = torch.cat([preds_data, pred])
        
        # precision = precision_score(labels_data, preds_data, pos_label=0)
        # recall = recall_score(labels_data, preds_data, pos_label=0)
        # f1 = f1_score(labels_data, preds_data, pos_label=0)
        # accuracy = accuracy_score(labels_data, preds_data)
        
        sur_precision = precision_score(labels_data, preds_data, pos_label=0)
        sur_recall = recall_score(labels_data, preds_data, pos_label=0)
        sur_f1 = f1_score(labels_data, preds_data, pos_label=0)
        kill_precision = precision_score(labels_data, preds_data, pos_label=1)
        kill_recall = recall_score(labels_data, preds_data, pos_label=1)
        kill_f1 = f1_score(labels_data, preds_data, pos_label=1)
        accuracy = accuracy_score(labels_data, preds_data)
        fpr, tpr, thresholds = metrics.roc_curve(labels_data, preds_data, pos_label=1)
        kill_auc = metrics.auc(fpr, tpr)
        fpr, tpr, thresholds = metrics.roc_curve(labels_data, preds_data, pos_label=0)
        sur_auc = metrics.auc(fpr, tpr)
        
        res_dict = {"threshold={}".format(threshold):{"val_kill_prec": kill_precision, "val_kill_recall": kill_recall, "val_kill_f1": kill_f1, "val_kill_auc": kill_auc, "val_sur_prec": sur_precision, "val_sur_recall": sur_recall, "val_sur_f1": sur_f1, "val_sur_auc": sur_auc, "val_accuracy": accuracy}}
        (config.result_output_path / config.experiment).mkdir(parents=True, exist_ok=True)
        with open(config.result_output_path / config.experiment / "result_{}.json".format(threshold),"w") as f:
            json.dump(res_dict, f)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument("--model_config", type=str, default=Macros.model_configs_dir / "pmt_baseline.json")
    parser.add_argument("--model_dir", type=str, default=Macros.defects4j_model_dir / "pmt_baseline")
    parser.add_argument("--file_preds", type=str, default="")
    parser.add_argument("--preds_output_path", type=str, default=Macros.results_dir / "eval_preds")
    parser.add_argument("--result_output_path", type=str, default=Macros.results_dir / "eval_results")

    parser.add_argument("--experiment", type=str, default="pmt_baseline_suite")
    # parser.add_argument("--num_saves_per_epoch", type=int, default=10)

    parser.add_argument("--is_trans", action="store_true")
    parser.add_argument("--is_cl", action="store_true")
    parser.add_argument("--validation_dir", type=str, default=Macros.defects4j_root_dir / "pmt_baseline_ordered_suite" / "val")
    parser.add_argument("--sent_vocab_path", type=str, default=Macros.defects4j_root_dir / "pmt_baseline_ordered_suite" / "vocab_method_name.pkl")
    parser.add_argument("--body_vocab_path", type=str, default=Macros.defects4j_root_dir / "pmt_baseline_ordered_suite" / "vocab_body.pkl")

    args = parser.parse_args()

    # with open(args.model_config, "r") as f:
    #     model_config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_scores = []
    if args.file_preds != "":
        with open(args.file_preds, "rb") as f:
            file_scores = pickle.load(f)
    else:
        file_scores = get_file_scores(args, device)
        (args.preds_output_path / args.experiment).mkdir(parents=True, exist_ok=True)
        with open(args.preds_output_path / args.experiment / "file_scores.pkl","wb") as f:
            pickle.dump(file_scores, f)


    thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]
    for threshold in thresholds:
        preds_to_log(file_scores, args, threshold)
