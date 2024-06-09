import json
import pickle
import os, sys
from statistics import mean
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from tqdm import tqdm


os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import argparse

import torch
import torch.nn as nn

from preprocessing.pmt_baseline_suite.PMTDatasetSuite import *
from preprocessing.codebert_token_diff_suite.CodebertDatasetSuite import *
from preprocessing.codebert_token_diff_suite.Defects4JLoaderSuite import Defects4JLoaderSuite
from preprocessing.pmt_baseline_suite.PMTDataLoaderSuite import *
from Macros import Macros
from sklearn import metrics
import time
import math

MODEL_CONFIG_DICT = {
    # "pmt_baseline_cv_0.1": {
    #     "data_name": "cross_version_suite_pmt_baseline_ordered",
    #     "model_name": "cross_version_pmt_baseline",
    #     "best_checkpoint": "best_model.pth.tar",
    #     "is_trans": False,
    #     "is_cl": False,
    #     "threshold": 0.1
    # },
    # "codebert_token_diff_cv_0.1": {
    #     "data_name": "cross_version_suite_codebert_token_diff",
    #     "model_name": "cross_version_codebert_token_diff",
    #     "best_checkpoint": "best_model.pth.tar",
    #     "is_trans": True,
    #     "is_cl": False,
    #     "threshold": 0.1
    # },
    "soda_token_diff_cv_0.1": {
        "data_name": "cross_version_suite_soda",
        "model_name": "cross_version_soda",
        "best_checkpoint": "best_model.pth.tar",
        "is_trans": True,
        "is_cl": True,
        "threshold": 0.1
    },
}


def compute_scores(model_details, dataloader, device, proj_name):
    data = []
    model_dict = torch.load(f"{Macros.model_dir}/{model_details['model_name']}/{proj_name}/{model_details['best_checkpoint']}")
    model = model_dict["model"]
    model.eval()
    with torch.no_grad():
        labels_data, preds_data, curr_preds = torch.Tensor([]), torch.Tensor([]), torch.Tensor([])
        with tqdm(dataloader, unit="batch") as tepoch:
            for suite_batch in tepoch:
                for suite in suite_batch:
                    if model_details["is_trans"]:
                        if model_details["is_cl"]:
                            curr_suite = {"scores": [], "label": suite["label"]}
                            for mutant in suite["mutants"]:
                                ids = torch.LongTensor(mutant["embed"]).to(device)
                                mask = torch.LongTensor(mutant["mask"]).to(device)
                                scores, _ = model(ids.unsqueeze(0), mask.unsqueeze(0), training_classifier=True)
                                # scores = model(ids.unsqueeze(0), mask.unsqueeze(0))
                                curr_suite["scores"].append(scores)
                            data.append(curr_suite)
                        else:
                            curr_suite = {"scores": [], "label": suite["label"]}
                            for mutant in suite["mutants"]:
                                ids = torch.LongTensor(mutant["embed"]).to(device)
                                mask = torch.LongTensor(mutant["mask"]).to(device)
                                scores = model(ids.unsqueeze(0), mask.unsqueeze(0))
                                curr_suite["scores"].append(scores)
                            data.append(curr_suite)

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
                        data.append(curr_suite)
    return data

    
def get_file_scores(model_details, device, proj_name):
    if model_details["is_trans"]:
        val_dataset = CodebertDatasetSuite(Macros.data_dir / model_details["data_name"] / proj_name / "val")
        val_dataloader = Defects4JLoaderSuite(val_dataset, 1)
        test_dataset = CodebertDatasetSuite(Macros.data_dir / model_details["data_name"] / proj_name / "test")
        test_dataloader = Defects4JLoaderSuite(test_dataset, 1)

    else:
        val_dataset = PMTDatasetSuite(Macros.data_dir / model_details["data_name"] / proj_name / "val",
                                    Macros.data_dir / model_details["data_name"] / proj_name / "vocab_method_name.pkl", 
                                    Macros.data_dir / model_details["data_name"] / proj_name / "vocab_body.pkl", max_sent_length=150)
        val_dataloader = PMTDataLoaderSuite(val_dataset, 1)
        test_dataset = PMTDatasetSuite(Macros.data_dir / model_details["data_name"] / proj_name / "test",
                                    Macros.data_dir / model_details["data_name"] / proj_name / "vocab_method_name.pkl", 
                                    Macros.data_dir / model_details["data_name"] / proj_name / "vocab_body.pkl", max_sent_length=150)
        test_dataloader = PMTDataLoaderSuite(test_dataset, 1)

    # val_preds = compute_scores(model_details, val_dataloader, device, proj_name)
    start_time = time.time()
    test_preds = compute_scores(model_details, test_dataloader, device, proj_name)
    end_time = time.time()

    # v_sur_prec, v_sur_recall, v_sur_f1, v_sur_auc, v_kill_prec, v_kill_recall, v_kill_f1, v_kill_auc, v_acc, v_true_pos, v_predicted_pos, v_mut_num, v_mut_score = preds_to_metrics(val_preds, model_details["threshold"])
    t_sur_prec, t_sur_recall, t_sur_f1, t_sur_auc, t_kill_prec, t_kill_recall, t_kill_f1, t_kill_auc, t_acc, t_true_pos, t_predicted_pos, t_mut_num, t_mut_score = preds_to_metrics(test_preds, model_details["threshold"])
    return {
            # "val_sur_prec": v_sur_prec, "val_sur_recall": v_sur_recall, "val_sur_f1": v_sur_f1, "val_sur_auc": v_sur_auc, "val_kill_prec": v_kill_prec, "val_kill_recall": v_kill_recall, "val_kill_f1": v_kill_f1, "val_kill_auc": v_kill_auc, "val_accuracy": v_acc, "val_true_pos": v_true_pos, "val_pre_pos": v_predicted_pos, "val_mut_num": v_mut_num, "val_mut_score": v_mut_score,
            "test_sur_prec": t_sur_prec, "test_sur_recall": t_sur_recall, "test_sur_f1": t_sur_f1, "test_sur_auc": t_sur_auc, "test_kill_prec": t_kill_prec, "test_kill_recall": t_kill_recall, "test_kill_f1": t_kill_f1, "test_kill_auc": t_kill_auc, "test_accuracy": t_acc, "test_true_pos": t_true_pos, "test_pre_pos": t_predicted_pos, "test_mut_num": t_mut_num, "test_mut_score": t_mut_score, 
            "test_time": end_time - start_time}


def preds_to_metrics(file_dict, threshold):
    labels_data, preds_data, curr_preds = torch.Tensor([]), torch.Tensor([]), torch.Tensor([])
    for suite in file_dict:
        for scores in suite["scores"]:
            predictions = (scores[:,1] > threshold).long()
            curr_preds = torch.cat([curr_preds, predictions.cpu()])
        
        label = torch.Tensor([suite["label"]])
        pred = torch.Tensor([1]) if torch.sum(curr_preds) > 0 else torch.Tensor([0])
        curr_preds = torch.Tensor([])
        labels_data = torch.cat([labels_data, label])
        preds_data = torch.cat([preds_data, pred])

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

    # calculate mutation score
    true_pos = float(torch.sum(labels_data).item())
    predicted_pos = float(torch.sum(preds_data).item())
    mut_num = labels_data.size()[0]
    mut_score = abs(true_pos - predicted_pos) / mut_num

    return sur_precision, sur_recall, sur_f1, sur_auc, kill_precision, kill_recall, kill_f1, kill_auc, accuracy, true_pos, predicted_pos, mut_num, mut_score


def eval_all_models(device, proj_name):
    aggregate_stats = {}
    for model in MODEL_CONFIG_DICT:

        model_details = MODEL_CONFIG_DICT[model]

        aggregate_stats[model] = get_file_scores(model_details, device, proj_name)
    
        (Macros.eval_dir / f"{model}" / proj_name).mkdir(parents=True, exist_ok=True)
        with open(Macros.eval_dir / f"{model}"/ proj_name / "results_suite.json", "w") as f:
            json.dump(aggregate_stats, f)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # if not os.path.exists(os.path.dirname(model_save_dir)):

    # projects = ["Lang", "Chart", "Gson", "JacksonCore", "Csv", "Cli"]
    # projects = ["Lang", "Gson", "JacksonCore", "Csv", "Cli"]
    # projects = ["Chart", "Csv", "Cli"]
    projects = ["Gson", "Lang", "JacksonCore"]

    for project in projects:
        eval_all_models(device, project)

