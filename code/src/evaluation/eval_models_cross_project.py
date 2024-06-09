import os, sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

import argparse

import torch
import torch.optim as optim
import pickle
import json
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from preprocessing.codebert.CodebertDataset import CodebertDataset
from preprocessing.Defects4JLoader import Defects4JLoader
from preprocessing.pmt_baseline.PMTDataset import PMTDataset
from preprocessing.pmt_baseline.PMTDataLoader import PMTDataLoader

from Macros import Macros
from preprocessing import utils
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn import metrics
from accelerate import Accelerator

MODEL_CONFIG_DICT = {
    # "seshat_cross_project_pmt_baseline_fold4": {
    #     "type": "pmt_baseline",
    #     "data_name": "seshat_cross_project_pmt_baseline_fold4",
    #     "best_checkpoint": "epoch_4.pth.tar",
    # },
    # "cross_project_codebert_token_diff_fold5": {
    #     "type": "unary",
    #     "data_name": "cross_project_codebert_token_diff_fold5",
    #     "best_checkpoint": "best_model.pth.tar",
    # },
    "cross_project_soda_fold4": {
        "type": "unary",
        "data_name": "cross_project_soda_fold4",
        "learning": "cl",
        "best_checkpoint": "rep_epoch_9_cla_epoch_2.pth.tar",
    },
}

def compute_metrics(preds, labels):
    sur_precision = precision_score(labels, preds, pos_label=0)
    sur_recall = recall_score(labels, preds, pos_label=0)
    sur_f1 = f1_score(labels, preds, pos_label=0)
    kill_precision = precision_score(labels, preds, pos_label=1)
    kill_recall = recall_score(labels, preds, pos_label=1)
    kill_f1 = f1_score(labels, preds, pos_label=1)
    accuracy = accuracy_score(labels, preds)
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)
    kill_auc = metrics.auc(fpr, tpr)
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=0)
    sur_auc = metrics.auc(fpr, tpr)

    return kill_precision, kill_recall, kill_f1, kill_auc, sur_precision, sur_recall, sur_f1, sur_auc, accuracy

def eval_dataset_pmt(model, device, dataloader):
    model.eval()
    with torch.no_grad():
        labels_data, preds_data = torch.Tensor([]), torch.Tensor([])
        with tqdm(dataloader, unit="batch") as tepoch:
            for _, sents1, sents2, body, before, after, mutator, labels in tepoch:
                sents1 = sents1.to(device)
                sents2 = sents2.to(device)
                body = body.to(device)
                before = before.to(device)
                after = after.to(device)
                mutator = mutator.to(device)
                labels = labels.to(device)

                labels_data = torch.cat([labels_data, labels.cpu()])

                scores, _, _, _ = model(sents1, sents2, body, before, after, mutator)
                predictions = scores.max(dim=1)[1]
                preds_data = torch.cat([preds_data, predictions.cpu()])

    return labels_data, preds_data


def eval_dataset_cl_unary(model, device, dataloader):
    model.eval()
    with torch.no_grad():
        labels_data, preds_data = torch.Tensor([]), torch.Tensor([])
        with tqdm(dataloader, unit="batch") as tepoch:
            for batch_idx, (ids, mask, idx, labels) in enumerate(tepoch):
                ids = ids.to(device)
                mask = mask.to(device)
                labels = labels.to(device)
                idx = idx.to(device)

                labels_data = torch.cat([labels_data, labels.cpu()])

                # scores = model(ids, mask)
                scores, _ = model(ids, mask, training_classifier=True)
                predictions = scores.max(dim=1)[1]
                preds_data = torch.cat([preds_data, predictions.cpu()])
    
    return labels_data, preds_data

def eval_dataset_dual_cl_unary(model, device, dataloader):
    model.eval()
    with torch.no_grad():
        labels_data, preds_data = torch.Tensor([]), torch.Tensor([])
        with tqdm(dataloader, unit="batch") as tepoch:
            for batch_idx, (exe_features, inf_features, prop_features, det_features, ids, mask, idx, labels) in enumerate(tepoch):
                exe_features = exe_features.to(device)
                inf_features = inf_features.to(device)
                prop_features = prop_features.to(device)
                det_features = det_features.to(device)
                ids = ids.to(device)
                mask = mask.to(device)
                labels = labels.to(device)
                idx = idx.to(device)

                labels_data = torch.cat([labels_data, labels.cpu()])

                # scores = model(ids, mask)
                scores, _ = model(ids, mask, e_feat1=exe_features, i_feat1=inf_features, p_feat1=prop_features, d_feat1=det_features, training_classifier=True)
                predictions = scores.max(dim=1)[1]
                preds_data = torch.cat([preds_data, predictions.cpu()])
    
    return labels_data, preds_data

def eval_dataset_feature_unary(model, device, dataloader):
    model.eval()
    with torch.no_grad():
        labels_data, preds_data = torch.Tensor([]), torch.Tensor([])
        with tqdm(dataloader, unit="batch") as tepoch:
            for batch_idx, (exe_features, inf_features, prop_features, det_features, labels) in enumerate(tepoch):
                exe_features = exe_features.to(device)
                inf_features = inf_features.to(device)
                prop_features = prop_features.to(device)
                det_features = det_features.to(device)
                labels = labels.to(device)

                # labels_data = torch.cat([labels_data, labels.cpu()])

                scores = model(exe_features, inf_features, prop_features, det_features)
                predictions = scores.max(dim=1)[1]
                # preds_data = torch.cat([preds_data, predictions.cpu()])

                # collect labels and batch_predictions on one machine to evaluate
                # batch_labels, batch_predictions = self.config.accelerator.gather_for_metrics((labels, predictions))
                labels_data = torch.cat([labels_data, labels.cpu()])
                preds_data = torch.cat([preds_data, predictions.cpu()])

    return labels_data, preds_data

def eval_dataset_unary(model, device, dataloader):
    model.eval()
    with torch.no_grad():
        labels_data, preds_data = torch.Tensor([]), torch.Tensor([])
        with tqdm(dataloader, unit="batch") as tepoch:
            for batch_idx, (ids, mask, idx, labels) in enumerate(tepoch):
                ids = ids.to(device)
                mask = mask.to(device)
                labels = labels.to(device)
                idx = idx.to(device)

                labels_data = torch.cat([labels_data, labels.cpu()])

                # scores = model(ids, mask)
                scores = model(ids, mask)
                predictions = scores.max(dim=1)[1]
                preds_data = torch.cat([preds_data, predictions.cpu()])
    
    return labels_data, preds_data

def get_dataset(dir):
    # read all the files and record the features and the labels
    X = []
    y = []
    for file in os.listdir(dir):
        with open(os.path.join(dir, file), "rb") as f:
            mutants = pickle.load(f)
            for mutant in mutants:
                x = [np.int64(mutant["num_executed"])] + list(mutant["mutator"]) + mutant["mut_static_features"] + [mutant["num_assert_in_tm"]]
                X.append(x)
                y.append(mutant["label"])
    return np.array(X), np.array(y)

def compute_feature_unary_stats(model_dir, device, model_details, model_name):
    # get mean and std for standarization, NOTE: only use training data to calculate mean and std 
    data, label = get_dataset(Macros.data_dir / model_details["data_name"] / "train")
    # for doing Z-score standarization
    mean = np.mean(data, axis=0)  
    std = np.std(data, axis=0)

    validation_dataset = HADataset(Macros.data_dir / model_details["data_name"] / "val", mean, std)
    validation_dataloader = HADataLoader(validation_dataset, 100)
    test_dataset = HADataset(Macros.data_dir / model_details["data_name"] / "test", mean, std)
    test_dataloader = HADataLoader(test_dataset, 100)

    model_dict = torch.load(model_dir / model_name / model_details["best_checkpoint"])
    model = model_dict["model"].to(device)

    val_labels, val_preds = eval_dataset_feature_unary(model, device, validation_dataloader)
    test_labels, test_preds = eval_dataset_feature_unary(model, device, test_dataloader)

    val_kill_precision, val_kill_recall, val_kill_f1, val_kill_auc, val_sur_precision, val_sur_recall, val_sur_f1, val_sur_auc, val_accuracy = compute_metrics(val_preds, val_labels)
    test_kill_precision, test_kill_recall, test_kill_f1, test_kill_auc, test_sur_precision, test_sur_recall, test_sur_f1, test_sur_auc, test_accuracy = compute_metrics(test_preds, test_labels)

    return {"val_kill_prec": val_kill_precision, "val_kill_recall": val_kill_recall, "val_kill_f1": val_kill_f1, "val_kill_auc": val_kill_auc, "val_sur_prec": val_sur_precision, "val_sur_recall": val_sur_recall, "val_sur_f1": val_sur_f1, "val_sur_auc": val_sur_auc, "val_accuracy": val_accuracy,  
            "test_kill_prec": test_kill_precision, "test_kill_recall": test_kill_recall, "test_kill_f1": test_kill_f1, "test_kill_auc": test_kill_auc, "test_sur_prec": test_sur_precision, "test_sur_recall": test_sur_recall, "test_sur_f1": test_sur_f1, "test_sur_auc": test_sur_auc, "test_accuracy": test_accuracy}


def compute_unary_stats(model_dir, device, model_details, model_name):
    validation_dataset = CodebertDataset(Macros.data_dir / model_details["data_name"] / "val")
    validation_dataloader = Defects4JLoader(validation_dataset, 400)
    # test_dataset = CodebertDataset(Macros.data_dir / model_details["data_name"] / "test")
    test_dataset = CodebertDataset(Macros.data_dir / model_details["data_name"] / "test")
    test_dataloader = Defects4JLoader(test_dataset, 400)

    model_dict = torch.load(model_dir / model_name / model_details["best_checkpoint"])
    model = model_dict["model"].to(device)

    val_labels, val_preds = eval_dataset_unary(model, device, validation_dataloader)
    test_labels, test_preds = eval_dataset_unary(model, device, test_dataloader)

    val_kill_precision, val_kill_recall, val_kill_f1, val_kill_auc, val_sur_precision, val_sur_recall, val_sur_f1, val_sur_auc, val_accuracy = compute_metrics(val_preds, val_labels)
    test_kill_precision, test_kill_recall, test_kill_f1, test_kill_auc, test_sur_precision, test_sur_recall, test_sur_f1, test_sur_auc, test_accuracy = compute_metrics(test_preds, test_labels)

    return {"val_kill_prec": val_kill_precision, "val_kill_recall": val_kill_recall, "val_kill_f1": val_kill_f1, "val_kill_auc": val_kill_auc, "val_sur_prec": val_sur_precision, "val_sur_recall": val_sur_recall, "val_sur_f1": val_sur_f1, "val_sur_auc": val_sur_auc, "val_accuracy": val_accuracy,  
            "test_kill_prec": test_kill_precision, "test_kill_recall": test_kill_recall, "test_kill_f1": test_kill_f1, "test_kill_auc": test_kill_auc, "test_sur_prec": test_sur_precision, "test_sur_recall": test_sur_recall, "test_sur_f1": test_sur_f1, "test_sur_auc": test_sur_auc, "test_accuracy": test_accuracy}

def compute_unary_cl_stats(model_dir, device, model_details, model_name):
    validation_dataset = CodebertDataset(Macros.data_dir / model_details["data_name"] / "val")
    validation_dataloader = Defects4JLoader(validation_dataset, 400)
    # test_dataset = CodebertDataset(Macros.data_dir / model_details["data_name"] / "test")
    test_dataset = CodebertDataset(Macros.data_dir / model_details["data_name"] / "test")
    test_dataloader = Defects4JLoader(test_dataset, 400)

    model_dict = torch.load(model_dir / model_name / model_details["best_checkpoint"])
    model = model_dict["model"].to(device)

    # optimizer = optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    # model, optimizer, validation_dataloader, test_dataloader = args.accelerator.prepare(
    #     model, optimizer, validation_dataloader, test_dataloader
    # )
    val_labels, val_preds = eval_dataset_cl_unary(model, device, validation_dataloader)
    test_labels, test_preds = eval_dataset_cl_unary(model, device, test_dataloader)

    val_kill_precision, val_kill_recall, val_kill_f1, val_kill_auc, val_sur_precision, val_sur_recall, val_sur_f1, val_sur_auc, val_accuracy = compute_metrics(val_preds, val_labels)
    test_kill_precision, test_kill_recall, test_kill_f1, test_kill_auc, test_sur_precision, test_sur_recall, test_sur_f1, test_sur_auc, test_accuracy = compute_metrics(test_preds, test_labels)

    return {"val_kill_prec": val_kill_precision, "val_kill_recall": val_kill_recall, "val_kill_f1": val_kill_f1, "val_kill_auc": val_kill_auc, "val_sur_prec": val_sur_precision, "val_sur_recall": val_sur_recall, "val_sur_f1": val_sur_f1, "val_sur_auc": val_sur_auc, "val_accuracy": val_accuracy,  
            "test_kill_prec": test_kill_precision, "test_kill_recall": test_kill_recall, "test_kill_f1": test_kill_f1, "test_kill_auc": test_kill_auc, "test_sur_prec": test_sur_precision, "test_sur_recall": test_sur_recall, "test_sur_f1": test_sur_f1, "test_sur_auc": test_sur_auc, "test_accuracy": test_accuracy}

def compute_unary_dual_cl_stats(model_dir, device, model_details, model_name):
    validation_dataset = DualConDataset(Macros.data_dir / model_details["data_name"] / "val")
    validation_dataloader = DualConDataLoader(validation_dataset, 300)
    test_dataset = DualConDataset(Macros.data_dir / model_details["data_name"] / "test")
    test_dataloader = DualConDataLoader(test_dataset, 300)

    model_dict = torch.load(model_dir / model_name / model_details["best_checkpoint"])
    model = model_dict["model"].to(device)

    # optimizer = optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    # model, optimizer, validation_dataloader, test_dataloader = args.accelerator.prepare(
    #     model, optimizer, validation_dataloader, test_dataloader
    # )
    val_labels, val_preds = eval_dataset_dual_cl_unary(model, device, validation_dataloader)
    test_labels, test_preds = eval_dataset_dual_cl_unary(model, device, test_dataloader)

    val_kill_precision, val_kill_recall, val_kill_f1, val_kill_auc, val_sur_precision, val_sur_recall, val_sur_f1, val_sur_auc, val_accuracy = compute_metrics(val_preds, val_labels)
    test_kill_precision, test_kill_recall, test_kill_f1, test_kill_auc, test_sur_precision, test_sur_recall, test_sur_f1, test_sur_auc, test_accuracy = compute_metrics(test_preds, test_labels)

    return {"val_kill_prec": val_kill_precision, "val_kill_recall": val_kill_recall, "val_kill_f1": val_kill_f1, "val_kill_auc": val_kill_auc, "val_sur_prec": val_sur_precision, "val_sur_recall": val_sur_recall, "val_sur_f1": val_sur_f1, "val_sur_auc": val_sur_auc, "val_accuracy": val_accuracy,  
            "test_kill_prec": test_kill_precision, "test_kill_recall": test_kill_recall, "test_kill_f1": test_kill_f1, "test_kill_auc": test_kill_auc, "test_sur_prec": test_sur_precision, "test_sur_recall": test_sur_recall, "test_sur_f1": test_sur_f1, "test_sur_auc": test_sur_auc, "test_accuracy": test_accuracy}


def compute_pmt_stats(model_dir, device, model_details, model_name):
    base_path = Macros.data_dir / model_details["data_name"]
    validation_dataset = PMTDataset(base_path / "val",
                                    base_path / "vocab_method_name.pkl", base_path / "vocab_body.pkl", max_sent_length=150)
    validation_dataloader = PMTDataLoader(validation_dataset, 512)
    
    test_dataset = PMTDataset(base_path / "test",
                                base_path / "vocab_method_name.pkl", base_path / "vocab_body.pkl", max_sent_length=150)
    test_dataloader = PMTDataLoader(test_dataset, 512)

    model_dict = torch.load(model_dir / model_name / model_details["best_checkpoint"])
    model = model_dict["model"].to(device)

    val_labels, val_preds = eval_dataset_pmt(model, device, validation_dataloader)
    test_labels, test_preds = eval_dataset_pmt(model, device, test_dataloader)

    val_kill_precision, val_kill_recall, val_kill_f1, val_kill_auc, val_sur_precision, val_sur_recall, val_sur_f1, val_sur_auc, val_accuracy = compute_metrics(val_preds, val_labels)
    test_kill_precision, test_kill_recall, test_kill_f1, test_kill_auc, test_sur_precision, test_sur_recall, test_sur_f1, test_sur_auc, test_accuracy = compute_metrics(test_preds, test_labels)

    return {"val_kill_prec": val_kill_precision, "val_kill_recall": val_kill_recall, "val_kill_f1": val_kill_f1, "val_kill_auc": val_kill_auc, "val_sur_prec": val_sur_precision, "val_sur_recall": val_sur_recall, "val_sur_f1": val_sur_f1, "val_sur_auc": val_sur_auc, "val_accuracy": val_accuracy,  
            "test_kill_prec": test_kill_precision, "test_kill_recall": test_kill_recall, "test_kill_f1": test_kill_f1, "test_kill_auc": test_kill_auc, "test_sur_prec": test_sur_precision, "test_sur_recall": test_sur_recall, "test_sur_f1": test_sur_f1, "test_sur_auc": test_sur_auc, "test_accuracy": test_accuracy}

def build_model_datapoints(data_dir):
    data_pts = {}
    i = 0
    for file_ind, path in enumerate(os.listdir(data_dir)):
        with open(os.path.join(data_dir, path), "rb") as f:
            mutants = pickle.load(f)
            for mutant in mutants:
                actual_ind = file_ind * 10_000 + mutant["id"]
                if actual_ind not in data_pts:
                    data_pts[actual_ind] = {}
                data_pts[actual_ind][mutant["type"]] = (torch.LongTensor(mutant["embed"]), torch.LongTensor(mutant["mask"]), torch.LongTensor([mutant["index"]]), mutant["label"])
    return data_pts

def get_binary_preds(model, device, datapts):
    model.eval()
    preds_map = {}
    labels_map = {}
    with torch.no_grad():
        labels_data, preds_data = torch.Tensor([]), torch.Tensor([])
        for idx in tqdm(datapts):
            preds_map[idx] = {}
            labels_map[idx] = {}
            ids, mask, _, labels = datapts[idx]["orig"]
            mut_ids, mut_mask, _, mut_labels = datapts[idx]["mutated"]

            ids = ids.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            labels = torch.LongTensor([labels]).to(device)
            
            mut_ids = mut_ids.unsqueeze(0).to(device)
            mut_mask = mut_mask.unsqueeze(0).to(device)
            mut_labels = torch.LongTensor([mut_labels]).to(device)

            labels_map[idx]["orig"] = labels[0].cpu()
            labels_map[idx]["mutated"] = mut_labels[0].cpu()

            scores = model(ids, mask)
            scores_mutated = model(mut_ids, mut_mask)

            preds_map[idx]["orig"] = scores[0][1].cpu()
            preds_map[idx]["mutated"] = scores_mutated[0][1].cpu()
    
    return labels_map, preds_map

def compute_threshold_metrics(labels_map, preds_map, threshold):
    labels, preds = [], []
    for idx in labels_map:
        pred = 1 if (preds_map[idx]["mutated"] - preds_map[idx]["orig"]) >= threshold else 0
        preds.append(pred)
        labels.append(labels_map[idx]["mutated"])
    
    return preds, labels

def compute_binary_metrics(labels_map, preds_map, final_threshold=None):
    MIN_THRESHOLD = 0.01
    MAX_THRESHOLD = 1.0
    STEP = 0.01

    metric_map = {"normal": {}, "threshold": {}}

    labels_norm, preds_norm = [], []
    for idx in labels_map:
        pred = 1 if preds_map[idx]["mutated"] >= 0.5 else 0
        preds_norm.append(pred)
        labels_norm.append(labels_map[idx]["mutated"])
    
    prec, recall, f1, acc = compute_metrics(preds_norm, labels_norm)
    metric_map["normal"] = {"prec": prec, "recall": recall, "f1": f1, "acc": acc}

    if final_threshold is None:
        max_f1 = 0
        final_threshold = 0
        for threshold in np.arange(MIN_THRESHOLD, MAX_THRESHOLD, STEP):
            preds, labels = compute_threshold_metrics(labels_map, preds_map, threshold)
            curr_prec, curr_recall, curr_f1, curr_acc = compute_metrics(preds, labels)
            if curr_f1 > max_f1:
                max_f1 = curr_f1
                final_threshold = threshold
    
    preds, labels = compute_threshold_metrics(labels_map, preds_map, final_threshold)
    curr_prec, curr_recall, curr_f1, curr_acc = compute_metrics(preds, labels)
    metric_map["threshold"] = {"prec": curr_prec, "recall": curr_recall, "f1": curr_f1, "acc": curr_acc, "threshold": final_threshold}
    print(metric_map)
    return metric_map, final_threshold

def compute_binary_stats(model_dir, device, model_details):
    model_dict = torch.load(model_dir / model_details["best_checkpoint"])
    model = model_dict["model"].to(device)

    val_datapts = build_model_datapoints(Macros.data_dir / model_details["data_name"] / "val")
    test_datapts = build_model_datapoints(Macros.data_dir / model_details["data_name"] / "test")

    val_label_map, val_pred_map = get_binary_preds(model, device, val_datapts)
    test_label_map, test_pred_map  = get_binary_preds(model, device, test_datapts)

    final_map = {}
    metric_map_val, final_threshold = compute_binary_metrics(val_label_map, val_pred_map)
    metric_map_test, _ = compute_binary_metrics(test_label_map, test_pred_map, final_threshold)
    final_map["val"] = metric_map_val
    final_map["test"] = metric_map_test
    return final_map



def eval_all_models(config, device):
    model_dir = Path(config.models_dir)
    aggregate_stats = {}
    for model in MODEL_CONFIG_DICT:
        aggregate_stats = {}
        model_details = MODEL_CONFIG_DICT[model]

        if model_details["type"] == "unary":
            if ("learning" in model_details) and model_details["learning"] == "cl":
                aggregate_stats[model] = compute_unary_cl_stats(model_dir, device, model_details, model)
                # if config.accelerator.is_main_process:
                #     with open("results.json", "w") as f:
                #         json.dump(aggregate_stats, f)
                # return
            elif ("learning" in model_details) and model_details["learning"] == "feature":
                aggregate_stats[model] = compute_feature_unary_stats(model_dir, device, model_details, model)
            elif ("learning" in model_details) and model_details["learning"] == "dual_cl":
                aggregate_stats[model] = compute_unary_dual_cl_stats(model_dir, device, model_details, model)
            else:
                aggregate_stats[model] = compute_unary_stats(model_dir, device, model_details, model)

        if model_details["type"] == "binary":
            aggregate_stats[model] = compute_binary_stats(model_dir, device, model_details)

        if model_details["type"] == "pmt_baseline":
            aggregate_stats[model] = compute_pmt_stats(model_dir, device, model_details, model)

        (Macros.eval_dir / f"{model}").mkdir(parents=True, exist_ok=True)
        with open(f"./{model}/results.json", "w") as f:
            json.dump(aggregate_stats, f)


            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--models_dir", type=str, default=Macros.model_dir)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # accelerator = Accelerator()
    # device = accelerator.device
    # args.accelerator = accelerator

    utils.set_seed(Macros.random_seed)

    # projects = ["Lang", "Chart", "Gson", "JacksonCore", "Csv", "Cli"]
    # projects = ["Chart"]

    # if not os.path.exists(os.path.dirname(model_save_dir)):
    # for project in projects:
    eval_all_models(args, device)


