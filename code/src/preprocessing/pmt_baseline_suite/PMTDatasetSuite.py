import ast
import pickle
import numpy as np
import os

import pandas as pd
import torch
from torch.utils.data import Dataset
import random
from tqdm import tqdm

class PMTDatasetSuite(Dataset):

    def __init__(self, data_path, sent_vocab_path, body_vocab_path, max_sent_length=150):
        self.max_sent_length = max_sent_length
        with open(sent_vocab_path, "rb") as f:
            self.sent_vocab = pickle.load(f)
        self.sent_w2i_map = {w: i for i, w in enumerate(self.sent_vocab)}

        with open(body_vocab_path, "rb") as f:
            self.body_vocab = pickle.load(f)
        self.body_w2i_map = {w: i for i, w in enumerate(self.body_vocab)}

        self.mutants = self.process_data(data_path)

    def process_data(self, data_path):
        new_mutants = []
        for path in tqdm(os.listdir(data_path), desc="Loading data"):
            with open(os.path.join(data_path, path), "rb") as f:
                mutants = pickle.load(f)
                for suite in mutants:
                    new_suite = {"label": suite["label"], "mutants": []}
                    for mutant in suite["mutants"]:
                        sent1 = self.sent_w2i(mutant["test_method"])
                        sent2 = self.sent_w2i(mutant["source_method"])
                        body = self.body_w2i(mutant["line"])
                        before = self.body_w2i(mutant["before"])
                        after = self.body_w2i(mutant["after"])

                        sent1 = sent1[:self.max_sent_length]
                        sent2 = sent2[:self.max_sent_length]
                        body = body[:self.max_sent_length]

                        new_suite["mutants"].append([mutant["mut_no"], torch.LongTensor(sent1).unsqueeze(0), torch.LongTensor(sent2).unsqueeze(0), torch.LongTensor(body).unsqueeze(0), torch.LongTensor(before).unsqueeze(0), torch.LongTensor(after).unsqueeze(0), torch.FloatTensor(mutant["mutator"]).unsqueeze(0), torch.LongTensor(mutant["label"])])
                    new_mutants.append(new_suite)
        random.shuffle(new_mutants)
        return new_mutants

    def sent_w2i(self, l):
        new_l = []
        for x in l:
            if x in self.sent_w2i_map:
                new_l.append(self.sent_w2i_map[x])
            else:
                new_l.append(1) # <unk>

        return new_l

    def body_w2i(self, l):
        new_l = []
        for x in l:
            if x in self.body_w2i_map:
                new_l.append(self.body_w2i_map[x])
            else:
                new_l.append(1) # <unk>

        return new_l

    def __getitem__(self, index):
        suite = self.mutants[index]
        
        return suite

    def __len__(self):
        return len(self.mutants)

    @property
    def name_vocab_size(self):
        return len(self.sent_vocab)

    @property
    def body_vocab_size(self):
        return len(self.body_vocab)

    @property
    def num_classes(self):
        return 2


def collate_fn_major(batch):
    return batch