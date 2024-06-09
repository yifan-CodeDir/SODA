import ast
import pickle
import numpy as np
import os

import pandas as pd
import torch
from torch.utils.data import Dataset
import random
from tqdm import tqdm

class PMTDataset(Dataset):

    def __init__(self, data_path, sent_vocab_path, body_vocab_path, max_sent_length=150):
        self.max_sent_length = max_sent_length
        self.mutants = self.process_data(data_path)

        with open(sent_vocab_path, "rb") as f:
            self.sent_vocab = pickle.load(f)
        self.sent_w2i_map = {w: i for i, w in enumerate(self.sent_vocab)}

        with open(body_vocab_path, "rb") as f:
            self.body_vocab = pickle.load(f)
        self.body_w2i_map = {w: i for i, w in enumerate(self.body_vocab)}

        self.data_path = data_path

    def process_data(self, data_path):
        new_mutants = []
        for path in tqdm(os.listdir(data_path), desc="Loading data"):
            with open(os.path.join(data_path, path), "rb") as f:
                mutants = pickle.load(f)
                for mutant in mutants:
                    new_mutants.append([mutant["mut_no"], mutant["test_method"], mutant["source_method"], mutant["line"], mutant["before"], mutant["after"], mutant["mutator"], mutant["label"]])
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

    def transform(self, mut_no, test_method, src_method, line, before, after, mutator, label):
        sent1 = self.sent_w2i(test_method)
        sent2 = self.sent_w2i(src_method)
        body = self.body_w2i(line)
        before = self.body_w2i(before)
        after = self.body_w2i(after)

        sent1 = sent1[:self.max_sent_length]
        sent2 = sent2[:self.max_sent_length]
        body = body[:self.max_sent_length]

        return mut_no, sent1, sent2, body, before, after, mutator, label

    def __getitem__(self, index):
        mut_no, test_method, src_method, line, before, after, mutator, label = self.mutants[index]
        mutant_no, sent1, sent2, body, before, after, mutator, label = self.transform(mut_no, test_method, src_method, line, before, after, mutator, label)

        return mutant_no, sent1, sent2, body, before, after, mutator, label

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

    @property
    def num_mutator(self):
        # get a sample of mutator
        sample_file = os.listdir(self.data_path)[0]
        with open(os.path.join(self.data_path, sample_file), "rb") as f:
            mutants = pickle.load(f)
        return len(mutants[0]["mutator"])


def collate_fn_major(batch):
    mut_no, sents1, sents2, body, before, after, mutator, labels = list(zip(*batch))
    bsz = len(labels)

    batch_max_length = max([len(x) for x in sents1])
    sents1_tensor = torch.zeros((bsz, batch_max_length)).long()
    for i, sent in enumerate(sents1):
        sents1_tensor[i, :len(sent)] = torch.LongTensor(sent)

    batch_max_length = max([len(x) for x in sents2])
    sents2_tensor = torch.zeros((bsz, batch_max_length)).long()
    for i, sent in enumerate(sents2):
        sents2_tensor[i, :len(sent)] = torch.LongTensor(sent)

    batch_max_length = max([len(x) for x in body])
    body_tensor = torch.zeros((bsz, batch_max_length)).long()
    for i, b in enumerate(body):
        body_tensor[i, :len(b)] = torch.LongTensor(b)

    batch_max_length = max([len(x) for x in before])
    before_tensor = torch.zeros((bsz, batch_max_length)).long()
    for i, b in enumerate(before):
        before_tensor[i, :len(b)] = torch.LongTensor(b)

    batch_max_length = max([len(x) for x in after])
    after_tensor = torch.zeros((bsz, batch_max_length)).long()
    for i, b in enumerate(after):
        after_tensor[i, :len(b)] = torch.LongTensor(b)

    return torch.LongTensor(mut_no), sents1_tensor, sents2_tensor, body_tensor, before_tensor, after_tensor, torch.FloatTensor(mutator), torch.LongTensor(labels)
