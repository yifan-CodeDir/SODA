import os
import pickle
import torch
import random
from torch.utils.data import IterableDataset

class CodebertDatasetSuite(IterableDataset):
    SUITES_PER_FILE = 200 

    def __init__(self, data_path):
        self.data_path = data_path
        self.length = self.get_length()

    def get_length(self):
        curr_len = (len(list(os.listdir(self.data_path)))-1)*self.SUITES_PER_FILE
        largest_suffix = -1
        largest_path = ""
        for path in os.listdir(self.data_path):
            if int(path.split("_")[1]) > largest_suffix:
                largest_suffix = int(path.split("_")[1])
                largest_path = path

        with open(os.path.join(self.data_path, largest_path), "rb") as f:
            mutants = pickle.load(f)
            curr_len += len(mutants)

        return curr_len

    def __iter__(self):
        directories = os.listdir(self.data_path)
        random.shuffle(directories)
        for path in directories:
            with open(os.path.join(self.data_path, path), "rb") as f:
                mutants = pickle.load(f)
                for suite in mutants:
                    yield suite
    
    def __len__(self):
        return self.length

    @property
    def num_classes(self):
        return 2

def collate_fn_codebert(batch):
    return batch
