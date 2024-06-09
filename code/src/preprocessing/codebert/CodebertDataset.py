import os
import pickle
import torch
import random
from torch.utils.data import IterableDataset

class CodebertDataset(IterableDataset):
    MUTANTS_PER_FILE = 10_000

    def __init__(self, data_path):
        self.data_path = data_path
        self.length = self.get_length()

    def get_length(self):
        curr_len = (len(list(os.listdir(self.data_path)))-1)*self.MUTANTS_PER_FILE
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
                for mutant in mutants:
                    yield torch.LongTensor(mutant["embed"]), torch.LongTensor(mutant["mask"]), torch.LongTensor([mutant["index"]]), mutant["label"]
    
    def __len__(self):
        return self.length

    @property
    def num_classes(self):
        return 2

def collate_fn_codebert(batch):
    ids, mask, idx, labels = list(zip(*batch))
    return torch.stack(ids), torch.stack(mask), torch.stack(idx), torch.LongTensor(labels)
