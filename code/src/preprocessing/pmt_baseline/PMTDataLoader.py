import os, sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

from torch.utils.data import DataLoader
from preprocessing.pmt_baseline.PMTDataset import *

class PMTDataLoader(DataLoader):
    def __init__(self, dataset, batch_size):
        self.n_samples = len(dataset)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'pin_memory': True,
            'collate_fn': collate_fn_major,
        }
        super().__init__(**self.init_kwargs)