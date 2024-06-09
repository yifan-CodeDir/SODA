
import sys, os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from torch.utils.data import DataLoader
# from preprocessing.codebert.CodebertDataset import *
from preprocessing.codet5.CodeT5Dataset import *

class Defects4JLoader(DataLoader):
    def __init__(self, dataset, batch_size):
        self.n_samples = len(dataset)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'pin_memory': True,
            'collate_fn': collate_fn_codebert,
            'shuffle': False,  # must be false to use sampler
            'drop_last': True
        }
        super().__init__(**self.init_kwargs)