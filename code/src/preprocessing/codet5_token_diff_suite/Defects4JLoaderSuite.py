
import sys, os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from torch.utils.data import DataLoader
from preprocessing.codet5_token_diff_suite.CodeT5DatasetSuite import *

class Defects4JLoaderSuite(DataLoader):
    def __init__(self, dataset, batch_size):
        self.n_samples = len(dataset)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'pin_memory': True,
            'collate_fn': collate_fn_codet5,
            'shuffle': False,  # must be false to use sampler
        }
        super().__init__(**self.init_kwargs)