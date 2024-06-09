import logging
import math
from contextlib import contextmanager
from functools import partial, wraps
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint
# from einops import rearrange, repeat
from transformers import T5EncoderModel, AutoModel

class CLIP(nn.Module):
    def __init__(
            self,
            *,
            dim_text,
            args,
            num_classes=2,
            model_path=None,
            **kwargs
    ):
        super().__init__()
        # store some parameters for access

        self.args = args

        self.dim_text = dim_text

        # codet5 classification
        self.codeT5 = T5EncoderModel.from_pretrained(model_path, local_files_only=True)
        # self.codeT5_2 = T5EncoderModel.from_pretrained(model_path, local_files_only=True)
        # self.codeT5 = AutoModel.from_pretrained("microsoft/codebert-base")
        # temperature
        self.to_text_latent1 = nn.Linear(dim_text, dim_text)

        self.to_text_latent2 = nn.Linear(dim_text, dim_text)

        self.dense = nn.Linear(dim_text, 2)

        self.fc1 = nn.Linear(dim_text, dim_text)
        self.fc2 = nn.Linear(dim_text, dim_text)
        # self.fc2 = nn.Linear(dim_text, 2)
        self.fc3 = nn.Linear(2 * dim_text, num_classes)

    def forward(
            self,
            text1=None,
            mask1=None,
            text2=None,
            mask2=None,
            training_classifier=False,
    ):

        # ssl
        if training_classifier:
            enc_text1 = self.codeT5(input_ids=text1, attention_mask=mask1).last_hidden_state
            enc_text1 = enc_text1.mean(dim=1)
            # # related information
            x1 = F.relu(self.to_text_latent1(enc_text1))
            x1 = F.relu(self.to_text_latent2(x1))

            # classificaiton information
            # enc_text1 = self.codeT5(input_ids=text1, attention_mask=mask1).last_hidden_state
            # enc_text1 = enc_text.mean(dim=1)
            x2 = F.relu(self.fc1(enc_text1))
            x2 = F.relu(self.fc2(x2))
            
            return self.fc3(torch.cat([x1, x2], dim=-1)), torch.cat([x1, x2], dim=-1)

            # classificaiton information
            # x2 = self.fc1(enc_text)
            # return self.fc2(x2)

        enc_text1 = self.codeT5(input_ids=text1, attention_mask=mask1).last_hidden_state
        enc_text2 = self.codeT5(input_ids=text2, attention_mask=mask2).last_hidden_state

        CLS1 = enc_text1.mean(dim=1)
        CLS2 = enc_text2.mean(dim=1)

        CLS1 = F.relu(self.to_text_latent1(CLS1))
        CLS2 = F.relu(self.to_text_latent1(CLS2))

        CLS1 = F.relu(self.to_text_latent2(CLS1))
        CLS2 = F.relu(self.to_text_latent2(CLS2))

        return self.dense(CLS1), self.dense(CLS2)