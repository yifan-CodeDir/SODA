import torch
import torch.nn as nn
import torch.nn.functional as F

"""
This model takes in the code or code/test around the mutated line to get an embedding for the mutated line

It also takes the the mutatation operator in addition to the line
"""
class PretrainedTrans(nn.Module):
    def __init__(self, trans, embedding_dim, num_classes):
        super(PretrainedTrans, self).__init__()
        self.trans = trans
        self.linear = nn.Linear(embedding_dim, num_classes)

    # Forward method, that takes method tokens and masks and the index of the line seperator token    
    def forward(self, tokens, mask):
        #embeds = self.trans.forward(tokens, mask)
        embeds = self.trans.forward(tokens, mask)[0][:, 0]
        # indices = class_idx.repeat(1, embeds.last_hidden_state.shape[2]).unsqueeze(1)
        # embeds = torch.gather(embeds.last_hidden_state, 1, indices).squeeze(1)
        output_preds = self.linear(embeds)
        return F.softmax(output_preds, dim=1)
