import torch
import torch.nn as nn
import torch.nn.functional as F

class PMTNLC(nn.Module):

    def __init__(self, num_classes, name_vocab_size, body_vocab_size, name_embed_dim, body_embed_dim,
                 gru_hidden_dim, gru_num_layers, att_dim, num_mutator, dropout):
        super(PMTNLC, self).__init__()

        self.embeddings1 = nn.Embedding(name_vocab_size, name_embed_dim)
        self.embeddings2 = nn.Embedding(body_vocab_size, body_embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(name_embed_dim, gru_hidden_dim, num_layers=gru_num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.gru_body = nn.GRU(body_embed_dim, gru_hidden_dim, num_layers=gru_num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = nn.Linear(2 * gru_hidden_dim, att_dim)
        self.attention_body = nn.Linear(2 * gru_hidden_dim, att_dim)

        # Word context vector (u_w) to take dot-product with
        self.context_vector = nn.Linear(att_dim, 1, bias=False)
        self.context_vector_body = nn.Linear(att_dim, 1, bias=False)

        # Comparison layer - NN
        self.compare_nn = nn.Linear(4 * gru_hidden_dim, 2 * gru_hidden_dim)
        # Comparison layer - NMT
        self.compare_nmt = nn.Bilinear(2 * gru_hidden_dim, 2 * gru_hidden_dim, 2 * gru_hidden_dim)
        # Comparison layer - Cos
        self.compare_cos = nn.CosineSimilarity(dim=1)

        # Comparison layer - NN
        self.mut_compare_nn = nn.Linear(4 * gru_hidden_dim, 2 * gru_hidden_dim)
        # Comparison layer - NMT
        self.mut_compare_nmt = nn.Bilinear(2 * gru_hidden_dim, 2 * gru_hidden_dim, 2 * gru_hidden_dim)
        # Comparison layer - Cos
        self.mut_compare_cos = nn.CosineSimilarity(dim=1)

        self._fc1 = nn.Linear(8 * gru_hidden_dim + 2, gru_hidden_dim)
        self._fc2 = nn.Linear(10 * gru_hidden_dim + 2 + num_mutator, gru_hidden_dim)

        # comp_vec1 = torch.cat([embed_vec_mul, embed_vec_sub, embed_vec_nn, embed_vec_nmt, embed_vec_sim], dim=1)
        # comp_vec2 = torch.cat([embed_vec_mul_mut, embed_vec_sub_mut, embed_vec_nn_mut, embed_vec_nmt_mut, embed_vec_sim_mut], dim=1)
        # embed_vec = torch.cat([sents1_vec, sents2_vec, body_vec, mutator], dim=1)
        # embed_vec = torch.cat([comp_vec1, body_vec, comp_vec2, mutator], dim=1)
        self.fc = nn.Linear(2 * gru_hidden_dim, num_classes)

    def init_embeddings(self, embeddings):
        self.embeddings.weight = nn.Parameter(embeddings)

    def freeze_embeddings(self, freeze=False):
        self.embeddings.weight.requires_grad = not freeze

    def forward(self, sents1, sents2, body, before, after, mutator):
        sents1 = self.embeddings1(sents1)
        sents2 = self.embeddings1(sents2)
        body = self.embeddings2(body)
        before = self.embeddings2(before)
        after = self.embeddings2(after)

        sents1 = self.dropout(sents1)
        sents2 = self.dropout(sents2)
        body = self.dropout(body)
        before = self.dropout(before)
        after = self.dropout(after)

        gru_outputs1, _ = self.gru(sents1)
        gru_outputs2, _ = self.gru(sents2)
        gru_outputs_body, _ = self.gru_body(body)
        gru_outputs_before, _ = self.gru_body(before)
        gru_outputs_after, _ = self.gru_body(after)

        # Word Attenton
        att1 = torch.tanh(self.attention(gru_outputs1))
        att1 = self.context_vector(att1).squeeze(1)
        val = att1.max()
        att1 = torch.exp(att1 - val)
        att1_weights = att1 / torch.sum(att1, dim=1, keepdim=True)

        att2 = torch.tanh(self.attention(gru_outputs2))
        att2 = self.context_vector(att2).squeeze(1)
        val = att2.max()
        att2 = torch.exp(att2 - val)
        att2_weights = att2 / torch.sum(att2, dim=1, keepdim=True)

        att_body = torch.tanh(self.attention_body(gru_outputs_body))
        att_body = self.context_vector_body(att_body).squeeze(1)
        val = att_body.max()
        att_body = torch.exp(att_body - val)
        att_body_weights = att_body / torch.sum(att_body, dim=1, keepdim=True)

        att_before = torch.tanh(self.attention_body(gru_outputs_before))
        att_before = self.context_vector_body(att_before).squeeze(1)
        val = att_before.max()
        att_before = torch.exp(att_before - val)
        att_before_weights = att_before / torch.sum(att_before, dim=1, keepdim=True)

        att_after = torch.tanh(self.attention_body(gru_outputs_after))
        att_after= self.context_vector_body(att_after).squeeze(1)
        val = att_after.max()
        att_after = torch.exp(att_after - val)
        att_after_weights = att_after / torch.sum(att_after, dim=1, keepdim=True)

        # print(sents1.shape) # 128, 150, 50
        # print(gru_outputs1.shape) # 128, 150, 200
        # print(att1.shape) # 128, 150, 1
        # print(att1_weights.shape) # 128, 150, 1

        # Compute sentence vectors
        sents1_vec = gru_outputs1 * att1_weights # 128, 150, 200
        sents1_vec = sents1_vec.sum(dim=1) # 128, 200

        sents2_vec = gru_outputs2 * att2_weights
        sents2_vec = sents2_vec.sum(dim=1)

        body_vec = gru_outputs_body * att_body_weights
        body_vec = body_vec.sum(dim=1)

        before_vec = gru_outputs_before * att_before_weights
        before_vec = before_vec.sum(dim=1)

        after_vec = gru_outputs_after * att_after_weights
        after_vec = after_vec.sum(dim=1)

        # Comparison layers
        embed_vec_mul = torch.mul(sents1_vec, sents2_vec)
        embed_vec_sub = torch.sub(sents1_vec, sents2_vec)
        embed_vec_nn = F.relu(self.compare_nn(torch.cat((sents1_vec, sents2_vec), dim=1)))
        embed_vec_nmt = F.relu(self.compare_nmt(sents1_vec, sents2_vec))

        euc_batch = []
        for i in range(sents1_vec.shape[0]):
            _euc = torch.dist(sents1_vec[i], sents2_vec[i], p=2)
            euc_batch.append(_euc)
        _euc = torch.stack(euc_batch, dim=0)
        _cos = self.compare_cos(sents1_vec, sents2_vec)
        embed_vec_sim = torch.stack([_euc, _cos], dim=1)

        #

        embed_vec_mul_mut = torch.mul(before_vec, after_vec)
        embed_vec_sub_mut = torch.sub(before_vec, after_vec)
        embed_vec_nn_mut = F.relu(self.mut_compare_nn(torch.cat((before_vec, after_vec), dim=1)))
        embed_vec_nmt_mut = F.relu(self.mut_compare_nmt(before_vec, after_vec))

        euc_batch_mut = []
        for i in range(before_vec.shape[0]):
            _euc = torch.dist(before_vec[i], after_vec[i], p=2)
            euc_batch_mut.append(_euc)
        _euc_mut = torch.stack(euc_batch_mut, dim=0)
        _cos_mut = self.mut_compare_cos(before_vec, after_vec)
        embed_vec_sim_mut = torch.stack([_euc_mut, _cos_mut], dim=1)

        comp_vec1 = torch.cat([embed_vec_mul, embed_vec_sub, embed_vec_nn, embed_vec_nmt, embed_vec_sim], dim=1)
        comp_vec2 = torch.cat([embed_vec_mul_mut, embed_vec_sub_mut, embed_vec_nn_mut, embed_vec_nmt_mut, embed_vec_sim_mut], dim=1)
        embed_fc1 = self._fc1(comp_vec1)
        embed_fc2_cat = torch.cat([comp_vec2, body_vec, mutator], dim=1)
        # print(embed_fc2_cat.shape)
        # print(comp_vec2.shape)
        # print(comp_vec1.shape)
        # print(mutator.shape)
        # print(body_vec.shape)
        embed_fc2 = self._fc2(embed_fc2_cat)
        # embed_vec = torch.cat([comp_vec1, body_vec, comp_vec2, mutator], dim=1)
        embed_vec = torch.cat([embed_fc1, embed_fc2], dim=1)
        scores = self.fc(embed_vec)
        return scores, att1_weights, att2_weights, att_body_weights
