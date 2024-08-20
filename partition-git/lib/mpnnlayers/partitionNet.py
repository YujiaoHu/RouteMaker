import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from .graph_embedding_v2 import graph_embedding_v2 as graph_embedding
from .prob_decoder_v2 import attention_decoding


class mtsp(nn.Module):
    def __init__(self, hidden_size, etypes, anum, common_hop, global_hop):
        super(mtsp, self).__init__()
        self.anum = anum
        self.embedding = graph_embedding(hidden_size, etypes, anum=1,
                                         common_hop=common_hop,
                                         global_hop=global_hop)
        # self.decoder = prob_decoding(hidden_size[-1] * (anum + 1), anum)
        # self.rnn = decodingLSTM(hidden_size[-1], anum)

        self.decoder = attention_decoding(hidden_size[-1], anum)

    def normalize_pred(self, feature):
        # feature: batch , f
        return F.normalize(feature, p=2, dim=1)

    def attention_decoder_beam_search(self, nfeature, last_current, mask):
        batch_size = nfeature.size(0)
        seq_len = nfeature.size(3)
        logits = self.decoder.forward_for_beam_search(nfeature, last_current, mask)
        logits, mask = self.apply_mask_to_logits(logits, mask, None)
        probs = F.softmax(logits.view(batch_size, seq_len * self.anum), dim=1)
        return probs

    def attention_decoder(self, step, nfeature, maxsample, instance_num):
        logits = self.decoder(nfeature, step)  #logits:[batch, cnum-1, anum]
        batch_size, cnum, anum = logits.size()

        # sample
        partition = []
        if maxsample is True:
            partition.append(torch.argmax(logits, dim=2, keepdim=False))
        else:
            for _ in range(instance_num):
                partition.append(logits.view(batch_size * cnum, anum).multinomial(1).view(batch_size, cnum))
        samples = torch.stack(partition, dim=1)
        return logits, samples

    def forward(self, step, inputs, nn_idx, maxsample=False, instance_num=1):
        nfeature = self.embedding(inputs, nn_idx)
        # batch_size, fsize, cnum = inputs.size()
        # for c in range(10):
        #     for r in range(cnum):
        #         x = nfeature[0, :, c] - nfeature[0, :, r]
        #         print("c = {}, r = {}, diff = {}".format(c, r, torch.sum(torch.abs(x))))
        nfeature = nfeature.squeeze(4).squeeze(1)
        probs, samples = self.attention_decoder(step, nfeature, maxsample, instance_num)
        return probs, samples
