# decoding part

import torch
from torch.nn import functional as F
import math
import os


class attention_decoding(torch.nn.Module):
    def __init__(self, fsize, anum):
        super(attention_decoding, self).__init__()
        self.anum = anum

        self.glb_embedding = torch.nn.Linear(fsize * 4, fsize)
        self.city_embedding = torch.nn.Linear(fsize * 2, 3 * fsize)
        self.agent_embedding = torch.nn.Linear(fsize * 3, fsize)

        self.project_out = torch.nn.Linear(fsize, fsize)

        self.mean_glb_embedding = torch.nn.Linear(fsize * 4, fsize)

        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, nfeature, step):
        nfeature = nfeature.permute(0, 2, 1)
        batch_size, total_num, fsize = nfeature.size()

        cnum = total_num - 2 * self.anum

        glbfeature, _ = torch.max(nfeature, dim=1)
        city_glb, _ = torch.max(nfeature[:, :cnum, :], dim=1)
        agent_start_glb, _ = torch.max(nfeature[:, cnum:cnum + self.anum, :], dim=1)
        agent_end_glb, _ = torch.max(nfeature[:, cnum + self.anum:, :], dim=1)
        glbfeature = torch.cat([glbfeature, city_glb, agent_start_glb, agent_end_glb], dim=1)
        deglb = self.glb_embedding(glbfeature).view(batch_size, 1, fsize)  # [batch=1, fsize, 1]

        # free agent
        ave_glbfeature = torch.mean(nfeature, dim=1)
        ave_city_glb = torch.mean(nfeature[:, :cnum, :], dim=1)
        ave_agent_start_glb = torch.mean(nfeature[:, cnum:cnum + self.anum, :], dim=1)
        ave_agent_end_glb = torch.mean(nfeature[:, cnum + self.anum:, :], dim=1)
        ave_glbfeature = torch.cat([ave_glbfeature, ave_city_glb,
                                    ave_agent_start_glb, ave_agent_end_glb], dim=1)
        free_agent = self.glb_embedding(ave_glbfeature).view(batch_size, 1, fsize)  # [batch=1, fsize, 1]

        city_glb = deglb.repeat(1, cnum, 1)
        city_feature = torch.cat([city_glb, nfeature[:, :cnum, :]], dim=2)
        city_feature = city_feature.contiguous().view(batch_size * cnum, -1)
        x, y, z = self.city_embedding(city_feature).view(batch_size, cnum, -1).chunk(3, dim=2)

        agent_glb2 = deglb.repeat(1, self.anum, 1)
        agent_feature = torch.cat([agent_glb2,
                                   nfeature[:, cnum:cnum + self.anum, :],
                                   nfeature[:, cnum + self.anum:, :]], dim=2)
        agent_feature = agent_feature.contiguous().view(batch_size * self.anum, -1)
        af = self.agent_embedding(agent_feature).view(batch_size, self.anum, -1)

        if step > 0:
            af = torch.cat([af, free_agent], dim=1)

        ucj = torch.bmm(af, x.transpose(1, 2)) / math.sqrt(x.size(2))
        batch_size, anum, fsize = af.size()
        new_af = torch.bmm(F.softmax(ucj, dim=2), y).contiguous().view(batch_size * anum, fsize)
        new_af = self.project_out(new_af).view(batch_size, anum, -1)

        logits = torch.bmm(new_af, z.transpose(1, 2)) / math.sqrt(z.size(2))  # [batch, anum, cnum]
        logits = torch.tanh(logits) * 10

        logits = logits.permute(0, 2, 1)
        probs = self.softmax(logits)

        return probs


class attention_decoding_for_mtsp(torch.nn.Module):
    def __init__(self, fsize, anum):
        super(attention_decoding_for_mtsp, self).__init__()
        self.anum = anum
        self.decoder = []
        self.glb_fixed_Q = []
        self.node_fixed_K = []
        self.node_fixed_V = []
        self.node_fixed_logit_K = []
        for a in range(self.anum):
            glb_embedding = torch.nn.Linear(self.anum * fsize + fsize, fsize)
            node_embedding = torch.nn.Conv2d(fsize, 3 * fsize, 1, 1)
            last_current_embedding = torch.nn.Conv2d(2 * fsize, fsize, 1, 1)
            project_out = torch.nn.Linear(fsize, fsize)

            self.add_module('glb_embedding_{}'.format(a), glb_embedding)
            self.add_module('node_embedding_{}'.format(a), node_embedding)
            self.add_module('last_current_{}'.format(a), last_current_embedding)
            self.add_module('project_out_{}'.format(a), project_out)

            self.decoder.append({'glb_embedding_{}'.format(a): glb_embedding,
                                 'node_embedding_{}'.format(a): node_embedding,
                                 'last_current_{}'.format(a): last_current_embedding,
                                 'project_out_{}'.format(a): project_out})
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, nfeature):
        batch_size = nfeature.size(0)

        self.glb_fixed_Q = []
        self.node_fixed_K = []
        self.node_fixed_V = []
        self.node_fixed_logit_K = []

        glbfeature, _ = torch.max(nfeature, dim=3)
        glbfeature = glbfeature.squeeze(3)
        glbfeature = glbfeature.contiguous().view(batch_size, -1)

        # glbfeature concat with depot feature
        for a in range(self.anum):
            # nfeature: [batch, anum, f, n, 1]
            deglb = torch.cat([glbfeature, nfeature[:, a, :, 0, 0]], dim=1)
            self.glb_fixed_Q.append(self.decoder[a]['glb_embedding_{}'.format(a)](deglb))
            self.glb_fixed_Q[a] = self.glb_fixed_Q[a].unsqueeze(2)

            x, y, z = self.decoder[a]['node_embedding_{}'.format(a)](nfeature[:, a, :, 1:]).chunk(3, dim=1)
            self.node_fixed_K.append(x)
            self.node_fixed_V.append(y)
            self.node_fixed_logit_K.append(z)
            self.node_fixed_K[a] = self.node_fixed_K[a].squeeze(3)
            self.node_fixed_V[a] = self.node_fixed_V[a].squeeze(3)
            self.node_fixed_logit_K[a] = self.node_fixed_logit_K[a].squeeze(3)

        result = []
        for a in range(self.anum):
            context_Q = self.glb_fixed_Q[a]
            # print("context_Q = ", context_Q, "self.node_fixed_K[a]", self.node_fixed_K[a])
            ucj = torch.bmm(context_Q.transpose(1, 2), self.node_fixed_K[a]) / math.sqrt(self.node_fixed_K[a].size(1))
            # temp_mask = mask.clone().unsqueeze(1)
            # ucj[temp_mask] = -math.inf
            new_context = torch.bmm(F.softmax(ucj, dim=2), self.node_fixed_V[a].transpose(1, 2))
            new_context = self.decoder[a]['project_out_{}'.format(a)](new_context.squeeze(1)).unsqueeze(1)
            logits = torch.bmm(new_context, self.node_fixed_logit_K[a]) / math.sqrt(self.node_fixed_K[a].size(1))
            logits = logits.squeeze(1)  # logits: [batch, n]
            # print("att_decoingd part: before tanh", logits)
            logits = torch.tanh(logits) * 10
            result.append(logits)
        result = torch.stack(result, dim=2)
        result = self.softmax(result)
        return result
