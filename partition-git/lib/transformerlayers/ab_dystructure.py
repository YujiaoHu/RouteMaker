import torch
from torch import nn
import torch.nn.functional as F
from .attention import MultiHeadAttentionLayer, MultiHeadAttention
import math
from torch.distributions import Categorical


class Dy_attention_layer(nn.Module):
    def __init__(
            self,
            n_heads,
            hidden_dim,
            mask=0,
    ):
        super(Dy_attention_layer, self).__init__()

        assert 0 <= mask <= 8
        self.mask = mask

        self.agent_city_mp_embedder = MultiHeadAttentionLayer(
            n_heads=n_heads,
            embed_dim=hidden_dim,
        )

        self.agent_agent_mp_embedder = MultiHeadAttentionLayer(
            n_heads=n_heads,
            embed_dim=hidden_dim,
        )

        self.agent_embed_weight = nn.Parameter(torch.Tensor(2, hidden_dim))
        self.agent_embed_weight.data.uniform_(-10, 10)

        self.city_agent_mp_embedder = MultiHeadAttentionLayer(
            n_heads=n_heads,
            embed_dim=hidden_dim,
        )

        self.city_city_mp_embedder = MultiHeadAttentionLayer(
            n_heads=n_heads,
            embed_dim=hidden_dim,
        )

        self.city_embed_weight = nn.Parameter(torch.Tensor(2, hidden_dim))
        self.city_embed_weight.data.uniform_(-10, 10)

        # self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)


    def forward(self, af, cf):
        '''
        af: batch_size, agent_num, hidden_dim
        cf: batch_size, city_num, hidden_dim
        '''
        batch_size, agent_num, agent_fsize = af.size()
        batch_size, city_num, city_fsize = cf.size()

        if self.mask == 0:
            af_new_city = self.agent_city_mp_embedder([af, cf])  # batch_size, agent_num, hidden_dim
            af_new_agent = self.agent_agent_mp_embedder([af, af])
            af_new = (af_new_city.view(batch_size * agent_num, agent_fsize) * self.agent_embed_weight[0] +
                      af_new_agent.view(batch_size * agent_num, agent_fsize) * self.agent_embed_weight[1]
                      ).view(batch_size, agent_num, agent_fsize)

            cf_new_agent = self.city_agent_mp_embedder([cf, af])
            cf_new_city = self.city_city_mp_embedder([cf, cf])
            cf_new = (cf_new_city.view(batch_size * city_num, city_fsize) * self.city_embed_weight[0] +
                      cf_new_agent.view(batch_size * city_num, city_fsize) * self.city_embed_weight[1]
                      ).view(batch_size, city_num, city_fsize)
        if self.mask == 1:
            af_new_agent = self.agent_agent_mp_embedder([af, af])
            af_new = af_new_agent
            cf_new_agent = self.city_agent_mp_embedder([cf, af])
            cf_new_city = self.city_city_mp_embedder([cf, cf])
            cf_new = (cf_new_city.view(batch_size * city_num, city_fsize) * self.city_embed_weight[0] +
                      cf_new_agent.view(batch_size * city_num, city_fsize) * self.city_embed_weight[1]
                      ).view(batch_size, city_num, city_fsize)
        if self.mask == 2:
            af_new_city = self.agent_city_mp_embedder([af, cf])  # batch_size, agent_num, hidden_dim
            af_new = af_new_city
            cf_new_agent = self.city_agent_mp_embedder([cf, af])
            cf_new_city = self.city_city_mp_embedder([cf, cf])
            cf_new = (cf_new_city.view(batch_size * city_num, city_fsize) * self.city_embed_weight[0] +
                      cf_new_agent.view(batch_size * city_num, city_fsize) * self.city_embed_weight[1]
                      ).view(batch_size, city_num, city_fsize)
        if self.mask == 3:
            af_new_city = self.agent_city_mp_embedder([af, cf])  # batch_size, agent_num, hidden_dim
            af_new_agent = self.agent_agent_mp_embedder([af, af])
            af_new = (af_new_city.view(batch_size * agent_num, agent_fsize) * self.agent_embed_weight[0] +
                      af_new_agent.view(batch_size * agent_num, agent_fsize) * self.agent_embed_weight[1]
                      ).view(batch_size, agent_num, agent_fsize)

            cf_new_agent = self.city_agent_mp_embedder([cf, af])
            cf_new = cf_new_agent
        if self.mask == 4:
            af_new_city = self.agent_city_mp_embedder([af, cf])  # batch_size, agent_num, hidden_dim
            af_new_agent = self.agent_agent_mp_embedder([af, af])
            af_new = (af_new_city.view(batch_size * agent_num, agent_fsize) * self.agent_embed_weight[0] +
                      af_new_agent.view(batch_size * agent_num, agent_fsize) * self.agent_embed_weight[1]
                      ).view(batch_size, agent_num, agent_fsize)

            cf_new_city = self.city_city_mp_embedder([cf, cf])
            cf_new = cf_new_city
        if self.mask == 5:
            af_new_agent = self.agent_agent_mp_embedder([af, af])  # batch_size, agent_num, hidden_dim
            af_new = af_new_agent
            cf_new_agent = self.city_agent_mp_embedder([cf, af])
            cf_new = cf_new_agent
        if self.mask == 6:
            af_new_city = self.agent_city_mp_embedder([af, cf])  # batch_size, agent_num, hidden_dim
            af_new = af_new_city
            cf_new_city = self.city_city_mp_embedder([cf, cf])
            cf_new = cf_new_city
        if self.mask == 7:
            af_new_agent = self.agent_agent_mp_embedder([af, af])  # batch_size, agent_num, hidden_dim
            af_new = af_new_agent
            cf_new_city = self.city_city_mp_embedder([cf, cf])
            cf_new = cf_new_city
        if self.mask == 8:
            af_new_city = self.agent_city_mp_embedder([af, cf])  # batch_size, agent_num, hidden_dim
            af_new = af_new_city
            cf_new_agent = self.city_agent_mp_embedder([cf, af])
            cf_new = cf_new_agent
        return af_new, cf_new


class DyTransformerMSTP(nn.Module):
    def __init__(self,
                 city_input_dim,
                 agent_input_dim,
                 hidden_dim,
                 encode_layers=2,
                 tanh_clipping=10,
                 n_heads=8,
                 mask=1):
        super(DyTransformerMSTP, self).__init__()

        self.city_input_dim = city_input_dim
        self.agent_input_dim = agent_input_dim
        self.hidden_dim = hidden_dim

        self.encode_layers = encode_layers
        self.tanh_clipping = tanh_clipping

        stdv = 1. / math.sqrt(self.hidden_dim)

        self.city_init_embed = nn.Parameter(torch.Tensor(self.city_input_dim, hidden_dim))
        self.city_init_embed.data.uniform_(-stdv, stdv)
        self.agent_init_embed = nn.Parameter(torch.Tensor(self.agent_input_dim, hidden_dim))
        self.agent_init_embed.data.uniform_(-stdv, stdv)

        self.dy_layers = []
        for l in range(self.encode_layers):
            module = Dy_attention_layer(n_heads, hidden_dim, mask)
            self.add_module("dy_att_{}".format(l), module)
            self.dy_layers.append(module)

        self.norm_factor = 1 / math.sqrt(self.hidden_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.W_query.data.uniform_(-stdv, stdv)
        self.W_key = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.W_key.data.uniform_(-stdv, stdv)

    def forward(self, af, cf, maxsample=False, instance_num=10):
        '''
            af: batch_size, agent_num, agent_input_dim
            cf: batch_size, city_num, city_input_dim
        '''
        af_embed = torch.matmul(af, self.agent_init_embed)
        cf_embed = torch.matmul(cf, self.city_init_embed)

        for layer in range(self.encode_layers):
            af_embed, cf_embed = self.dy_layers[layer](af_embed, cf_embed)

        req_Q = torch.matmul(cf_embed, self.W_query)
        cnode_K = torch.matmul(af_embed, self.W_key)

        compatibility = torch.bmm(req_Q, cnode_K.transpose(1, 2))
        compatibility = torch.tanh(compatibility) * 10
        probs = F.softmax(compatibility, dim=-1)  # [batch, cnum, anum]

        # print("---------- probs = ", probs[0][:5])
        # batch_size, cnum, anum = probs.size()
        # x = torch.argmax(probs, dim=-1)
        # batch_assign = []
        # for a in range(anum):
        #     y = torch.sum(torch.eq(x, a), dim=1)
        #     batch_assign.append(y)
        # batch_assign = torch.stack(batch_assign, 1)
        # diff = torch.max(batch_assign, dim=1)[0] - torch.min(batch_assign, dim=1)[0]
        # print("---------------------- max ----------------batch_assign ", batch_assign[:5])
        # print("---------------------- max ----------------diff ", diff)
        batch_size, cnum, anum = probs.size()
        if maxsample is True:
            partitions = torch.argmax(probs, dim=2, keepdim=True)  # [batch, cnum, 1]
        else:
            partitions = (probs.view(batch_size * cnum, anum)
                          .multinomial(instance_num, replacement=True).view(batch_size, cnum, -1))

        return probs, partitions
