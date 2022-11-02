import torch as th
import torch.nn as nn
import numpy as np


class QMixerCentralAtten(nn.Module):
    def __init__(self, args):
        super(QMixerCentralAtten, self).__init__()

        self.args = args

        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.input_dim = self.n_agents * self.args.central_action_embed
        self.embed_dim = args.central_mixing_embed_dim

        # assert self.embed_dim % self.n_agents == 0
        self.heads = self.embed_dim

        self.atten_layer = nn.Sequential(nn.Linear(self.state_dim, self.args.hypernet_embed),
                                         nn.ReLU(),
                                         nn.Linear(self.args.hypernet_embed, self.heads * self.args.n_agents))

        self.net = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.embed_dim, self.embed_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.embed_dim, 1))

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, self.args.central_action_embed, self.n_agents)

        atten_output = self.atten_layer(states)
        atten_output = atten_output.reshape(-1, self.n_agents, self.heads)
        atten_output = atten_output.softmax(dim=1)

        inputs = th.bmm(agent_qs, atten_output).reshape(-1, self.heads)

        advs = self.net(inputs)
        vs = self.V(states)

        y = advs + vs
        q_tot = y.view(bs, -1, 1)
        return q_tot