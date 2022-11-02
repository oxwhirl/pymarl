import torch as th
import torch.nn as nn
import torch.nn.functional as F

"""TODO:
This class contains a lot redundant code. It can simply inherit the RNNAgent
call its super in the init and simply override the fc2 layer. In the forward
call its super and reshape the q as required and return.
"""

class CentralRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CentralRNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions * args.central_action_embed)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        q = q.reshape(-1, self.args.n_actions, self.args.central_action_embed)
        return q, h