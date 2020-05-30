import torch.nn as nn
import torch.nn.functional as F

class SimPLeStateModel(nn.Module):

    def __init__(self, hidden_size, state_size, action_size, reward_size=1, term_size=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.term_size = term_size

        input_size = state_size + action_size
        output_size = state_size + reward_size + term_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.rnn = nn.LSTMCell(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, xt, ht, ct):
        xt = F.relu(self.fc1(xt))
        ht, ct = self.rnn(xt, (ht, ct))
        yt = self.fc2(ht)

        return yt, ht, ct

