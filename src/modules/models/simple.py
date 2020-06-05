import torch
import torch.nn as nn
import torch.nn.functional as F

class SimPLeModel(nn.Module):

    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.rnn = nn.LSTMCell(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, xt, ht_ct):
        xt = F.relu(self.fc1(xt))
        ht, ct = self.rnn(xt, ht_ct)
        yt = self.fc2(ht)

        return yt, (ht, ct)

    def init_hidden(self, batch_size, device):
        ht = torch.zeros(batch_size, self.hidden_size).to(device)
        ct = torch.zeros(batch_size, self.hidden_size).to(device)
        return (ht, ct)