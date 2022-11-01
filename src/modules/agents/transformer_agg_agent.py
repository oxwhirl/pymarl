import torch.nn as nn
import torch.nn.functional as F
import torch
import argparse


class TransformerAggregationAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(TransformerAggregationAgent, self).__init__()
        self.args = args
        self.transformer = Transformer(args.token_dim, args.emb, args.heads, args.depth, args.emb)
        self.q_linear = nn.Linear(args.emb, 6 + args.enemy_num)

    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(1, self.args.emb).cuda()

    def forward(self, inputs, hidden_state, task_enemy_num, task_ally_num):
        outputs, _ = self.transformer.forward(inputs, hidden_state, None)

        # last output for hidden state
        h = outputs[:,-1:,:]
        q_agg = torch.mean(outputs, 1)
        q = self.q_linear(q_agg)

        return q, h


class SelfAttention(nn.Module):
    def __init__(self, emb, heads=8, mask=False):

        super().__init__()

        self.emb = emb
        self.heads = heads
        self.mask = mask

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x, mask):

        b, t, e = x.size()
        h = self.heads
        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b * h, t, t)

        if self.mask:  # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        if mask is not None:
            dot = dot.masked_fill(mask == 0, -1e9)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)

class TransformerBlock(nn.Module):

    def __init__(self, emb, heads, mask, ff_hidden_mult=4, dropout=0.0):
        super().__init__()

        self.attention = SelfAttention(emb, heads=heads, mask=mask)
        self.mask = mask

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x_mask):
        x, mask = x_mask

        attended = self.attention(x, mask)

        x = self.norm1(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x, mask


class Transformer(nn.Module):

    def __init__(self, input_dim, emb, heads, depth, output_dim):
        super().__init__()

        self.num_tokens = output_dim

        self.token_embedding = nn.Linear(input_dim, emb)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, mask=False))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, output_dim)

    def forward(self, x, h, mask):

        tokens = self.token_embedding(x)
        tokens = torch.cat((tokens, h), 1)

        b, t, e = tokens.size()

        x, mask = self.tblocks((tokens, mask))

        x = self.toprobs(x.view(b * t, e)).view(b, t, self.num_tokens)

        return x, tokens

def mask_(matrices, maskval=0.0, mask_diagonal=True):

    b, h, w = matrices.size()
    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unit Testing')
    parser.add_argument('--token_dim', default='5', type=int)
    parser.add_argument('--emb', default='32', type=int)
    parser.add_argument('--heads', default='3', type=int)
    parser.add_argument('--depth', default='2', type=int)
    parser.add_argument('--ally_num', default='5', type=int)
    parser.add_argument('--enemy_num', default='5', type=int)
    parser.add_argument('--episode', default='20', type=int)
    args = parser.parse_args()


    # testing the agent
    agent = TransformerAggregationAgent(None, args).cuda()
    hidden_state = agent.init_hidden().cuda().expand(args.ally_num, 1, -1)
    tensor = torch.rand(args.ally_num, args.ally_num+args.enemy_num, args.token_dim).cuda()
    q_list = []
    for _ in range(args.episode):
        q, hidden_state = agent.forward(tensor, hidden_state, args.ally_num, args.enemy_num)
        q_list.append(q)