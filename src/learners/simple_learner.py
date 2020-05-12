import torch as th
from components.episode_buffer import EpisodeBatch
from modules.models import SimPLeModel
import numpy as np

class SimPLeLearner:
    def __init__(self, args):
        self.args = args

        # model takes current state and joint-action and predicts next-state and reward
        self.state_dim = int(np.prod(args.state_shape))
        self.joint_action_dim = args.n_actions * args.n_agents
        self.input_dim = self.state_dim + self.joint_action_dim
        self.output_dim = self.state_dim + 1 # reward dim is 1

        self.hidden_dim = 64 # TODO expose the model hidden_dim via args
        self.model = SimPLeModel(self.input_dim, self.output_dim, self.hidden_dim)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):

        # Get the relevant quantities
        states = batch["state"][:, :-1]
        actions = batch["actions"][:, :-1]
        rewards = batch["reward"][:, :-1]

        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

    def cuda(self):
        pass