import time
import torch
import torch.nn.functional as F
from modules.models.simple import SimPLeStateModel
import numpy as np
import random

class SimPLeLearner:
    def __init__(self, mac, scheme, logger, args):

        self.mac = mac
        self.params = list(mac.parameters())
        self.args = args
        self.logger = logger

        # model takes current state and joint-action and predicts next-state, reward and termination
        self.action_size = args.n_actions * args.n_agents
        self.state_size = args.state_shape - self.action_size if args.env_args["state_last_action"] else args.state_shape
        self.reward_size = scheme["reward"]["vshape"][0]
        self.term_size = scheme["terminated"]["vshape"][0]

        self.hidden_size = args.model_hidden_dim
        self.state_model = SimPLeStateModel(self.hidden_size, self.state_size, self.action_size,
                                            reward_size=self.reward_size, term_size=self.term_size)

    def train_test_split(self, indices, test_ratio=0.1, shuffle=True):

        if shuffle:
            random.shuffle(indices)

        n = len(indices)
        n_test = max(1, int(test_ratio * n))
        train_indices = range(n - n_test)
        test_indices = range(len(train_indices), n)

        return train_indices, test_indices

    def get_episode_state(self, ep):

        state = ep["state"][:, :-1]
        reward = ep["reward"][:, :-1]

        # termination signal
        terminated = ep["terminated"][:, :-1].float()
        term_idx = torch.squeeze(terminated).max(0)[1].item()
        term_signal = torch.ones_like(terminated)
        term_signal[:, :term_idx, :] = 0

        # mask for active timesteps
        mask = torch.ones_like(terminated)
        mask[:, term_idx + 1:, :] = 0

        if self.args.env_args["state_last_action"]:
            x = state  # input state includes joint-action
            y = state[:, :, :self.state_size]  # output state excludes joint-action
        else:
            x = state  # state does not include joint-action
            y = state

            # append joint-action to input
            actions = ep["actions_onehot"][:, :-1]
            n_episodes, n_timesteps, _ = state.size()
            actions = actions.view(n_episodes, n_timesteps, -1)
            x = torch.cat((state, actions), -1)

        # append reward and termination signal to output
        y = torch.cat((y, reward, term_signal), -1)

        # mask input and output by valid timesteps
        x *= mask
        y[:, :, :-1] *= mask  # ignore masking term_signal

        # slice to get input at time t and supervised signal at t+1
        x = x[:, :-1, :]  # skip the last timestep
        y = y[:, 1:, :]  # skip the first timestep

        return x, y, mask

    def generate_batch(self, episodes, batch_size, mask=False):
        batch = random.sample(episodes, min(batch_size, len(episodes)))
        x, y, m = (torch.cat(t) for t in zip(*batch))
        if mask:
            mask_index = int(torch.sum(m, 1).max().item())  # exclude post-terminal timesteps
            x = x[:, :mask_index, :]
            y = y[:, :mask_index, :]
        return x.to(self.args.device), y.to(self.args.device)

    def train(self, episode_buffer):

        print(f"Starting model training with {episode_buffer.episodes_in_buffer} episodes")

        # generate training and test episode indices
        indices = list(range(0, episode_buffer.episodes_in_buffer))
        train_indices, test_indices = self.train_test_split(indices, test_ratio=0.1, shuffle=True)

        # extract episodes
        train_episodes = [self.get_episode_state(episode_buffer[i]) for i in train_indices]
        test_episodes = [self.get_episode_state(episode_buffer[i]) for i in train_indices]

        # model learning parameters
        lr = self.args.model_learning_rate
        grad_clip = self.args.model_grad_clip_norm
        batch_size = self.args.model_train_batch_size
        batch_size = min(batch_size, len(test_episodes))
        output_size = self.state_size + self.reward_size + self.term_size
        optimizer = torch.optim.Adam(self.state_model.parameters(), lr=lr)
        epochs = self.args.model_train_epochs
        n_timesteps = self.args.episode_limit - 1
        log_epochs = self.args.model_train_log_epochs
        mask = False # learning a termination signal is easier with unmasked input

        train_err = []
        val_err = []
        for i in range(epochs):
            t_start = time.time()
            self.state_model.train()

            x, y = self.generate_batch(train_episodes, batch_size, mask=mask)

            bs = x.size()[0]
            a = x[:, :, self.state_size:]
            ht = torch.zeros(bs, self.hidden_size).to(self.args.device)
            ct = torch.zeros(bs, self.hidden_size).to(self.args.device)
            yp = torch.zeros(bs, n_timesteps, output_size).to(self.args.device)

            st = x[:, 0, :self.state_size]  # initial state
            for t in range(0, n_timesteps):
                at = a[:, t, :]  # joint-action
                xt = torch.cat((st, at), dim=-1)

                yt, ht, ct = self.state_model(xt, ht, ct)
                yp[:, t, :] = yt

                st = yt[:, :self.state_size]

            optimizer.zero_grad()   # zero the gradient buffers
            loss = F.mse_loss(yp, y)
            train_err.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.state_model.parameters(), grad_clip)
            optimizer.step()

            # validate
            with torch.no_grad():
                self.state_model.eval()
                x, y = self.generate_batch(test_episodes, batch_size, mask=False)

                bs = x.size()[0]
                a = x[:, :, self.state_size:]
                ht = torch.zeros(bs, self.hidden_size).to(self.args.device)
                ct = torch.zeros(bs, self.hidden_size).to(self.args.device)
                yp = torch.zeros(bs, n_timesteps, output_size).to(self.args.device)

                st = x[:, 0, :self.state_size]  # initial state
                for t in range(0, n_timesteps):
                    at = a[:, t, :]  # joint-action
                    xt = torch.cat((st, at), dim=-1)

                    yt, ht, ct = self.state_model(xt, ht, ct)
                    yp[:, t, :] = yt

                    st = yt[:, :self.state_size]

                val_loss = F.mse_loss(yp, y)
                val_err.append(val_loss.item())
                t_epoch = time.time() - t_start

            if (i+1) % log_epochs == 0:
                train_err = np.array(train_err)
                val_err = np.array(val_err)
                print(f"epoch: {i + 1:<3}   train loss: mean {train_err.mean():.5f}, std: {train_err.std():.5f}   val loss: mean {val_err.mean():.5f}, std: {val_err.std():.5f}   time: {t_epoch:.2f} s")
                train_err = []
                val_err = []

            # self.logger.console_logger.info(f"Model training epoch {i}")


    def generate_episodes(self):
        # fill a buffer using self.args.model_buffer_size
        self.logger.console_logger.info("Generating model based episodes")


    # def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
    #
    #     # Get the relevant quantities
    #     states = batch["state"][:, :-1]
    #     actions = batch["actions_onehot"][:, :-1]
    #     rewards = batch["reward"][:, :-1]
    #
    #     terminated = batch["terminated"][:, :-1].float()
    #     mask = batch["filled"][:, :-1].float()
    #     mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
    #
    #     # save tensors into temp dir
    #     import os
    #     tmp_dir = "tmp"
    #     if not os.path.exists(tmp_dir):
    #         os.mkdir("tmp")
    #     for t, n in [(states, "states"), (actions, "actions"), (rewards, "rewards"), (terminated, "terminated")
    #         , (mask, "mask")]:
    #         th.save(t, os.path.join(tmp_dir, f"{n}_{episode_num:04}.pt"))


    def cuda(self):
        self.state_model.cuda()