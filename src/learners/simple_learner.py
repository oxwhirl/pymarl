import time
import torch
import torch.nn.functional as F
from modules.models.simple import SimPLeModel
import numpy as np
import random

class SimPLeLearner:
    def __init__(self, mac, scheme, logger, args):

        self.mac = mac
        self.params = list(mac.parameters())
        self.args = args
        self.logger = logger
        self.device = self.args.device

        self.action_size = args.n_actions * args.n_agents
        self.state_size = args.state_shape - self.action_size if args.env_args["state_last_action"] else args.state_shape
        self.reward_size = scheme["reward"]["vshape"][0]
        self.term_size = scheme["terminated"]["vshape"][0]

        # state model
        self.state_model_input_size = self.state_size + self.action_size
        self.state_model_output_size = self.state_size + self.reward_size + self.term_size
        self.state_model = SimPLeModel(self.state_model_input_size, self.state_model_output_size, args.state_model_hidden_dim)

        # observation model
        self.obs_model_input_size = self.state_size + self.term_size
        if args.obs_model_include_last_action:
            self.obs_model_input_size += self.action_size
        self.obs_model_output_size = self.args.n_agents * (scheme["avail_actions"]["vshape"][0] + scheme["obs"]["vshape"])
        self.obs_model = SimPLeModel(self.obs_model_input_size, self.obs_model_output_size, args.obs_model_hidden_dim)

    def train_test_split(self, indices, test_ratio=0.1, shuffle=True):

        if shuffle:
            random.shuffle(indices)

        n = len(indices)
        n_test = max(1, int(test_ratio * n))
        train_indices = range(n - n_test)
        test_indices = range(len(train_indices), n)

        return train_indices, test_indices

    def get_episode_vars(self, ep):

        # per-agent quantities
        obs = ep["obs"][:, :-1, :]  # observations
        aa = ep["avail_actions"][:, :-1, :].float()  # available actions
        action = ep["actions_onehot"][:, :-1, :]  # actions taken

        # flatten per-agent quantities
        nbatch, ntimesteps, _, _ = obs.size()
        obs = obs.view((nbatch, ntimesteps, -1))
        aa = aa.view((nbatch, ntimesteps, -1))
        action = action.view(nbatch, ntimesteps, -1)

        # state
        state = ep["state"][:, :-1, :]
        if self.args.env_args["state_last_action"]:
            state = state[:, :, :self.state_size]

        # reward
        reward = ep["reward"][:, :-1, :]

        # termination signal
        terminated = ep["terminated"][:, :-1].float()
        term_idx = torch.squeeze(terminated).max(0)[1].item()
        term_signal = torch.ones_like(terminated)
        term_signal[:, :term_idx, :] = 0

        # mask for active timesteps (except for term_signal which is always valid)
        mask = torch.ones_like(terminated)
        mask[:, term_idx + 1:, :] = 0

        obs *= mask
        aa *= mask
        action *= mask
        reward *= mask
        state *= mask

        return state, action, reward, term_signal, obs, aa, mask

    def get_batch(self, episodes, batch_size, use_mask=False):
        # TOOD: refactor to use list of ids passed to replay buffer
        bs = min(batch_size, len(episodes))
        batch = random.sample(episodes, bs)
        props = [torch.cat(t) for t in zip(*batch)]
        if use_mask:
            mask = props[-1]
            idx = int(mask.sum(1).max().item())
            props = [x[:, :idx, :] for x in props]
        return props

    def get_state_model_input_output(self, state, action, reward, term_signal, obs, aa, mask):
        # inputs and outputs are offset by 1 timestep for 1-step-ahead prediction
        s = state[:, :-1, :]
        a = action[:, :-1, :]
        y = torch.cat((state, reward, term_signal), dim=-1)
        y = y[:, 1:, :]
        return s, a, y

    def run_state_model(self, state, action):

        bs, steps, state_size = state.size()
        ht_ct = self.state_model.init_hidden(bs, self.device)
        yp = torch.zeros(bs, steps, self.state_model_output_size).to(self.device)

        st = state[:, 0, :]  # initial state
        for t in range(0, steps):
            at = action[:, t, :]
            xt = torch.cat((st, at), dim=-1)

            yt, ht_ct = self.state_model(xt, ht_ct)
            yp[:, t, :] = yt

            st = yt[:, :state_size]
        return yp

    def train_state_model(self, train_episodes, test_episodes):
        # model learning parameters
        lr = self.args.state_model_learning_rate
        grad_clip = self.args.state_model_grad_clip_norm
        batch_size = self.args.state_model_train_batch_size
        batch_size = min(batch_size, len(test_episodes))
        optimizer = torch.optim.Adam(self.state_model.parameters(), lr=lr)
        epochs = self.args.state_model_train_epochs
        log_epochs = self.args.state_model_train_log_epochs
        use_mask = False # learning a termination signal is easier with unmasked input
        train_err = []
        val_err = []

        # train state model
        for e in range(epochs):

            self.state_model.train()
            t_start = time.time()

            props = self.get_batch(train_episodes, batch_size, use_mask=use_mask)
            state, action, y = self.get_state_model_input_output(*props)
            yp = self.run_state_model(state.to(self.device), action.to(self.device))
            optimizer.zero_grad()
            loss = F.mse_loss(yp, y.to(self.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.state_model.parameters(), grad_clip)
            optimizer.step()

            train_err.append(loss.item())

            self.state_model.eval()
            with torch.no_grad():

                props = self.get_batch(test_episodes, batch_size, use_mask=use_mask)
                state, action, y = self.get_state_model_input_output(*props)
                yp = self.run_state_model(state.to(self.device), action.to(self.device))
                val_err.append(F.mse_loss(yp, y.to(self.device)).item())

            if (e + 1) % log_epochs == 0:
                # report epoch losses
                train_err = np.array(train_err)
                val_err = np.array(val_err)
                t_epoch = time.time() - t_start
                print(f"epoch: {e + 1:<3}   train loss: mean {train_err.mean():.5f}, std: {train_err.std():.5f}   val loss: mean {val_err.mean():.5f}, std: {val_err.std():.5f}   time: {t_epoch:.2f} s")
                train_err = []
                val_err = []
                # self.logger.console_logger.info(f"Model training epoch {i}")

    def get_obs_model_input_output(self, state, actions, reward, term_signal, obs, aa, mask):
        y = torch.cat((obs, aa), dim=-1)
        return state, actions, term_signal, y

    def run_obs_model(self, state, action, term_signal):
        bs, steps, state_size = state.size()
        ht_ct = self.obs_model.init_hidden(bs, self.device)
        yp = torch.zeros(bs, steps, self.obs_model_output_size).to(self.device)

        # add last action
        # actions are generated after the observation so the first one is null and the last one is omitted
        last_action = action[:, :-1, :].to(self.device)
        a0 = torch.zeros_like(last_action[:, 0, :]).to(self.device)
        a0 = torch.unsqueeze(a0, dim=1)
        last_action = torch.cat((a0, last_action), dim=1)

        for t in range(0, steps):

            st = state[:, 0, :]
            at = last_action[:, t, :]
            tt = term_signal[:, t, :]

            if self.args.obs_model_include_last_action:
                xt = torch.cat((st, at, tt), dim=-1)
            else:
                xt = torch.cat((st, tt), dim=-1)

            yt, ht_ct = self.obs_model(xt, ht_ct)
            yp[:, t, :] = yt

            st = yt[:, :state_size]
        return yp

    def train_obs_model(self, train_episodes, test_episodes):
        # observation model training
        print(f"Observation Model Training ...")

        # model learning parameters
        lr = self.args.obs_model_learning_rate
        grad_clip = self.args.obs_model_grad_clip_norm
        batch_size = self.args.obs_model_train_batch_size
        batch_size = min(batch_size, len(test_episodes))
        optimizer = torch.optim.Adam(self.obs_model.parameters(), lr=lr)
        epochs = self.args.obs_model_train_epochs
        log_epochs = self.args.obs_model_train_log_epochs
        use_mask = self.args.obs_model_use_mask
        optimizer = torch.optim.Adam(self.obs_model.parameters(), lr=lr)
        train_err = []
        val_err = []

        self.state_model.eval()
        for e in range(epochs):
            t_start = time.time()
            # use state model and real actions to generate synthetic episodes from real starts
            with torch.no_grad():
                props = self.get_batch(train_episodes, batch_size, use_mask=use_mask)
                r_state, action, term_signal, y = self.get_obs_model_input_output(*props)
                m_state = self.run_state_model(r_state.to(self.device), action.to(self.device))
                m_state = m_state[:, :-1, :r_state.size()[-1]]  # exclude reward and term_signal and final timestep

                # prepend first real state to model generated states
                s0 = torch.unsqueeze(r_state[:, 0, :], dim=1).to(self.device)
                m_state = torch.cat((s0, m_state), dim=1)

            # generate obs from states
            self.obs_model.train()
            yp = self.run_obs_model(m_state, action, term_signal.to(self.device))

            # train obs model
            optimizer.zero_grad()
            loss = F.mse_loss(yp, y.to(self.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.obs_model.parameters(), grad_clip)
            optimizer.step()
            train_err.append(loss.item())

            # validate obs model
            with torch.no_grad():
                props = self.get_batch(test_episodes, batch_size, use_mask=use_mask)
                r_state, action, term_signal, y = self.get_obs_model_input_output(*props)
                m_state = self.run_state_model(r_state.to(self.device), action.to(self.device))
                m_state = m_state[:, :-1, :r_state.size()[-1]]  # exclude reward and term_signal and final timestep

                # prepend first real state to model generated states
                s0 = torch.unsqueeze(r_state[:, 0, :], dim=1).to(self.device)
                m_state = torch.cat((s0, m_state), dim=1)

                # generate obs from states
                self.obs_model.eval()
                yp = self.run_obs_model(m_state, action, term_signal.to(self.device))
                val_err.append(F.mse_loss(yp, y.to(self.device)).item())

            if (e + 1) % log_epochs == 0:
                # report epoch losses
                train_err = np.array(train_err)
                val_err = np.array(val_err)
                t_epoch = time.time() - t_start
                print(
                    f"epoch: {e + 1:<3}   train loss: mean {train_err.mean():.5f}, std: {train_err.std():.5f}   val loss: mean {val_err.mean():.5f}, std: {val_err.std():.5f}   time: {t_epoch:.2f} s")
                train_err = []
                val_err = []

    def train(self, episode_buffer):

        print(f"Training with {episode_buffer.episodes_in_buffer} episodes")
        print(f"Model Training ...")

        # generate training and test episode indices
        indices = list(range(0, episode_buffer.episodes_in_buffer))
        train_indices, test_indices = self.train_test_split(indices, test_ratio=self.args.model_training_test_ratio, shuffle=True)

        # extract episodes
        train_episodes = [self.get_episode_vars(episode_buffer[i]) for i in train_indices]
        test_episodes = [self.get_episode_vars(episode_buffer[i]) for i in train_indices]

        self.train_state_model(train_episodes, test_episodes)
        self.train_obs_model(train_episodes, test_episodes)

    def generate_episodes(self, episode_buffer, n_rollouts):

        self.state_model.eval()
        self.obs_model.eval()

        # sample real starts from the replay buffer
        self.logger.console_logger.info("Generating model based episodes")
        batch_size = min(episode_buffer.episodes_in_buffer, self.args.model_rollout_batch_size)
        episodes = episode_buffer.sample(batch_size)

        # construct observation model input
        state = episodes["state"][:, 0, self.state_size]
        term_signal = episodes["terminated"][:, 0, :]

        # optionally append last action
        if self.args.obs_model_include_last_action:
            last_action = torch.zeros_like(episodes["actions_onehot"][:, 0, :].view(batch_size, 1, -1))
            x = torch.cat((state, last_action, term_signal), dim=-1)
        else:
            x = torch.cat((state, term_signal), dim=-1)

        # generate fist observation
        # TODO: check what device x is on
        obs = self.obs_model(x.to(self.device))






        # generate rollout from real start and current policy

    def cuda(self):
        self.state_model.cuda()
        self.obs_model.cuda()