import math
import random

import gym
import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from ddpg_pytorch.memory import ReplayBuffer


class Actor(nn.Module):
    def __init__(self, features_n, actions_n, unit_nums=256):
        super().__init__()
        self.fc1 = nn.Linear(features_n, unit_nums)
        self.fc2 = nn.Linear(unit_nums, unit_nums)
        self.fc3 = nn.Linear(unit_nums, unit_nums)
        self.fc4 = nn.Linear(unit_nums, unit_nums)
        self.fc5 = nn.Linear(unit_nums, actions_n)
        self.layers = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weight()

    def forward(self, x):
        # x = utils.feature_normalize(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.tanh(self.fc5(x))
        return x

    def init_weight(self):
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, mode='fan_in')
            # nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.0001)


class Critic(nn.Module):
    def __init__(self, features_n, actions_n, unit_nums=256):
        super().__init__()
        self.fc1 = nn.Linear(features_n + actions_n, unit_nums)
        self.fc2 = nn.Linear(unit_nums, unit_nums)
        self.fc3 = nn.Linear(unit_nums, actions_n)
        self.nets = [self.fc1, self.fc2, self.fc3]
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Trainer():
    """DDPG Trainer.
    """
    def __init__(self, env: gym.Env, actor_lr=1e-4, critic_lr=1e-4,
                 capacity=int(1e4), batch_size=64, retrain=False,
                 load_dir='ddpg', train_threshold=10000):
        super().__init__()
        self.name = 'Trainer'
        self.features_n = env.observation_space.shape[0]
        self.actions_n = env.action_space.shape[0]
        self.action_space = env.action_space
        self.action_high = self.action_space.high
        self.action_low = self.action_space.low
        # nets
        self.actor = Actor(self.features_n, self.actions_n)
        self.actor_target = Actor(self.features_n, self.actions_n)
        for p in self.actor_target.parameters():
            p.requires_grad = False
        self.actor_optim = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic = Critic(self.features_n, self.actions_n)
        self.critic_target = Critic(self.features_n, self.actions_n)
        for p in self.critic_target.parameters():
            p.requires_grad = False
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)
        # when steps >= train_threshold, optimize model
        self.train_threshold = train_threshold
        self.steps = 0
        # retrain
        if retrain:
            self.load_weights(load_dir)
            self.steps = self.train_threshold
        # hard update
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
        # replay buffer
        self.replay_buffer = ReplayBuffer(capacity=capacity, batch_size=batch_size)
        # hyper params
        self.discount = 0.999
        self.tau = 0.0001
        self.act_noise = 0.1
        self.is_trainning = True

    def optimize(self):
        if self.steps < self.train_threshold:
            return
        if not self.replay_buffer.can_sample():
            return
        batch = self.replay_buffer.sample()
        s, a, r, d, s_ = [], [], [], [], []
        for trans in batch:
            s.append(trans[0])
            a.append(trans[1])
            r.append(trans[2])
            d.append(trans[3])
            s_.append(trans[4])
        s = torch.tensor(s, dtype=torch.float)
        a = torch.tensor(a, dtype=torch.float)
        r = torch.tensor(r, dtype=torch.float).unsqueeze(1)
        d = torch.tensor(d, dtype=torch.float).unsqueeze(1)
        s_ = torch.tensor(s_, dtype=torch.float)
        with torch.no_grad():
            next_a_target = self.actor_target(s_).type(torch.float)
            td_target = r + self.discount * (1 - d) * \
                        self.critic_target(torch.cat([s_, next_a_target], dim=1))
        self.critic.zero_grad()
        q = self.critic(torch.cat([s, a], dim=1))
        value_loss = nn.MSELoss()(td_target, q)
        # update critic net
        value_loss.backward()
        self.critic_optim.step()
        # policy loss
        self.actor.zero_grad()
        # set actor net's params not need grads to save computations
        # for p in self.critic.parameters():
        #     p.requires_grad = False
        policy_loss = self.critic(torch.cat((s, self.actor(s).type(torch.float)), dim=1))
        policy_loss = - policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()
        # reset actor net's params grad = True
        # for p in self.critic.parameters():
        #     p.requires_grad = True
        # soft update
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)

    def soft_update(self, target: nn.Module, source: nn.Module):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def hard_update(self, target: nn.Module, source: nn.Module):
        """Replace target params with source params.
        """
        target.load_state_dict(source.state_dict())

    def random_act(self):
        return self.action_space.sample()

    def act(self, s):
        self.steps += 1
        action = self.actor(torch.tensor(s, dtype=torch.float).unsqueeze(0))
        action = action.data.numpy()[0]
        sign = random.random()
        action += self.is_trainning * self.act_noise * (1 if sign > 0.5 else -1) *\
                  np.random.randn(self.actions_n)
        action = np.clip(action, self.action_low, self.action_high)
        print(action)
        return action

    def load_weights(self, output):
        print('..............Model updated!..............')
        if output is None: return
        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )

    def save_model(self, output):
        print('.................Model saved!..............')
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )
