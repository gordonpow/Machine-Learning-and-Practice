import random
import numpy as np
from collections import deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from ddqn_mlp import DQNMLP


class ReplayBuffer:
    def __init__(self, max_size=50000):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            np.array(s),
            np.array(a),
            np.array(r),
            np.array(s2),
            np.array(d),
        )

    def __len__(self):
        return len(self.buffer)


class DDQNAgent:
    def __init__(self, action_size=7, lr=1e-4, gamma=0.99):
        self.action_size = action_size
        self.gamma = gamma

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.online_net = DQNMLP(action_size).to(self.device)
        self.target_net = DQNMLP(action_size).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.replay = ReplayBuffer()

        # epsilon-greedy 探索率
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995

    def select_action(self, state):
        """ ε-greedy """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        q = self.online_net(state_t)
        return int(torch.argmax(q).item())

    def update(self, batch_size):
        if len(self.replay) < batch_size:
            return

        # ---- sample batch ----
        state, action, reward, next_state, done = self.replay.sample(batch_size)

        state      = torch.tensor(state, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        action     = torch.tensor(action, dtype=torch.int64).to(self.device)
        reward     = torch.tensor(reward, dtype=torch.float32).to(self.device)
        done       = torch.tensor(done, dtype=torch.float32).to(self.device)

        # ---- Q(s,a) ----
        q_values = self.online_net(state)  # (B,7)
        q_a = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        # ---- Double DQN target ----
        next_q_online = self.online_net(next_state)
        next_actions = next_q_online.argmax(dim=1)

        next_q_target = self.target_net(next_state)
        next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

        target = reward + (1 - done) * self.gamma * next_q

        # ---- loss ----
        loss = torch.nn.functional.mse_loss(q_a, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ⭐⭐⭐【最重要：TensorBoard 會讀取這兩個變數！！！！】
        self.loss_value = loss.item()                    # <- Loss
        self.last_q_mean = q_values.mean().item()        # <- 平均 Q 值

        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
