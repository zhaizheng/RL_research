import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import deque
import random
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''
device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)
'''
# =====================================================
# Replay Buffer
# =====================================================

class ReplayBuffer:
    def __init__(self, size=100000):
        self.buffer = deque(maxlen=size)

    def add(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        s, a, r, s2, d = zip(*batch)

        return (
            torch.FloatTensor(np.array(s)).to(device),
            torch.FloatTensor(np.array(a)).to(device),
            torch.FloatTensor(np.array(r)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(s2)).to(device),
            torch.FloatTensor(np.array(d)).unsqueeze(1).to(device),
        )

    def __len__(self):
        return len(self.buffer)


# =====================================================
# Low-Rank Critic
# =====================================================

class LowRankQ(nn.Module):
    def __init__(self, s_dim, a_dim, rank=32):
        super().__init__()

        self.rank = rank

        self.phi = nn.Sequential(
            nn.Linear(s_dim, 128),
            nn.ReLU(),
            nn.Linear(128, rank)
        )

        self.psi = nn.Sequential(
            nn.Linear(a_dim, 128),
            nn.ReLU(),
            nn.Linear(128, rank)
        )

        self.residual = nn.Sequential(
            nn.Linear(s_dim + a_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, s, a):

        phi = self.phi(s)
        psi = self.psi(a)

        phi = F.normalize(phi, dim=-1)
        psi = F.normalize(psi, dim=-1)

        low_rank = self.scale * torch.sum(phi * psi, dim=-1, keepdim=True)

        residual = self.residual(torch.cat([s, a], dim=-1))

        return low_rank + residual


# =====================================================
# Actor
# =====================================================

class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, action_max):
        super().__init__()

        self.action_max = action_max

        self.net = nn.Sequential(
            nn.Linear(s_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, a_dim)
        )

    def forward(self, s):

        a = self.net(s)

        return self.action_max * torch.tanh(a)


# =====================================================
# TD3 Low-Rank Agent
# =====================================================

env = gym.make("Pendulum-v1")

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
print([s_dim,a_dim])

action_max = float(env.action_space.high[0])

actor = Actor(s_dim, a_dim, action_max).to(device)
actor_target = copy.deepcopy(actor)

q1 = LowRankQ(s_dim, a_dim, rank=128).to(device)
q2 = LowRankQ(s_dim, a_dim, rank=128).to(device)

q1_target = copy.deepcopy(q1)
q2_target = copy.deepcopy(q2)

opt_actor = optim.Adam(actor.parameters(), lr=1e-5)
opt_q1 = optim.Adam(q1.parameters(), lr=1e-3)
opt_q2 = optim.Adam(q2.parameters(), lr=1e-3)

buffer = ReplayBuffer()

gamma = 0.99
tau = 0.005

batch_size = 256

warmup_steps = 5000

policy_delay = 2

total_steps = 0

episodes = 150


# =====================================================
# Training
# =====================================================
reward_history = []

for ep in range(episodes):

    s, _ = env.reset()

    ep_reward = 0

    for t in range(200):

        total_steps += 1

        # -------------------------
        # exploration
        # -------------------------

        if total_steps < warmup_steps:

            a = env.action_space.sample()

        else:

            with torch.no_grad():

                s_t = torch.FloatTensor(s).unsqueeze(0).to(device)

                a = actor(s_t).cpu().numpy()[0]

                a += np.random.normal(
                    0,
                    0.1 * action_max,
                    size=a_dim
                )

                a = np.clip(
                    a,
                    -action_max,
                    action_max
                )

        s2, r, terminated, truncated, _ = env.step(a)

        done = terminated or truncated

        buffer.add(s, a, r, s2, float(done))

        s = s2
        ep_reward += r

        # -------------------------
        # update
        # -------------------------

        if len(buffer) < warmup_steps:
            continue

        states, actions, rewards, next_states, dones = \
            buffer.sample(batch_size)

        with torch.no_grad():

            next_actions = actor_target(next_states)

            noise = (
                torch.randn_like(next_actions) * 0.2
            ).clamp(-0.5, 0.5)

            next_actions = (
                next_actions + noise
            ).clamp(-action_max, action_max)

            target_q1 = q1_target(
                next_states,
                next_actions
            )

            target_q2 = q2_target(
                next_states,
                next_actions
            )

            target_q = torch.min(
                target_q1,
                target_q2
            )

            y = rewards + gamma * (1 - dones) * target_q

        # -------------------------
        # critic update
        # -------------------------

        q1_pred = q1(states, actions)
        q2_pred = q2(states, actions)

        loss_q1 = F.mse_loss(q1_pred, y)
        loss_q2 = F.mse_loss(q2_pred, y)

        opt_q1.zero_grad()
        loss_q1.backward()
        torch.nn.utils.clip_grad_norm_(
            q1.parameters(),
            5.0
        )
        opt_q1.step()

        opt_q2.zero_grad()
        loss_q2.backward()
        torch.nn.utils.clip_grad_norm_(
            q2.parameters(),
            5.0
        )
        opt_q2.step()

        # -------------------------
        # delayed actor update
        # -------------------------

        if total_steps % policy_delay == 0:

            actor_loss = -q1(
                states,
                actor(states)
            ).mean()

            opt_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                5.0
            )
            opt_actor.step()

            # ---------------------
            # soft update
            # ---------------------

            for p, p_t in zip(
                    actor.parameters(),
                    actor_target.parameters()):
                p_t.data.copy_(
                    tau * p.data +
                    (1 - tau) * p_t.data
                )

            for p, p_t in zip(
                    q1.parameters(),
                    q1_target.parameters()):
                p_t.data.copy_(
                    tau * p.data +
                    (1 - tau) * p_t.data
                )

            for p, p_t in zip(
                    q2.parameters(),
                    q2_target.parameters()):
                p_t.data.copy_(
                    tau * p.data +
                    (1 - tau) * p_t.data
                )

    print(
        f"Episode {ep:3d} | "
        f"Reward {ep_reward:.1f}"
    )
    reward_history.append(ep_reward)

env.close()

# =====================================================
# Visualization / Test
# =====================================================

import time

test_env = gym.make(
    "Pendulum-v1",
    render_mode="human"
)

s, _ = test_env.reset()

total_reward = 0

for t in range(20000):

    with torch.no_grad():

        s_t = torch.FloatTensor(s).unsqueeze(0).to(device)

        action = actor(s_t)

        action = action.cpu().numpy()[0]


    s, r, terminated, truncated, _ = test_env.step(action)

    total_reward += r

    time.sleep(0.2)   # 控制动画速度

    if terminated or truncated:
        break


print("Test reward:", total_reward)

test_env.close()