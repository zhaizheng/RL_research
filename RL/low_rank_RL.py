import numpy as np


class LowRankQAgent:
    def __init__(
        self,
        n_states,
        n_actions,
        rank=8,
        gamma=0.99,
        lr=1e-2,
        epsilon=0.1,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.rank = rank
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon

        # Q(s,a)=U[s]·V[a]
        self.U = 0.1 * np.random.randn(n_states, rank)
        self.V = 0.1 * np.random.randn(n_actions, rank)

    def q_value(self, s, a):
        return np.dot(self.U[s], self.V[a])

    def q_row(self, s):
        return self.V @ self.U[s]

    def select_action(self, s):

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)

        q_values = self.q_row(s)
        return np.argmax(q_values)

    def update(self, s, a, r, s_next, done):

        q_sa = self.q_value(s, a)

        if done:
            target = r
        else:
            q_next = np.max(self.q_row(s_next))
            target = r + self.gamma * q_next

        td_error = target - q_sa

        u_old = self.U[s].copy()
        v_old = self.V[a].copy()

        # SGD on
        # L=(target-U_s^T V_a)^2

        self.U[s] += self.lr * td_error * v_old
        self.V[a] += self.lr * td_error * u_old

        return td_error

import gymnasium as gym

env = gym.make(
    "FrozenLake-v1",
    is_slippery=False
)

n_states = env.observation_space.n
n_actions = env.action_space.n

agent = LowRankQAgent(
    n_states=n_states,
    n_actions=n_actions,
    rank=4,
    lr=0.05,
    gamma=0.99,
    epsilon=0.1,
)

episodes = 50000

for ep in range(episodes):

    state, _ = env.reset()
    done = False

    while not done:

        action = agent.select_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        agent.update(
            state,
            action,
            reward,
            next_state,
            done
        )

        state = next_state

    if (ep + 1) % 500 == 0:

        agent.epsilon *= 0.95

        print(
            f"Episode {ep+1}, epsilon={agent.epsilon:.3f}"
        )


Q = agent.U @ agent.V.T

print(Q)