import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


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
        return np.argmax(self.q_row(s))

    def update(self, s, a, r, s_next, done):
        q_sa = self.q_value(s, a)

        if done:
            target = r
        else:
            target = r + self.gamma * np.max(self.q_row(s_next))

        td_error = target - q_sa

        u_old = self.U[s].copy()
        v_old = self.V[a].copy()

        self.U[s] += self.lr * td_error * v_old
        self.V[a] += self.lr * td_error * u_old

        return td_error


# =========================
# Environment
# =========================
env = gym.make("FrozenLake-v1", is_slippery=False)

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

# =========================
# Logging
# =========================
episodes = 50000

success_history = []
td_error_history = []
episode_rewards = []


def moving_average(x, w=200):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid")


# =========================
# Training
# =========================
for ep in range(episodes):

    state, _ = env.reset()
    done = False

    ep_reward = 0
    ep_td_errors = []

    while not done:

        action = agent.select_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        td_error = agent.update(
            state, action, reward, next_state, done
        )

        ep_td_errors.append(td_error)
        ep_reward += reward

        state = next_state

    episode_rewards.append(ep_reward)
    success_history.append(ep_reward)
    td_error_history.append(np.mean(ep_td_errors) if len(ep_td_errors) > 0 else 0)

    if (ep + 1) % 500 == 0:
        agent.epsilon *= 0.95
        print(f"Episode {ep+1}, epsilon={agent.epsilon:.3f}")


# =========================
# Final Q
# =========================

print(env.unwrapped.desc)

Q = agent.U @ agent.V.T

print("Final Q-table:\n", Q)


# =========================
# 1. Success curve
# =========================
plt.figure()
plt.plot(moving_average(success_history, 200))
plt.title("Success Rate (Moving Average)")
plt.xlabel("Episode")
plt.ylabel("Success")
plt.show()


# =========================
# 2. TD error curve
# =========================
plt.figure()
plt.plot(moving_average(td_error_history, 200))
plt.title("TD Error (Moving Average)")
plt.xlabel("Episode")
plt.ylabel("TD Error")
plt.show()


# =========================
# 3. Q heatmap
# =========================
plt.figure()
plt.imshow(Q, cmap="coolwarm")
plt.colorbar()
plt.title("Learned Q(s,a)")
plt.xlabel("Action")
plt.ylabel("State")
plt.show()


# =========================
# 4. Policy visualization (4x4 FrozenLake)
# =========================
policy = np.argmax(Q, axis=1).reshape(4, 4)

plt.figure(figsize=(4, 4))
plt.imshow(policy, cmap="tab10")

for i in range(4):
    for j in range(4):
        plt.text(
            j, i,
            str(policy[i,j]),
            ha="center",
            va="center",
            color="white",
        )

plt.title("Learned Policy")
plt.show()


# =========================
# 5. Arrow policy (print form)
# =========================
arrow = ["←", "↓", "→", "↑"]
policy_grid = np.array([arrow[a] for a in policy.flatten()]).reshape(4, 4)

print("\nPolicy (arrows):")
print(policy_grid)


