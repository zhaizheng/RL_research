import os
import re
import random
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ============================================================
# Configuration
# ============================================================

SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MAX_TOTAL_STEPS = 500_000

START_TIMESTEPS = 5_000

BATCH_SIZE = 256

BUFFER_SIZE = 100_000

GAMMA = 0.99

TAU = 0.005

ACTOR_LR = 1e-3

CRITIC_LR = 1e-3

POLICY_NOISE = 0.2

NOISE_CLIP = 0.5

EXPLORATION_NOISE = 0.1

POLICY_DELAY = 2

SAVE_INTERVAL = 10_000

KEEP_CHECKPOINTS = 5

CHECKPOINT_DIR = Path("checkpoints")


# ============================================================
# Random seed
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Ball-Beam Environment
# ============================================================

class BallBeamEnv(gym.Env):

    metadata = {
        "render_modes": ["human"],
        "render_fps": 50,
    }

    def __init__(self, render_mode=None):

        super().__init__()

        self.render_mode = render_mode

        # ========================================================
        # Simulation
        # ========================================================

        self.dt = 0.02

        self.max_steps = 1000

        # ========================================================
        # Physical parameters
        # ========================================================

        self.g = 9.81

        # --------------------------------------------------------
        # Ball
        # --------------------------------------------------------

        self.ball_mass = 0.05

        self.ball_radius = 0.02

        # Solid sphere:
        #
        # J = 2/5 m r^2
        #

        self.ball_inertia = (
            2.0 / 5.0
            * self.ball_mass
            * self.ball_radius ** 2
        )

        # Effective translational mass for rolling ball:
        #
        # m_eff = m + J/r^2 = 7/5 m
        #

        self.ball_effective_mass = (
            self.ball_mass
            + self.ball_inertia
            / self.ball_radius ** 2
        )

        # --------------------------------------------------------
        # Beam
        # --------------------------------------------------------

        self.I = 0.01

        # --------------------------------------------------------
        # Damping
        # --------------------------------------------------------

        self.ball_damping = 0.01

        self.beam_damping = 0.05

        # --------------------------------------------------------
        # Optional beam stiffness
        # --------------------------------------------------------

        self.beam_stiffness = 0.0

        # ========================================================
        # Geometry
        # ========================================================

        self.beam_half_length = 2.0

        self.x_limit = self.beam_half_length

        self.theta_limit = np.deg2rad(35.0)

        # ========================================================
        # Observation space
        # ========================================================

        self.observation_space = spaces.Box(

            low=np.array(
                [
                    -2.5,
                    -15.0,
                    -np.pi / 2,
                    -25.0,
                ],
                dtype=np.float32,
            ),

            high=np.array(
                [
                    2.5,
                    15.0,
                    np.pi / 2,
                    25.0,
                ],
                dtype=np.float32,
            ),

            dtype=np.float32,
        )

        # ========================================================
        # Action space
        # ========================================================

        self.torque_limit = 0.5

        self.action_space = spaces.Box(

            low=np.array(
                [-self.torque_limit],
                dtype=np.float32,
            ),

            high=np.array(
                [self.torque_limit],
                dtype=np.float32,
            ),

            dtype=np.float32,
        )

        # ========================================================
        # State
        # ========================================================

        self.state = None

        self.step_count = 0

        # ========================================================
        # Rendering
        # ========================================================

        self.fig = None

        self.ax = None

    # ============================================================
    # Reset
    # ============================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.step_count = 0

        x0 = self.np_random.uniform(
            -0.8,
            0.8,
        )

        x_dot0 = self.np_random.uniform(
            -0.05,
            0.05,
        )

        theta0 = self.np_random.uniform(
            -0.10,
            0.10,
        )

        theta_dot0 = self.np_random.uniform(
            -0.05,
            0.05,
        )

        self.state = np.array(
            [
                x0,
                x_dot0,
                theta0,
                theta_dot0,
            ],
            dtype=np.float64,
        )

        if self.render_mode == "human":

            self.render()

        return (
            self.state.astype(np.float32).copy(),
            {},
        )

    # ============================================================
    # Continuous dynamics
    # ============================================================

    def dynamics(
        self,
        state,
        torque,
    ):

        x, x_dot, theta, theta_dot = state

        m = self.ball_mass

        m_eff = self.ball_effective_mass

        # --------------------------------------------------------
        # Ball dynamics
        #
        # m_eff x_ddot =
        #
        #   m x theta_dot^2
        #   - m g sin(theta)
        #   - c_x x_dot
        #
        # --------------------------------------------------------

        x_ddot = (

            m
            * x
            * theta_dot ** 2

            - m
            * self.g
            * np.sin(theta)

            - self.ball_damping
            * x_dot

        ) / m_eff

        # --------------------------------------------------------
        # Beam dynamics
        #
        # (I + m x^2) theta_ddot =
        #
        #   torque
        #   - c_theta theta_dot
        #   - k_theta theta
        #   - 2m x x_dot theta_dot
        #   - m g x cos(theta)
        #
        # --------------------------------------------------------

        effective_inertia = (

            self.I

            + m
            * x ** 2

        )

        theta_ddot = (

            torque

            - self.beam_damping
            * theta_dot

            - self.beam_stiffness
            * theta

            - 2.0
            * m
            * x
            * x_dot
            * theta_dot

            - m
            * self.g
            * x
            * np.cos(theta)

        ) / effective_inertia

        return np.array(
            [
                x_dot,
                x_ddot,
                theta_dot,
                theta_ddot,
            ],
            dtype=np.float64,
        )

    # ============================================================
    # RK4
    # ============================================================

    def _rk4_step(
        self,
        state,
        torque,
    ):

        dt = self.dt

        k1 = self.dynamics(
            state,
            torque,
        )

        k2 = self.dynamics(
            state
            + 0.5 * dt * k1,
            torque,
        )

        k3 = self.dynamics(
            state
            + 0.5 * dt * k2,
            torque,
        )

        k4 = self.dynamics(
            state
            + dt * k3,
            torque,
        )

        next_state = (

            state

            + dt
            * (
                k1
                + 2.0 * k2
                + 2.0 * k3
                + k4
            )
            / 6.0
        )

        return next_state

    # ============================================================
    # Step
    # ============================================================

    def step(self, action):

        state = self.state.copy()

        action = np.asarray(
            action,
            dtype=np.float64,
        ).reshape(-1)

        torque = float(
            np.clip(
                action[0],
                -self.torque_limit,
                self.torque_limit,
            )
        )

        # --------------------------------------------------------
        # Physics integration
        # --------------------------------------------------------

        next_state = self._rk4_step(
            state,
            torque,
        )

        x, x_dot, theta, theta_dot = next_state

        self.step_count += 1

        # --------------------------------------------------------
        # Termination
        # --------------------------------------------------------

        out_of_track = (
            abs(x)
            > self.x_limit
        )

        excessive_angle = (
            abs(theta)
            > self.theta_limit
        )

        terminated = (
            out_of_track
            or excessive_angle
        )

        truncated = (
            self.step_count
            >= self.max_steps
        )

        # --------------------------------------------------------
        # Only clip terminal observation.
        # Do NOT clip normal dynamics.
        # --------------------------------------------------------

        if terminated:

            next_state = np.clip(
                next_state,
                self.observation_space.low,
                self.observation_space.high,
            )

        self.state = next_state

        # ========================================================
        # Reward
        # ========================================================

        reward = -(
            2.0 * x ** 2

            + 0.10
            * x_dot ** 2

            + 0.50
            * theta ** 2

            + 0.01
            * theta_dot ** 2

            + 0.001
            * torque ** 2
        )

        # --------------------------------------------------------
        # Stabilization bonus
        # --------------------------------------------------------

        if (

            abs(x)
            < 0.05

            and abs(x_dot)
            < 0.10

            and abs(theta)
            < 0.05

            and abs(theta_dot)
            < 0.10

        ):

            reward += 5.0

        # --------------------------------------------------------
        # Terminal penalty
        # --------------------------------------------------------

        if out_of_track:

            reward -= 20.0

        if excessive_angle:

            reward -= 10.0

        # ========================================================
        # Info
        # ========================================================

        info = {

            "x": float(x),

            "x_dot": float(x_dot),

            "theta": float(theta),

            "theta_dot": float(theta_dot),

            "torque": float(torque),

        }

        if self.render_mode == "human":

            self.render()

        return (

            self.state
            .astype(np.float32)
            .copy(),

            float(reward),

            terminated,

            truncated,

            info,
        )

    # ============================================================
    # Render
    # ============================================================

    def render(self):

        if self.fig is None:

            self.fig, self.ax = plt.subplots(
                figsize=(7, 5),
            )

            plt.ion()

        x, _, theta, _ = self.state

        self.ax.clear()

        L = self.beam_half_length

        # Beam
        beam_x = np.array(
            [
                -L * np.cos(theta),
                L * np.cos(theta),
            ]
        )

        beam_y = np.array(
            [
                -L * np.sin(theta),
                L * np.sin(theta),
            ]
        )

        self.ax.plot(
            beam_x,
            beam_y,
            linewidth=5,
        )

        # Ball
        ball_x = (
            x * np.cos(theta)
        )

        ball_y = (
            x * np.sin(theta)
        )

        self.ax.scatter(
            [ball_x],
            [ball_y],
            s=200,
        )

        # Pivot
        self.ax.scatter(
            [0],
            [0],
            s=50,
        )

        # Beam endpoints
        self.ax.scatter(
            beam_x,
            beam_y,
            s=30,
        )

        self.ax.set_xlim(
            -2.5,
            2.5,
        )

        self.ax.set_ylim(
            -2.5,
            2.5,
        )

        self.ax.set_aspect(
            "equal"
        )

        self.ax.set_xlabel(
            "x"
        )

        self.ax.set_ylabel(
            "y"
        )

        self.ax.set_title(
            "Nonlinear Ball-Beam TD3"
        )

        self.ax.grid(
            True,
            alpha=0.3,
        )

        plt.pause(
            0.001
        )

    # ============================================================
    # Close
    # ============================================================

    def close(self):

        if self.fig is not None:

            plt.close(
                self.fig
            )

            self.fig = None

            self.ax = None


# ============================================================
# Replay Buffer
# ============================================================

class ReplayBuffer:

    def __init__(
        self,
        state_dim,
        action_dim,
        max_size,
    ):

        self.max_size = max_size

        self.ptr = 0

        self.size = 0

        self.states = np.zeros(
            (
                max_size,
                state_dim,
            ),
            dtype=np.float32,
        )

        self.actions = np.zeros(
            (
                max_size,
                action_dim,
            ),
            dtype=np.float32,
        )

        self.next_states = np.zeros(
            (
                max_size,
                state_dim,
            ),
            dtype=np.float32,
        )

        self.rewards = np.zeros(
            (
                max_size,
                1,
            ),
            dtype=np.float32,
        )

        self.dones = np.zeros(
            (
                max_size,
                1,
            ),
            dtype=np.float32,
        )

    # ========================================================
    # Add
    # ========================================================

    def add(
        self,
        state,
        action,
        next_state,
        reward,
        done,
    ):

        self.states[self.ptr] = state

        self.actions[self.ptr] = action

        self.next_states[self.ptr] = next_state

        self.rewards[self.ptr] = reward

        self.dones[self.ptr] = float(done)

        self.ptr = (
            self.ptr + 1
        ) % self.max_size

        self.size = min(
            self.size + 1,
            self.max_size,
        )

    # ========================================================
    # Sample
    # ========================================================

    def sample(
        self,
        batch_size,
        device,
    ):

        indices = np.random.randint(
            0,
            self.size,
            size=batch_size,
        )

        states = torch.as_tensor(
            self.states[indices],
            dtype=torch.float32,
            device=device,
        )

        actions = torch.as_tensor(
            self.actions[indices],
            dtype=torch.float32,
            device=device,
        )

        next_states = torch.as_tensor(
            self.next_states[indices],
            dtype=torch.float32,
            device=device,
        )

        rewards = torch.as_tensor(
            self.rewards[indices],
            dtype=torch.float32,
            device=device,
        )

        dones = torch.as_tensor(
            self.dones[indices],
            dtype=torch.float32,
            device=device,
        )

        return (
            states,
            actions,
            next_states,
            rewards,
            dones,
        )

    # ========================================================
    # State dict
    # ========================================================

    def state_dict(self):

        return {

            "max_size":
                self.max_size,

            "ptr":
                self.ptr,

            "size":
                self.size,

            "states":
                self.states,

            "actions":
                self.actions,

            "next_states":
                self.next_states,

            "rewards":
                self.rewards,

            "dones":
                self.dones,
        }

    # ========================================================
    # Load state
    # ========================================================

    def load_state_dict(
        self,
        data,
    ):

        self.ptr = data["ptr"]

        self.size = data["size"]

        self.states[:] = data["states"]

        self.actions[:] = data["actions"]

        self.next_states[:] = (
            data["next_states"]
        )

        self.rewards[:] = (
            data["rewards"]
        )

        self.dones[:] = (
            data["dones"]
        )


# ============================================================
# Actor
# ============================================================

class Actor(nn.Module):

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
    ):

        super().__init__()

        self.l1 = nn.Linear(
            state_dim,
            256,
        )

        self.l2 = nn.Linear(
            256,
            256,
        )

        self.l3 = nn.Linear(
            256,
            action_dim,
        )

        self.max_action = max_action

    def forward(
        self,
        state,
    ):

        x = F.relu(
            self.l1(state)
        )

        x = F.relu(
            self.l2(x)
        )

        x = torch.tanh(
            self.l3(x)
        )

        return (
            self.max_action
            * x
        )


# ============================================================
# Critic
# ============================================================

class Critic(nn.Module):

    def __init__(
        self,
        state_dim,
        action_dim,
    ):

        super().__init__()

        # Q1
        self.l1 = nn.Linear(
            state_dim + action_dim,
            256,
        )

        self.l2 = nn.Linear(
            256,
            256,
        )

        self.l3 = nn.Linear(
            256,
            1,
        )

        # Q2
        self.l4 = nn.Linear(
            state_dim + action_dim,
            256,
        )

        self.l5 = nn.Linear(
            256,
            256,
        )

        self.l6 = nn.Linear(
            256,
            1,
        )

    # ========================================================
    # Q1
    # ========================================================

    def Q1(
        self,
        state,
        action,
    ):

        xu = torch.cat(
            [
                state,
                action,
            ],
            dim=1,
        )

        x = F.relu(
            self.l1(xu)
        )

        x = F.relu(
            self.l2(x)
        )

        return self.l3(x)

    # ========================================================
    # Q2
    # ========================================================

    def Q2(
        self,
        state,
        action,
    ):

        xu = torch.cat(
            [
                state,
                action,
            ],
            dim=1,
        )

        x = F.relu(
            self.l4(xu)
        )

        x = F.relu(
            self.l5(x)
        )

        return self.l6(x)

    # ========================================================
    # Both
    # ========================================================

    def forward(
        self,
        state,
        action,
    ):

        return (
            self.Q1(
                state,
                action,
            ),
            self.Q2(
                state,
                action,
            ),
        )


# ============================================================
# TD3 Agent
# ============================================================

class TD3Agent:

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
    ):

        self.device = device

        self.max_action = float(
            max_action
        )

        # --------------------------------------------------------
        # Networks
        # --------------------------------------------------------

        self.actor = Actor(
            state_dim,
            action_dim,
            self.max_action,
        ).to(device)

        self.actor_target = Actor(
            state_dim,
            action_dim,
            self.max_action,
        ).to(device)

        self.critic = Critic(
            state_dim,
            action_dim,
        ).to(device)

        self.critic_target = Critic(
            state_dim,
            action_dim,
        ).to(device)

        # --------------------------------------------------------
        # Copy parameters
        # --------------------------------------------------------

        self.actor_target.load_state_dict(
            self.actor.state_dict()
        )

        self.critic_target.load_state_dict(
            self.critic.state_dict()
        )

        # --------------------------------------------------------
        # Optimizers
        # --------------------------------------------------------

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=ACTOR_LR,
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=CRITIC_LR,
        )

        # --------------------------------------------------------
        # Update counter
        # --------------------------------------------------------

        self.total_updates = 0

    # ============================================================
    # Select action
    # ============================================================

    def select_action(
        self,
        state,
        noise=0.0,
    ):

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():

            action = self.actor(
                state_tensor
            ).cpu().numpy()[0]

        if noise > 0.0:

            action += np.random.normal(
                0.0,
                noise,
                size=action.shape,
            )

        action = np.clip(
            action,
            -self.max_action,
            self.max_action,
        )

        return action.astype(
            np.float32
        )

    # ============================================================
    # Train
    # ============================================================

    def train(
        self,
        replay_buffer,
        batch_size,
    ):

        (
            state,
            action,
            next_state,
            reward,
            done,
        ) = replay_buffer.sample(
            batch_size,
            self.device,
        )

        # --------------------------------------------------------
        # Target action smoothing
        # --------------------------------------------------------

        noise = (
            torch.randn_like(action)
            * POLICY_NOISE
        )

        noise = noise.clamp(
            -NOISE_CLIP,
            NOISE_CLIP,
        )

        next_action = (
            self.actor_target(
                next_state
            )
            + noise
        )

        next_action = next_action.clamp(
            -self.max_action,
            self.max_action,
        )

        # --------------------------------------------------------
        # Target Q
        # --------------------------------------------------------

        with torch.no_grad():

            target_q1, target_q2 = (
                self.critic_target(
                    next_state,
                    next_action,
                )
            )

            target_q = torch.min(
                target_q1,
                target_q2,
            )

            target = (
                reward
                + (1.0 - done)
                * GAMMA
                * target_q
            )

        # --------------------------------------------------------
        # Current Q
        # --------------------------------------------------------

        current_q1, current_q2 = (
            self.critic(
                state,
                action,
            )
        )

        critic_loss = (

            F.mse_loss(
                current_q1,
                target,
            )

            + F.mse_loss(
                current_q2,
                target,
            )
        )

        # --------------------------------------------------------
        # Critic update
        # --------------------------------------------------------

        self.critic_optimizer.zero_grad()

        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            10.0,
        )

        self.critic_optimizer.step()

        # --------------------------------------------------------
        # Delayed actor update
        # --------------------------------------------------------

        self.total_updates += 1

        if (
            self.total_updates
            % POLICY_DELAY
            == 0
        ):

            actor_loss = (
                -self.critic.Q1(
                    state,
                    self.actor(state),
                ).mean()
            )

            self.actor_optimizer.zero_grad()

            actor_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                10.0,
            )

            self.actor_optimizer.step()

            # ----------------------------------------------------
            # Soft target updates
            # ----------------------------------------------------

            for param, target_param in zip(
                self.actor.parameters(),
                self.actor_target.parameters(),
            ):

                target_param.data.copy_(
                    TAU
                    * param.data
                    + (1.0 - TAU)
                    * target_param.data
                )

            for param, target_param in zip(
                self.critic.parameters(),
                self.critic_target.parameters(),
            ):

                target_param.data.copy_(
                    TAU
                    * param.data
                    + (1.0 - TAU)
                    * target_param.data
                )

        return (
            float(critic_loss.item()),
            float(actor_loss.item())
            if (
                self.total_updates
                % POLICY_DELAY
                == 0
            )
            else None,
        )


# ============================================================
# Checkpoint utilities
# ============================================================

def find_latest_checkpoint(
    checkpoint_dir,
):

    checkpoint_dir = Path(
        checkpoint_dir
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = list(
        checkpoint_dir.glob(
            "td3_step_*.pth"
        )
    )

    if len(files) == 0:

        return None

    def get_step(path):

        match = re.search(
            r"td3_step_(\d+)\.pth",
            path.name,
        )

        if match is None:

            return -1

        return int(
            match.group(1)
        )

    files.sort(
        key=get_step
    )

    return files[-1]


# ============================================================
# Save checkpoint
# ============================================================

def save_checkpoint(
    path,
    agent,
    replay_buffer,
    total_steps,
    episode,
    episode_reward,
):

    checkpoint = {

        # --------------------------------------------------------
        # Networks
        # --------------------------------------------------------

        "actor":
            agent.actor.state_dict(),

        "actor_target":
            agent.actor_target.state_dict(),

        "critic":
            agent.critic.state_dict(),

        "critic_target":
            agent.critic_target.state_dict(),

        # --------------------------------------------------------
        # Optimizers
        # --------------------------------------------------------

        "actor_optimizer":
            agent.actor_optimizer.state_dict(),

        "critic_optimizer":
            agent.critic_optimizer.state_dict(),

        # --------------------------------------------------------
        # Training state
        # --------------------------------------------------------

        "total_steps":
            total_steps,

        "episode":
            episode,

        "episode_reward":
            episode_reward,

        "total_updates":
            agent.total_updates,

        # --------------------------------------------------------
        # Replay buffer
        # --------------------------------------------------------

        "replay_buffer":
            replay_buffer.state_dict(),

        # --------------------------------------------------------
        # Random states
        # --------------------------------------------------------

        "numpy_random_state":
            np.random.get_state(),

        "torch_random_state":
            torch.random.get_rng_state(),

    }

    if torch.cuda.is_available():

        checkpoint[
            "cuda_random_state"
        ] = torch.cuda.get_rng_state_all()

    torch.save(
        checkpoint,
        path,
    )

    print(
        f"\nCheckpoint saved: {path}"
    )


# ============================================================
# Load checkpoint
# ============================================================

def load_checkpoint(
    path,
    agent,
    replay_buffer,
    device,
):

    print(
        f"\nLoading checkpoint:"
        f" {path}"
    )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    # --------------------------------------------------------
    # Networks
    # --------------------------------------------------------

    agent.actor.load_state_dict(
        checkpoint["actor"]
    )

    agent.actor_target.load_state_dict(
        checkpoint["actor_target"]
    )

    agent.critic.load_state_dict(
        checkpoint["critic"]
    )

    agent.critic_target.load_state_dict(
        checkpoint["critic_target"]
    )

    # --------------------------------------------------------
    # Optimizers
    # --------------------------------------------------------

    agent.actor_optimizer.load_state_dict(
        checkpoint["actor_optimizer"]
    )

    agent.critic_optimizer.load_state_dict(
        checkpoint["critic_optimizer"]
    )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    total_steps = checkpoint.get(
        "total_steps",
        0,
    )

    episode = checkpoint.get(
        "episode",
        0,
    )

    episode_reward = checkpoint.get(
        "episode_reward",
        0.0,
    )

    agent.total_updates = checkpoint.get(
        "total_updates",
        0,
    )

    # --------------------------------------------------------
    # Replay buffer
    # --------------------------------------------------------

    if (
        "replay_buffer"
        in checkpoint
    ):

        replay_buffer.load_state_dict(
            checkpoint[
                "replay_buffer"
            ]
        )

        print(
            "Replay buffer restored:"
            f" {replay_buffer.size}"
            " transitions"
        )

    else:

        print(
            "Warning: checkpoint "
            "contains no replay buffer."
        )

    # --------------------------------------------------------
    # Random states
    # --------------------------------------------------------

    if (
        "numpy_random_state"
        in checkpoint
    ):

        np.random.set_state(
            checkpoint[
                "numpy_random_state"
            ]
        )

    if (
        "torch_random_state"
        in checkpoint
    ):

        torch.random.set_rng_state(
            checkpoint[
                "torch_random_state"
            ].cpu()
        )

    if (
        torch.cuda.is_available()
        and
        "cuda_random_state"
        in checkpoint
    ):

        torch.cuda.set_rng_state_all(
            checkpoint[
                "cuda_random_state"
            ]
        )

    print(
        f"Resume from "
        f"step={total_steps}, "
        f"episode={episode}"
    )

    return (
        total_steps,
        episode,
        episode_reward,
    )


# ============================================================
# Remove old checkpoints
# ============================================================

def cleanup_old_checkpoints(
    checkpoint_dir,
    keep=5,
):

    checkpoint_dir = Path(
        checkpoint_dir
    )

    files = list(
        checkpoint_dir.glob(
            "td3_step_*.pth"
        )
    )

    def get_step(path):

        match = re.search(
            r"td3_step_(\d+)\.pth",
            path.name,
        )

        if match is None:

            return -1

        return int(
            match.group(1)
        )

    files.sort(
        key=get_step,
        reverse=True,
    )

    for old_file in files[keep:]:

        try:

            old_file.unlink()

            print(
                f"Removed old checkpoint:"
                f" {old_file}"
            )

        except OSError as e:

            print(
                f"Could not remove:"
                f" {old_file}"
                f" ({e})"
            )


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    agent,
    env,
    episodes=5,
):

    rewards = []

    for _ in range(episodes):

        state, _ = env.reset()

        done = False

        total_reward = 0.0

        while not done:

            action = agent.select_action(
                state,
                noise=0.0,
            )

            (
                next_state,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(action)

            state = next_state

            total_reward += reward

            done = (
                terminated
                or truncated
            )

        rewards.append(
            total_reward
        )

    return float(
        np.mean(rewards)
    )


# ============================================================
# Main training
# ============================================================

def main():

    set_seed(SEED)

    print(
        "======================================"
    )

    print(
        "Ball-Beam TD3"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        "======================================"
    )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = BallBeamEnv()

    eval_env = BallBeamEnv()

    state_dim = (
        env.observation_space.shape[0]
    )

    action_dim = (
        env.action_space.shape[0]
    )

    max_action = float(
        env.action_space.high[0]
    )

    print(
        f"State dimension: {state_dim}"
    )

    print(
        f"Action dimension: {action_dim}"
    )

    print(
        f"Torque limit: +/- {max_action}"
    )

    # --------------------------------------------------------
    # Agent
    # --------------------------------------------------------

    agent = TD3Agent(
        state_dim,
        action_dim,
        max_action,
        DEVICE,
    )

    # --------------------------------------------------------
    # Replay buffer
    # --------------------------------------------------------

    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        BUFFER_SIZE,
    )

    # --------------------------------------------------------
    # Checkpoint directory
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Find latest checkpoint
    # --------------------------------------------------------

    latest_checkpoint = (
        find_latest_checkpoint(
            CHECKPOINT_DIR
        )
    )

    total_steps = 0

    episode = 0

    episode_reward = 0.0

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    if latest_checkpoint is not None:

        (
            total_steps,
            episode,
            episode_reward,
        ) = load_checkpoint(
            latest_checkpoint,
            agent,
            replay_buffer,
            DEVICE,
        )

        print(
            "Training will continue "
            "from the latest checkpoint."
        )

    else:

        print(
            "No checkpoint found."
        )

        print(
            "Starting training from scratch."
        )

    print()

    # --------------------------------------------------------
    # Current environment state
    # --------------------------------------------------------

    state, _ = env.reset(
        seed=SEED
    )

    done = False

    current_episode_reward = (
        episode_reward
    )

    # ========================================================
    # Training loop
    # ========================================================

    try:

        while (
            total_steps
            < MAX_TOTAL_STEPS
        ):

            # ------------------------------------------------
            # Reset episode
            # ------------------------------------------------

            if done:

                episode += 1

                state, _ = env.reset()

                done = False

                current_episode_reward = 0.0

            # ------------------------------------------------
            # Select action
            # ------------------------------------------------

            if (
                total_steps
                < START_TIMESTEPS
                and replay_buffer.size
                < START_TIMESTEPS
            ):

                action = (
                    env.action_space
                    .sample()
                )

            else:

                action = (
                    agent.select_action(
                        state,
                        noise=EXPLORATION_NOISE,
                    )
                )

            # ------------------------------------------------
            # Environment step
            # ------------------------------------------------

            (
                next_state,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            done = (
                terminated
                or truncated
            )

            # ------------------------------------------------
            # Replay buffer
            #
            # For TD3, use termination rather
            # than time-limit truncation for
            # the bootstrap mask.
            # ------------------------------------------------

            replay_done = float(
                terminated
            )

            replay_buffer.add(
                state,
                action,
                next_state,
                reward,
                replay_done,
            )

            state = next_state

            current_episode_reward += reward

            total_steps += 1

            # ------------------------------------------------
            # TD3 update
            # ------------------------------------------------

            if (
                replay_buffer.size
                >= BATCH_SIZE
            ):

                critic_loss, actor_loss = (
                    agent.train(
                        replay_buffer,
                        BATCH_SIZE,
                    )
                )

            # ------------------------------------------------
            # Episode finished
            # ------------------------------------------------

            if done:

                print(
                    f"Step: {total_steps:8d} | "
                    f"Episode: {episode:6d} | "
                    f"Reward: "
                    f"{current_episode_reward:10.3f} | "
                    f"Buffer: "
                    f"{replay_buffer.size:6d}"
                )

            # ------------------------------------------------
            # Save checkpoint
            # ------------------------------------------------

            if (
                total_steps > 0
                and
                total_steps
                % SAVE_INTERVAL
                == 0
            ):

                checkpoint_path = (
                    CHECKPOINT_DIR
                    / (
                        "td3_step_"
                        f"{total_steps:08d}.pth"
                    )
                )

                save_checkpoint(
                    checkpoint_path,
                    agent,
                    replay_buffer,
                    total_steps,
                    episode,
                    current_episode_reward,
                )

                cleanup_old_checkpoints(
                    CHECKPOINT_DIR,
                    KEEP_CHECKPOINTS,
                )

                # ------------------------------------------------
                # Evaluation
                # ------------------------------------------------

                eval_reward = evaluate(
                    agent,
                    eval_env,
                    episodes=5,
                )

                print(
                    f"Evaluation reward: "
                    f"{eval_reward:.3f}"
                )

                print()

    except KeyboardInterrupt:

        print(
            "\nTraining interrupted."
        )

        # --------------------------------------------------------
        # Save emergency checkpoint
        # --------------------------------------------------------

        checkpoint_path = (
            CHECKPOINT_DIR
            / (
                "td3_step_"
                f"{total_steps:08d}.pth"
            )
        )

        save_checkpoint(
            checkpoint_path,
            agent,
            replay_buffer,
            total_steps,
            episode,
            current_episode_reward,
        )

        print(
            "Current training state has "
            "been saved."
        )

    finally:

        env.close()

        eval_env.close()

    print(
        "\nTraining finished."
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()