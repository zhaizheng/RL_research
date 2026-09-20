import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ballbeam import BallBeamEnv


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device("cpu")

CHECKPOINT_DIR = Path("checkpoints")

TEST_EPISODES = 10

RENDER = True

RENDER_DELAY = 0.01


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

    def forward(self, state):

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
# Find latest checkpoint
# ============================================================

def find_latest_checkpoint(
    checkpoint_dir,
):

    checkpoint_dir = Path(
        checkpoint_dir
    )

    files = list(
        checkpoint_dir.glob(
            "td3_step_*.pth"
        )
    )

    if len(files) == 0:

        raise FileNotFoundError(
            f"No checkpoint found in "
            f"{checkpoint_dir.resolve()}"
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

    return files[0]


# ============================================================
# Load actor
# ============================================================

def load_latest_actor(
    env,
    device,
):

    checkpoint_path = (
        find_latest_checkpoint(
            CHECKPOINT_DIR
        )
    )

    print(
        f"Latest checkpoint:\n"
        f"  {checkpoint_path}"
    )

    # --------------------------------------------------------
    # Read checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    # --------------------------------------------------------
    # Environment dimensions
    # --------------------------------------------------------

    s_dim = (
        env.observation_space.shape[0]
    )

    a_dim = (
        env.action_space.shape[0]
    )

    action_max = float(
        env.action_space.high[0]
    )

    # --------------------------------------------------------
    # Actor
    # --------------------------------------------------------

    actor = Actor(
        s_dim,
        a_dim,
        action_max,
    ).to(device)

    actor.load_state_dict(
        checkpoint["actor"]
    )

    actor.eval()

    # --------------------------------------------------------
    # Training information
    # --------------------------------------------------------

    total_steps = checkpoint.get(
        "total_steps",
        None,
    )

    episode = checkpoint.get(
        "episode",
        None,
    )

    print(
        "Actor loaded successfully."
    )

    if total_steps is not None:

        print(
            f"Training steps: "
            f"{total_steps:,}"
        )

    if episode is not None:

        print(
            f"Training episodes: "
            f"{episode:,}"
        )

    print(
        f"Device: {device}"
    )

    return (
        actor,
        checkpoint_path,
    )


# ============================================================
# Deterministic action
# ============================================================

def select_action(
    actor,
    state,
    device,
):

    state_tensor = torch.as_tensor(
        state,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():

        action = actor(
            state_tensor
        ).cpu().numpy()[0]

    return action.astype(
        np.float32
    )


# ============================================================
# Evaluate
# ============================================================

def evaluate(
    actor,
    env,
    device,
    episodes=10,
    render=False,
):

    episode_rewards = []

    episode_lengths = []

    final_states = []

    success_count = 0

    print(
        "\n========================================"
    )

    print(
        "Evaluation"
    )

    print(
        "========================================"
    )

    for ep in range(episodes):

        state, _ = env.reset()

        total_reward = 0.0

        success = False

        start_time = time.time()

        for t in range(
            env.max_steps
        ):

            action = select_action(
                actor,
                state,
                device,
            )

            (
                next_state,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            state = next_state

            total_reward += reward

            if render:

                env.render()

                if RENDER_DELAY > 0:

                    time.sleep(
                        RENDER_DELAY
                    )

            # ------------------------------------------------
            # Define success as surviving
            # the full episode while remaining
            # inside the valid operating region.
            # ------------------------------------------------

            if truncated and not terminated:

                success = True

            if terminated or truncated:

                break

        episode_length = t + 1

        episode_rewards.append(
            total_reward
        )

        episode_lengths.append(
            episode_length
        )

        final_states.append(
            state.copy()
        )

        if success:

            success_count += 1

        elapsed = (
            time.time()
            - start_time
        )

        print(
            f"Episode {ep + 1:2d} | "
            f"Reward {total_reward:10.3f} | "
            f"Steps {episode_length:4d} | "
            f"x {state[0]:8.4f} | "
            f"x_dot {state[1]:8.4f} | "
            f"theta {state[2]:8.4f} | "
            f"theta_dot {state[3]:8.4f} | "
            f"{'SUCCESS' if success else 'TERMINATED'} | "
            f"{elapsed:.2f}s"
        )

    # ========================================================
    # Statistics
    # ========================================================

    episode_rewards = np.asarray(
        episode_rewards,
        dtype=np.float64,
    )

    episode_lengths = np.asarray(
        episode_lengths,
        dtype=np.float64,
    )

    final_states = np.asarray(
        final_states,
        dtype=np.float64,
    )

    mean_reward = (
        episode_rewards.mean()
    )

    std_reward = (
        episode_rewards.std()
    )

    mean_length = (
        episode_lengths.mean()
    )

    std_length = (
        episode_lengths.std()
    )

    success_rate = (
        success_count
        / episodes
    )

    mean_abs_final_x = np.mean(
        np.abs(
            final_states[:, 0]
        )
    )

    mean_abs_final_xdot = np.mean(
        np.abs(
            final_states[:, 1]
        )
    )

    mean_abs_final_theta = np.mean(
        np.abs(
            final_states[:, 2]
        )
    )

    mean_abs_final_thetadot = np.mean(
        np.abs(
            final_states[:, 3]
        )
    )

    # ========================================================
    # Summary
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "Evaluation Summary"
    )

    print(
        "========================================"
    )

    print(
        f"Mean reward       : "
        f"{mean_reward:.3f}"
    )

    print(
        f"Std reward        : "
        f"{std_reward:.3f}"
    )

    print(
        f"Mean episode len  : "
        f"{mean_length:.2f}"
    )

    print(
        f"Std episode len   : "
        f"{std_length:.2f}"
    )

    print(
        f"Success rate      : "
        f"{100.0 * success_rate:.1f}%"
    )

    print(
        "\nMean absolute final state:"
    )

    print(
        f"  |x|             : "
        f"{mean_abs_final_x:.5f}"
    )

    print(
        f"  |x_dot|         : "
        f"{mean_abs_final_xdot:.5f}"
    )

    print(
        f"  |theta|         : "
        f"{mean_abs_final_theta:.5f}"
    )

    print(
        f"  |theta_dot|     : "
        f"{mean_abs_final_thetadot:.5f}"
    )

    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_length": mean_length,
        "std_length": std_length,
        "success_rate": success_rate,
        "final_states": final_states,
    }


# ============================================================
# Main
# ============================================================

def main():

    print(
        "========================================"
    )

    print(
        "Ball-Beam TD3 Evaluation"
    )

    print(
        "========================================"
    )

    # --------------------------------------------------------
    # Environment used for loading dimensions
    # --------------------------------------------------------

    env = BallBeamEnv()

    # --------------------------------------------------------
    # Load latest checkpoint
    # --------------------------------------------------------

    actor, checkpoint_path = (
        load_latest_actor(
            env,
            DEVICE,
        )
    )

    env.close()

    # --------------------------------------------------------
    # Test environment
    # --------------------------------------------------------

    if RENDER:

        test_env = BallBeamEnv(
            render_mode="human"
        )

    else:

        test_env = BallBeamEnv()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    evaluate(
        actor,
        test_env,
        DEVICE,
        episodes=TEST_EPISODES,
        render=RENDER,
    )

    # --------------------------------------------------------
    # Close
    # --------------------------------------------------------

    test_env.close()

    print(
        "\nEvaluation finished."
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()