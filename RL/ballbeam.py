import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class BallBeamEnv(gym.Env):
    """
    Nonlinear Ball-and-Beam environment.

    State:
        x       : ball position along the beam [m]
        x_dot   : ball velocity [m/s]
        theta   : beam angle [rad]
        theta_dot : beam angular velocity [rad/s]

    Action:
        beam torque [N*m]

    Dynamics:
        m_eff * x_ddot
            = m*x*theta_dot^2
              - m*g*sin(theta)
              - c_x*x_dot

        (I + m*x^2) * theta_ddot
            = torque
              - c_theta*theta_dot
              - k_theta*theta
              - 2*m*x*x_dot*theta_dot
              - m*g*x*cos(theta)

    The rolling-ball effective mass is

        m_eff = m + J_ball / r_ball^2.

    For a solid sphere:
        J_ball = 2/5 * m * r_ball^2
    and therefore
        m_eff = 7/5 * m.
    """

    metadata = {
        "render_modes": ["human"],
        "render_fps": 50,
    }

    def __init__(self, render_mode=None):

        super().__init__()

        self.render_mode = render_mode

        # ============================================================
        # Simulation parameters
        # ============================================================

        self.dt = 0.02
        self.max_steps = 1000

        # ============================================================
        # Physical parameters
        # ============================================================

        self.g = 9.81

        # ------------------------------------------------------------
        # Ball
        # ------------------------------------------------------------

        self.ball_mass = 0.05       # kg
        self.ball_radius = 0.02     # m

        # Solid sphere inertia
        self.ball_inertia = (
            2.0 / 5.0
            * self.ball_mass
            * self.ball_radius ** 2
        )

        # Effective translational mass due to rolling
        self.ball_effective_mass = (
            self.ball_mass
            + self.ball_inertia
            / self.ball_radius ** 2
        )

        # ------------------------------------------------------------
        # Beam
        # ------------------------------------------------------------

        # Beam + actuator rotational inertia
        self.I = 0.01               # kg*m^2

        # ------------------------------------------------------------
        # Damping
        # ------------------------------------------------------------

        # Ball translational viscous damping
        self.ball_damping = 0.01    # N*s/m

        # Beam rotational viscous damping
        self.beam_damping = 0.05    # N*m*s/rad

        # ------------------------------------------------------------
        # Optional beam restoring stiffness
        # ------------------------------------------------------------

        # Standard ball-beam model normally does not require this.
        # Keep it as an optional parameter.
        self.beam_stiffness = 0.0   # N*m/rad

        # ============================================================
        # Geometry
        # ============================================================

        # Beam extends from -L to +L
        self.beam_half_length = 2.0

        # Safe beam-angle limit
        self.theta_limit = np.deg2rad(35.0)

        # ============================================================
        # State limits used by the environment
        # ============================================================

        self.x_limit = self.beam_half_length

        # Observation space is deliberately wider than the
        # termination region, so no normal-state clipping is required.
        self.observation_space = spaces.Box(
            low=np.array(
                [
                    -2.5,       # x
                    -15.0,      # x_dot
                    -np.pi / 2,
                    -25.0
                ],
                dtype=np.float32
            ),
            high=np.array(
                [
                    2.5,
                    15.0,
                    np.pi / 2,
                    25.0
                ],
                dtype=np.float32
            ),
            dtype=np.float32
        )

        # ============================================================
        # Action
        # ============================================================

        self.torque_limit = 0.5

        self.action_space = spaces.Box(
            low=np.array(
                [-self.torque_limit],
                dtype=np.float32
            ),
            high=np.array(
                [self.torque_limit],
                dtype=np.float32
            ),
            dtype=np.float32
        )

        # ============================================================
        # State
        # ============================================================

        self.state = None
        self.step_count = 0

        # ============================================================
        # Rendering
        # ============================================================

        self.fig = None
        self.ax = None

    # ================================================================
    # Reset
    # ================================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.step_count = 0

        # Small random initial condition
        x0 = self.np_random.uniform(
            -0.8,
            0.8
        )

        x_dot0 = self.np_random.uniform(
            -0.05,
            0.05
        )

        theta0 = self.np_random.uniform(
            -0.10,
            0.10
        )

        theta_dot0 = self.np_random.uniform(
            -0.05,
            0.05
        )

        self.state = np.array(
            [
                x0,
                x_dot0,
                theta0,
                theta_dot0
            ],
            dtype=np.float64
        )

        if self.render_mode == "human":
            self.render()

        return self.state.astype(np.float32).copy(), {}

    # ================================================================
    # Continuous dynamics
    # ================================================================

    def dynamics(self, state, torque):

        x, x_dot, theta, theta_dot = state

        m = self.ball_mass
        m_eff = self.ball_effective_mass

        # ============================================================
        # Ball dynamics
        #
        # m_eff * x_ddot =
        #       m*x*theta_dot^2
        #       - m*g*sin(theta)
        #       - c_x*x_dot
        # ============================================================

        x_ddot = (
            m * x * theta_dot ** 2
            - m * self.g * np.sin(theta)
            - self.ball_damping * x_dot
        ) / m_eff

        # ============================================================
        # Beam dynamics
        #
        # (I + m*x^2) * theta_ddot =
        #
        #       torque
        #       - c_theta*theta_dot
        #       - k_theta*theta
        #       - 2*m*x*x_dot*theta_dot
        #       - m*g*x*cos(theta)
        # ============================================================

        effective_inertia = (
            self.I
            + m * x ** 2
        )

        theta_ddot = (
            torque
            - self.beam_damping * theta_dot
            - self.beam_stiffness * theta
            - 2.0 * m * x * x_dot * theta_dot
            - m * self.g * x * np.cos(theta)
        ) / effective_inertia

        return np.array(
            [
                x_dot,
                x_ddot,
                theta_dot,
                theta_ddot
            ],
            dtype=np.float64
        )

    # ================================================================
    # RK4 integration
    # ================================================================

    def _rk4_step(self, state, torque):

        dt = self.dt

        k1 = self.dynamics(
            state,
            torque
        )

        k2 = self.dynamics(
            state + 0.5 * dt * k1,
            torque
        )

        k3 = self.dynamics(
            state + 0.5 * dt * k2,
            torque
        )

        k4 = self.dynamics(
            state + dt * k3,
            torque
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

    # ================================================================
    # Step
    # ================================================================

    def step(self, action):

        # ------------------------------------------------------------
        # Current state
        # ------------------------------------------------------------

        state = self.state.copy()

        # ------------------------------------------------------------
        # Action
        # ------------------------------------------------------------

        action = np.asarray(
            action,
            dtype=np.float64
        ).reshape(-1)

        torque = float(
            np.clip(
                action[0],
                -self.torque_limit,
                self.torque_limit
            )
        )

        # ------------------------------------------------------------
        # Integrate dynamics
        # ------------------------------------------------------------

        next_state = self._rk4_step(
            state,
            torque
        )

        x, x_dot, theta, theta_dot = next_state

        self.step_count += 1

        # ------------------------------------------------------------
        # Termination conditions
        # ------------------------------------------------------------

        out_of_track = (
            abs(x) > self.x_limit
        )

        excessive_angle = (
            abs(theta) > self.theta_limit
        )

        terminated = (
            out_of_track
            or excessive_angle
        )

        truncated = (
            self.step_count >= self.max_steps
        )

        # ------------------------------------------------------------
        # Keep observation numerically valid
        #
        # No clipping during normal operation.
        # Only terminal states are clipped to observation bounds.
        # ------------------------------------------------------------

        if terminated:

            obs_low = self.observation_space.low
            obs_high = self.observation_space.high

            next_state = np.clip(
                next_state,
                obs_low,
                obs_high
            )

        self.state = next_state

        # ============================================================
        # Reward
        # ============================================================

        # State cost
        reward = -(
            2.0 * x ** 2
            + 0.10 * x_dot ** 2
            + 0.50 * theta ** 2
            + 0.01 * theta_dot ** 2
            + 0.001 * torque ** 2
        )

        # Stabilization bonus
        if (
            abs(x) < 0.05
            and abs(x_dot) < 0.10
            and abs(theta) < 0.05
            and abs(theta_dot) < 0.10
        ):
            reward += 5.0

        # ------------------------------------------------------------
        # Terminal penalty
        # ------------------------------------------------------------

        if out_of_track:
            reward -= 20.0

        if excessive_angle:
            reward -= 10.0

        # ============================================================
        # Info
        # ============================================================

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
            self.state.astype(np.float32).copy(),
            float(reward),
            terminated,
            truncated,
            info
        )

    # ================================================================
    # Render
    # ================================================================

    def render(self):

        if self.fig is None:

            self.fig, self.ax = plt.subplots(
                figsize=(7, 5)
            )

            plt.ion()

        x, _, theta, _ = self.state

        self.ax.clear()

        # ------------------------------------------------------------
        # Beam
        # ------------------------------------------------------------

        L = self.beam_half_length

        beam_x = np.array(
            [
                -L * np.cos(theta),
                 L * np.cos(theta)
            ]
        )

        beam_y = np.array(
            [
                -L * np.sin(theta),
                 L * np.sin(theta)
            ]
        )

        self.ax.plot(
            beam_x,
            beam_y,
            linewidth=5
        )

        # ------------------------------------------------------------
        # Ball
        # ------------------------------------------------------------

        ball_x = x * np.cos(theta)
        ball_y = x * np.sin(theta)

        self.ax.scatter(
            [ball_x],
            [ball_y],
            s=200
        )

        # ------------------------------------------------------------
        # Pivot
        # ------------------------------------------------------------

        self.ax.scatter(
            [0],
            [0],
            s=50
        )

        # ------------------------------------------------------------
        # Track limits
        # ------------------------------------------------------------

        left_x = (
            -L * np.cos(theta)
        )

        right_x = (
            L * np.cos(theta)
        )

        left_y = (
            -L * np.sin(theta)
        )

        right_y = (
            L * np.sin(theta)
        )

        self.ax.scatter(
            [left_x, right_x],
            [left_y, right_y],
            s=30
        )

        # ------------------------------------------------------------
        # Plot settings
        # ------------------------------------------------------------

        self.ax.set_xlim(
            -2.5,
            2.5
        )

        self.ax.set_ylim(
            -2.5,
            2.5
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
            "Nonlinear Ball-Beam TD3 Environment"
        )

        self.ax.grid(
            True,
            alpha=0.3
        )

        plt.pause(
            0.001
        )

    # ================================================================
    # Close
    # ================================================================

    def close(self):

        if self.fig is not None:

            plt.close(self.fig)

            self.fig = None
            self.ax = None