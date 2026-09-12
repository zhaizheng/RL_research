import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class BallBeamEnv(gym.Env):

    metadata = {
        "render_modes": ["human"],
        "render_fps": 50
    }


    def __init__(self, render_mode=None):

        super().__init__()

        self.render_mode = render_mode

        # simulation step
        self.dt = 0.02


        # ==========================
        # physical parameters
        # ==========================

        self.g = 9.81

        # beam inertia
        self.I = 0.01


        # damping
        self.ball_damping = 0.1
        self.beam_damping = 0.05

        # beam restoring coefficient
        self.beam_stiffness = 0.5



        # ==========================
        # state
        #
        # x       ball position
        # xd      ball velocity
        # theta   beam angle
        # thetad  beam angular velocity
        #
        # ==========================


        high = np.array(
            [
                2.0,
                5.0,
                np.pi/2,
                10.0
            ],
            dtype=np.float32
        )


        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )


        # ==========================
        # action
        # torque
        # ==========================

        self.action_space = spaces.Box(
            low=np.array(
                [-2.0],
                dtype=np.float32
            ),

            high=np.array(
                [2.0],
                dtype=np.float32
            ),

            dtype=np.float32
        )


        self.state=None


        # rendering

        self.fig=None
        self.ax=None



    # =====================================================
    # Reset
    # =====================================================

    def reset(
        self,
        seed=None,
        options=None
    ):

        super().reset(seed=seed)


        self.state=np.array(
            [
                self.np_random.uniform(
                    -1.0,
                    1.0
                ),

                0.0,

                self.np_random.uniform(
                    -0.2,
                    0.2
                ),

                0.0

            ],
            dtype=np.float32
        )


        if self.render_mode=="human":
            self.render()


        return self.state.copy(), {}



    # =====================================================
    # Dynamics
    # =====================================================

    def step(self, action):


        x,xd,theta,thetad = self.state



        # action constraint

        torque=float(
            np.clip(
                action[0],
                -2.0,
                2.0
            )
        )



        # ==========================
        # ball dynamics
        #
        # rolling ball:
        #
        # xdd = 5/7*g*sin(theta)
        #
        # ==========================

        xdd = (
            (5.0/7.0)
            *self.g
            *np.sin(theta)

            -self.ball_damping*xd
        )



        # ==========================
        # beam dynamics
        #
        # I theta_ddot =
        # torque - damping - stiffness
        #
        # ==========================


        thetadd = (

            torque/self.I

            -self.beam_damping*thetad

            -self.beam_stiffness*theta
        )



        # ==========================
        # Euler integration
        # ==========================


        xd += xdd*self.dt

        x += xd*self.dt


        thetad += thetadd*self.dt

        theta += thetad*self.dt



        # ==========================
        # termination BEFORE clipping
        # ==========================


        out_of_bound = (
            abs(x)>2.0
        )



        # keep state valid

        x=np.clip(
            x,
            -2.0,
            2.0
        )


        xd=np.clip(
            xd,
            -5.0,
            5.0
        )


        theta=np.clip(
            theta,
            -np.pi/2,
            np.pi/2
        )


        thetad=np.clip(
            thetad,
            -10.0,
            10.0
        )



        self.state=np.array(
            [
                x,
                xd,
                theta,
                thetad
            ],
            dtype=np.float32
        )



        # ==========================
        # reward
        # ==========================

        reward = -(
            2.0*x*x
            +0.1*xd*xd
            +0.5*theta*theta
            +0.01*thetad*thetad
            +0.001*torque*torque
        )


        # small stabilization bonus

        if (
            abs(x)<0.05
            and abs(theta)<0.05
        ):

            reward += 5.0



        terminated=False

        truncated=out_of_bound



        if self.render_mode=="human":
            self.render()



        return (
            self.state.copy(),
            reward,
            terminated,
            truncated,
            {}
        )



    # =====================================================
    # Rendering
    # =====================================================

    def render(self):

        if self.fig is None:


            self.fig,self.ax=plt.subplots(
                figsize=(6,4)
            )

            plt.ion()



        x,_,theta,_=self.state


        self.ax.clear()



        # beam

        L=2.0


        beam_x=np.array(
            [
                -L*np.cos(theta),
                 L*np.cos(theta)
            ]
        )


        beam_y=np.array(
            [
                -L*np.sin(theta),
                 L*np.sin(theta)
            ]
        )


        self.ax.plot(
            beam_x,
            beam_y,
            linewidth=5
        )



        # ball

        ball_x=x*np.cos(theta)

        ball_y=x*np.sin(theta)


        self.ax.scatter(
            [ball_x],
            [ball_y],
            s=200
        )



        # pivot

        self.ax.scatter(
            [0],
            [0],
            s=40
        )


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


        self.ax.set_title(
            "Ball-Beam TD3 Control"
        )


        plt.pause(
            0.001
        )



    # =====================================================
    # Close
    # =====================================================

    def close(self):

        if self.fig is not None:

            plt.close(
                self.fig
            )

            self.fig=None