import numpy as np
import torch
import torch.nn as nn
import time

from ballbeam import BallBeamEnv



device=torch.device("cpu")



# =====================================================
# Actor
# =====================================================

class Actor(nn.Module):

    def __init__(
            self,
            s_dim,
            a_dim,
            action_max):

        super().__init__()

        self.action_max=action_max


        self.net=nn.Sequential(

            nn.Linear(
                s_dim,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                a_dim
            )
        )


    def forward(self,s):

        return self.action_max*torch.tanh(
            self.net(s)
        )



# =====================================================
# Environment
# =====================================================

env=BallBeamEnv()


s_dim=env.observation_space.shape[0]

a_dim=env.action_space.shape[0]


action_max=float(
    env.action_space.high[0]
)



# =====================================================
# Load Actor
# =====================================================

actor=Actor(
    s_dim,
    a_dim,
    action_max
).to(device)



checkpoint=torch.load(
    "lowrank_td3_ballbeam.pth",
    map_location=device
)


actor.load_state_dict(
    checkpoint["actor"]
)


actor.eval()


print("Actor loaded!")



# =====================================================
# Test
# =====================================================


test_episodes=5


test_env=BallBeamEnv(
    render_mode="human"
)



for ep in range(test_episodes):


    s,_=test_env.reset()


    total_reward=0



    for t in range(500):


        with torch.no_grad():


            s_tensor=torch.FloatTensor(
                s
            ).unsqueeze(0).to(device)



            # deterministic action

            action=actor(
                s_tensor
            ).cpu().numpy()[0]



        s,r,terminated,truncated,_=test_env.step(
            action
        )


        total_reward+=r



        test_env.render()


        time.sleep(0.01)



        if terminated or truncated:

            break



    print(
        f"Episode {ep+1} | "
        f"Reward {total_reward:.2f} | "
        f"Steps {t+1}"
    )



test_env.close()