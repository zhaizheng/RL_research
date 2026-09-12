import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ballbeam import BallBeamEnv

from collections import deque
import random
import copy
import time


# =====================================================
# Device
# =====================================================

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

print("device:", device)


device = torch.device("cpu")
# =====================================================
# Replay Buffer
# =====================================================

class ReplayBuffer:

    def __init__(self, size=100000):

        self.buffer = deque(maxlen=size)


    def add(self, s, a, r, s2, d):

        self.buffer.append(
            (
                s,
                a,
                r,
                s2,
                d
            )
        )


    def sample(self, batch_size):

        batch=random.sample(
            self.buffer,
            batch_size
        )

        s,a,r,s2,d=zip(*batch)


        return (

            torch.FloatTensor(
                np.array(s)
            ).to(device),

            torch.FloatTensor(
                np.array(a)
            ).to(device),

            torch.FloatTensor(
                np.array(r)
            ).unsqueeze(1).to(device),

            torch.FloatTensor(
                np.array(s2)
            ).to(device),

            torch.FloatTensor(
                np.array(d)
            ).unsqueeze(1).to(device)

        )


    def __len__(self):

        return len(self.buffer)




# =====================================================
# Low Rank Critic
# =====================================================

class LowRankQ(nn.Module):

    def __init__(
            self,
            s_dim,
            a_dim,
            rank=64):

        super().__init__()


        self.rank=rank


        self.phi=nn.Sequential(

            nn.Linear(s_dim,128),
            nn.ReLU(),

            nn.Linear(128,rank)

        )


        self.psi=nn.Sequential(

            nn.Linear(a_dim,128),
            nn.ReLU(),

            nn.Linear(128,rank)

        )


        self.residual=nn.Sequential(

            nn.Linear(
                s_dim+a_dim,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                1
            )

        )


        # positive scale

        self.log_scale=nn.Parameter(
            torch.tensor(2.0)
        )



    def forward(self,s,a):


        phi=self.phi(s)

        psi=self.psi(a)


        phi=F.normalize(
            phi,
            dim=-1
        )

        psi=F.normalize(
            psi,
            dim=-1
        )


        scale=torch.exp(
            self.log_scale
        )


        low_rank=scale*torch.sum(
            phi*psi,
            dim=-1,
            keepdim=True
        )


        residual=self.residual(
            torch.cat(
                [s,a],
                dim=-1
            )
        )


        return low_rank+residual




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


print(
    "state dim:",
    s_dim,
    "action dim:",
    a_dim
)



# =====================================================
# Networks
# =====================================================


actor=Actor(
    s_dim,
    a_dim,
    action_max
).to(device)


actor_target=copy.deepcopy(actor)



q1=LowRankQ(
    s_dim,
    a_dim,
    rank=64
).to(device)


q2=LowRankQ(
    s_dim,
    a_dim,
    rank=64
).to(device)



q1_target=copy.deepcopy(q1)

q2_target=copy.deepcopy(q2)



opt_actor=optim.Adam(
    actor.parameters(),
    lr=3e-4
)


opt_q1=optim.Adam(
    q1.parameters(),
    lr=1e-3
)


opt_q2=optim.Adam(
    q2.parameters(),
    lr=1e-3
)




# =====================================================
# TD3 Parameters
# =====================================================


buffer=ReplayBuffer()


gamma=0.99

tau=0.005


batch_size=256


warmup_steps=10000


policy_delay=2


total_steps=0


episodes=200

max_steps=500




# =====================================================
# Training
# =====================================================


reward_history=[]


for ep in range(episodes):


    s,_=env.reset()


    ep_reward=0



    for t in range(max_steps):


        total_steps+=1



        # exploration


        if total_steps<warmup_steps:


            action=env.action_space.sample()



        else:


            with torch.no_grad():


                s_tensor=torch.FloatTensor(
                    s
                ).unsqueeze(0).to(device)


                action=actor(
                    s_tensor
                ).cpu().numpy()[0]



            action+=np.random.normal(
                0,
                0.1*action_max,
                size=a_dim
            )


            action=np.clip(
                action,
                -action_max,
                action_max
            )




        s2,r,terminated,truncated,_=env.step(action)


        done=terminated or truncated



        buffer.add(
            s,
            action,
            r,
            s2,
            float(done)
        )


        s=s2

        ep_reward+=r



        # update


        if len(buffer)<warmup_steps:

            continue



        states,actions,rewards,next_states,dones=buffer.sample(
            batch_size
        )



        with torch.no_grad():


            next_actions=actor_target(
                next_states
            )


            noise=torch.randn_like(
                next_actions
            )*0.1


            noise=noise.clamp(
                -0.3,
                0.3
            )


            next_actions=(
                next_actions+noise
            ).clamp(
                -action_max,
                action_max
            )



            target_q=torch.min(

                q1_target(
                    next_states,
                    next_actions
                ),

                q2_target(
                    next_states,
                    next_actions
                )

            )



            y=rewards+gamma*(1-dones)*target_q





        # critic


        q1_loss=F.mse_loss(
            q1(states,actions),
            y
        )


        q2_loss=F.mse_loss(
            q2(states,actions),
            y
        )



        opt_q1.zero_grad()

        q1_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            q1.parameters(),
            5
        )

        opt_q1.step()



        opt_q2.zero_grad()

        q2_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            q2.parameters(),
            5
        )

        opt_q2.step()




        # actor update


        if total_steps%policy_delay==0:


            actor_loss=-q1(
                states,
                actor(states)
            ).mean()



            opt_actor.zero_grad()

            actor_loss.backward()


            torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                5
            )

            opt_actor.step()



            # soft update


            for p,pt in zip(
                actor.parameters(),
                actor_target.parameters()
            ):

                pt.data.copy_(
                    tau*p.data+
                    (1-tau)*pt.data
                )


            for p,pt in zip(
                q1.parameters(),
                q1_target.parameters()
            ):

                pt.data.copy_(
                    tau*p.data+
                    (1-tau)*pt.data
                )


            for p,pt in zip(
                q2.parameters(),
                q2_target.parameters()
            ):

                pt.data.copy_(
                    tau*p.data+
                    (1-tau)*pt.data
                )



        if done:
            break



    reward_history.append(ep_reward)


    print(
        f"Episode {ep:4d} | Reward {ep_reward:8.2f}"
    )

# =====================================================
# Save Model
# =====================================================

torch.save(
    {
        "actor": actor.state_dict(),

        "actor_target": actor_target.state_dict(),

        "q1": q1.state_dict(),

        "q2": q2.state_dict(),

        "q1_target": q1_target.state_dict(),

        "q2_target": q2_target.state_dict(),

        "reward_history": reward_history

    },
    "lowrank_td3_ballbeam.pth"
)


print("Model saved!")



env.close()




# =====================================================
# Test
# =====================================================


# =====================================================
# Multiple Test Episodes
# =====================================================
'''
test_episodes = 5

test_env = BallBeamEnv(
    render_mode="human"
)


for ep in range(test_episodes):

    s, _ = test_env.reset()

    total_reward = 0


    for t in range(1000):

        with torch.no_grad():

            s_tensor = torch.FloatTensor(
                s
            ).unsqueeze(0).to(device)


            # deterministic policy
            action = actor(
                s_tensor
            ).cpu().numpy()[0]



        s, r, terminated, truncated, _ = test_env.step(
            action
        )


        total_reward += r



        # render animation
        test_env.render()


        time.sleep(0.01)



        if terminated or truncated:

            break



    print(
        f"Test Episode {ep+1:2d} | "
        f"Reward {total_reward:8.2f} | "
        f"Steps {t+1}"
    )



test_env.close()
'''