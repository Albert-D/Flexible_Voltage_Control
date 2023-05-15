from Environment import *
from DDPG import *

import torch
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

import pandapower as pp
import pandapower.networks as pn
import pandas as pd 

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
print(f"Using {device} device")

### create trainning environment
injection_bus = np.array([18, 21, 30, 45, 53])-1
pp_net = create_56bus()
env = VoltageCtrl_Env(pp_net, injection_bus)

seed = 10
num_agent = 5
obs_dim = env.obs_dim
action_dim = env.action_dim
hidden_dim = 100
num_episodes = 200
num_steps = 30  # trajetory length each episode
batch_size = 64
plot = False    # if/not plot trained policy every # episodes

"""
Create Agent list and replay buffer
"""
torch.manual_seed(seed)

# plot policy
def plot_policy(policy_net, episode):
    s_array = np.zeros(30,)

    a_array_baseline = np.zeros(30,)
    a_array = np.zeros(30,)
    for i in range(30):
        state = torch.tensor([0.85+0.01*i])
        s_array[i] = state

        action_baseline = -(np.maximum(state-1.05, 0)-np.maximum(0.95-state, 0)).reshape((1,))
        action = -agent_list[3].policy_net(state.reshape(1,1))

        a_array_baseline[i] = action_baseline[0]
        a_array[i] = action[0]
        
    plt.figure() 
    plt.plot(s_array, a_array_baseline, label = 'Baseline')
    plt.plot(s_array, a_array, label = 'RL')
    plt.savefig('Policy{0}.png'.format(episode), dpi=100)

agent_list = []
replay_buffer_list = []

for i in range(num_agent):
    value_net  = ValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    policy_net = FlexiblePolicyNet(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)

    target_value_net  = ValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    target_policy_net = FlexiblePolicyNet(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(param.data)

    for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(param.data)

    agent = DDPG(policy_net=policy_net, value_net=value_net,
                 target_policy_net=target_policy_net, target_value_net=target_value_net)
    
    replay_buffer = ReplayBuffer(capacity=1000000)
    
    agent_list.append(agent)
    replay_buffer_list.append(replay_buffer)

rewards = []
avg_reward_list = []

for episode in range(num_episodes):
    if(episode%50==0 and plot == True):
        plot_policy(agent_list[3].policy_net, episode)

    state = env.reset(seed = episode)
    topology = env.network.line.x_ohm_per_km
    episode_reward = 0
    last_action = np.zeros((num_agent,1))

    for step in range(num_steps):
        action = []
        topology = torch.cuda.FloatTensor(topology).unsqueeze(0)
        for i in range(num_agent):
            # sample action according to the current policy and exploration noise
            state_i = torch.cuda.FloatTensor(state[i].reshape(1,)).unsqueeze(0)
            action_agent = agent_list[i].policy_net(state_i, topology)
            action_agent = action_agent.detach().cpu().numpy()[0] + np.random.normal(0, 0.05)
            #print(action_agent)
            action_agent = np.clip(action_agent, -0.3, 0.3)
            action.append(action_agent)

        # PI policy    
        action = last_action - np.asarray(action)

        # execute action a_t and observe reward r_t and observe next state s_{t+1}
        next_state, topology, reward, done = env.step_uncertain(action)

        if(np.min(next_state)<0.75): #if voltage violation > 25%, episode ends.
            break
        else:
            for i in range(num_agent): 
                state_buffer = state[i].reshape(1,)
                action_buffer = action[i].reshape(1,)
                last_action_buffer = last_action[i].reshape(1,)
                next_state_buffer = next_state[i].reshape(1, )

                # store transition (s_t, a_t, r_t, s_{t+1}) in R
                replay_buffer_list[i].push(state_buffer, topology, action_buffer, last_action_buffer,
                                            reward, next_state_buffer, done)

                # update both critic and actor network
                if len(replay_buffer_list[i]) > batch_size:
                    agent_list[i].train_step_uncertain(replay_buffer=replay_buffer_list[i], 
                                                batch_size=batch_size)

            if(done):
                episode_reward += reward  
                break
            else:
                state = np.copy(next_state)
                episode_reward += reward    

        last_action = np.copy(action)

    rewards.append(episode_reward)
    avg_reward = np.mean(rewards[-40:])
    print(action)
    if(episode%10==0):
        print("Episode * {} * Avg Reward is ==> {}".format(episode, avg_reward))
    avg_reward_list.append(avg_reward)
