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
num_episodes = 50
num_steps = 30  # trajetory length each episode
batch_size = 256
plot = False    # if/not plot trained policy every # episodes

"""
Create Agent list and replay buffer
"""
torch.manual_seed(seed)


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

### load nn model parameter from saved model 
for i in range(num_agent):
    value_net_dict = torch.load(f'saved_models/value_net/Step_200_Seed_10_a{i}.pth')
    policy_net_dict = torch.load(f'saved_models/policy_net/Step_200_Seed_10_a{i}.pth')

    agent_list[i].value_net.load_state_dict(value_net_dict)
    agent_list[i].policy_net.load_state_dict(policy_net_dict)

rewards = []
avg_reward_list = []

for episode in range(num_episodes+1):

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
    
    ### seve nn model parameters
    if(episode%100 == 0):
        for i in range(num_agent):
            value_pth = f'saved_models/value_net/Step_{episode}_Seed_{seed}_a{i}.pth'
            policy_pth = f'saved_models/policy_net/Step_{episode}_Seed_{seed}_a{i}.pth'

            torch.save(agent_list[i].value_net.state_dict(), value_pth)
            torch.save(agent_list[i].policy_net.state_dict(), policy_pth)
            print('value net parameters had saved to ', value_pth)
            print('policy_net parameters had saved to', policy_pth)
        
    avg_reward_list.append(avg_reward)

## test policy
title = ['Bus 18', 'Bus 21', 'Bus 30', 'Bus 45', 'Bus 53']

fig, axs = plt.subplots(1, 5, figsize=(15,3))

topology = env.network.line.x_ohm_per_km
topology = torch.cuda.FloatTensor(topology).unsqueeze(0)
for i in range(num_agent):
    # plot policy
    N = 40
    s_array = np.zeros(N,)
    
    a_array_baseline = np.zeros(N,)
    a_array = np.zeros(N,)
    
    for j in range(N):
        state = torch.tensor([[0.8+0.01*j]])
        s_array[j] = state

        action_baseline = (np.maximum(state.cpu()-1.05, 0)-np.maximum(0.95-state.cpu(), 0)).reshape((1,))
    
        action = agent_list[i].policy_net(state, topology)
        action = action.detach().cpu().numpy()[0]
        
        a_array_baseline[j] = -action_baseline[0]
        a_array[j] = -action

    axs[i].plot(12*s_array, 2*a_array_baseline, '-.', label = 'Linear')
    axs[i].plot(12*s_array, a_array, label = 'Stable-DDPG')
    axs[i].legend(loc='lower left')

plt.show()

# plot the reward 
plt.plot(range(len(avg_reward_list)), avg_reward_list)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.grid(True)
plt.show()