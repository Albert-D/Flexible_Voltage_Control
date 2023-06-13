from Environment import *
from DDPG import *
from TD3 import *
from NN_Module import *
from Utils import ReplayBuffer
import sys
from loguru import logger
import os
from datetime import date
import keyboard

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

### a simple logger
logger.remove()
logger.add(sys.stderr, level='DEBUG')

### define training algorithm 'DDPG' or 'TD3'
algorithm = 'TD3'

### create trainning environment
injection_bus = np.array([18, 21, 30, 45, 53])-1
pp_net = create_56bus()
env = VoltageCtrl_Env(pp_net, injection_bus)

# Read the seed value from the configuration file or initialize it to 0
try:
    with open('seed.txt', 'r') as file:
        seed = int(file.read())
except FileNotFoundError:
    seed = -1
# Increment the seed value by 1 and update the configuration file
seed += 1
with open('seed.txt', 'w') as file:
    file.write(str(seed))

num_agent = 5
obs_dim = env.obs_dim
action_dim = env.action_dim
hidden_dim = 512
num_episodes = 500
num_steps = 32  # trajetory length each episode
batch_size = 128
plot = False    # if/not plot trained policy every # episodes

"""
Create Agent list and replay buffer
"""
torch.manual_seed(seed)

#be careful that this figure is defined galbal but use in function below
fig, axs = plt.subplots(1, 5, figsize=(15,3))
title = ['Bus 18', 'Bus 21', 'Bus 30', 'Bus 45', 'Bus 53']
def plot_policy(agent_list, topology):
    plt.cla()
    for i in range(num_agent):
        axs[i].clear()
        # plot policy
        N = 40
        s_array = np.zeros(N,)
        
        a_array_baseline = np.zeros(N,)
        a_array = np.zeros(N,)
        
        for j in range(N):
            state = torch.tensor([[0.80+0.01*j]])
            s_array[j] = state

            action_baseline = (np.maximum(state.cpu()-1.05, 0)-np.maximum(0.95-state.cpu(), 0)).reshape((1,))
        
            action = agent_list[i].policy_net(state, topology)
            action = action.detach().cpu().numpy()[0]
            
            a_array_baseline[j] = -action_baseline[0]
            a_array[j] = -action

        axs[i].plot(12*s_array, 2*a_array_baseline, '-.', label = 'Linear')
        axs[i].plot(12*s_array, a_array, label = 'Flexible-DDPG')
        axs[i].set_title(title[i])
        axs[i].legend(loc='lower left')

    plt.pause(0.1)

agent_list = []
replay_buffer_list = []
high_buffer_list = []
low_buffer_list = []

### initilize network and DDPG
for i in range(num_agent):
    # Initialize the critic network 
    if algorithm == 'DDPG':
        value_net = ValueNetwork(obs_dim=1, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
        target_value_net = ValueNetwork(obs_dim=1, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    if algorithm == 'TD3':
        value_net = Q_Network(obs_dim=1, action_dim=action_dim,hidden_dim=256).to(device)
        target_value_net = Q_Network(obs_dim=1, action_dim=action_dim,hidden_dim=256).to(device)

    # Initialize the actor netowrk
    topology_net = TopologyNet(topology_dim=55, output_dim=1, hidden_dim=100)
    policy_net = FlexiblePolicyNet(env=env, topology_net=topology_net, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    target_policy_net = FlexiblePolicyNet(env=env, topology_net=topology_net, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(param.data)

    for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(param.data)

    if algorithm == 'DDPG':
        agent = DDPG(policy_net=policy_net, value_net=value_net,
                    target_policy_net=target_policy_net, target_value_net=target_value_net)
    if algorithm == 'TD3':
        agent = TD3(policy_net=policy_net, value_net=value_net, target_policy_net=target_policy_net,
                    target_value_net=target_value_net, max_action=0.5)
    
    replay_buffer = ReplayBuffer(capacity=1000000)
    high_buffer = ReplayBuffer(capacity=1000000)
    low_buffer = ReplayBuffer(capacity=1000000)
    
    agent_list.append(agent)
    replay_buffer_list.append(replay_buffer)
    high_buffer_list.append(high_buffer)
    low_buffer_list.append(low_buffer)
    

# load nn model parameter from saved model 
# for i in range(num_agent):
#     value_net_dict = torch.load(f'check_points/value_net/2023-06-12/Step_300_Seed_7_a{i}.pth')
#     policy_net_dict = torch.load(f'check_points/policy_net/2023-06-12/Step_300_Seed_7_a{i}.pth')

#     agent_list[i].value_net.load_state_dict(value_net_dict)
#     agent_list[i].policy_net.load_state_dict(policy_net_dict)

rewards = []
avg_reward_list = []

for episode in range(num_episodes+1):
    #logger.info('------ now training episode {}  ------', episode)
    state, topology, senario = env.reset(seed = episode)
    #topology = env.network.line.x_ohm_per_km
    episode_reward = 0
    last_action = np.zeros((num_agent,1))
    plot_policy(agent_list,torch.cuda.FloatTensor(topology).unsqueeze(0))
    if episode%50==0:
        today = date.today()
        if not os.path.exists(f'images/policy_img/{today}/'):
            os.makedirs(f'images/policy_img/{today}/')
        plt.savefig(f'images/policy_img/{today}/seed{seed}_episode_{episode}.png')
        logger.info(f'save policy image to images/policy_img/{today}/seed{seed}_episode_{episode}.png')

    for step in range(num_steps):
        if keyboard.is_pressed('q') or keyboard.is_pressed('esc'):
            break

        action = []
        topology = torch.cuda.FloatTensor(topology).unsqueeze(0)
        for i in range(num_agent):
            # sample action according to the current policy and exploration noise
            state_i = torch.cuda.FloatTensor(state[i].reshape(1,)).unsqueeze(0)
            action_agent = agent_list[i].policy_net(state_i, topology)
            if episode < 20:
                epsilon = np.random.normal(0, 0.2) / (episode+1)
            else:
                epsilon = np.random.normal(0, 0.02)
            epsilon = np.clip(epsilon, -0.3, 0.3)
            action_agent = action_agent.detach().cpu().numpy()[0] + epsilon #exploration
            logger.trace(action_agent)
            action_agent = np.clip(action_agent, -0.5, 0.5)
            action.append(action_agent)

        # PI policy    
        action = last_action - np.asarray(action)

        # execute action a_t and observe reward r_t and observe next state s_{t+1}
        next_state, topology, reward, reward_sep, done = env.step_uncertain(action)

        if(np.min(next_state) < 0.75 or np.max(next_state) > 1.25): #if voltage violation > 25%, episode ends.
            logger.warning('step {} break, min_state is {}, max_state is {}', step, np.min(next_state), np.max(next_state))
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
                    if algorithm == 'DDPG':
                        agent_list[i].train_step_uncertain(replay_buffer=replay_buffer_list[i], batch_size=batch_size)
                    if algorithm == 'TD3':
                        agent_list[i].train(replay_buffer=replay_buffer_list[i], iterations= i, batch_size=batch_size, 
                                            policy_noise=0.1, noise_clip=0.3, policy_freq=3)
                    
                if senario == 0:    # low voltage
                    low_buffer_list[i].push(state_buffer, topology, action_buffer, last_action_buffer,
                                            reward, next_state_buffer, done)
                if senario == 1:    # high voltage
                    high_buffer_list[i].push(state_buffer, topology, action_buffer, last_action_buffer,
                                            reward, next_state_buffer, done)
                    
                # if len(high_buffer_list[i]) > batch_size:
                #     agent_list[i].policy_net.z.requires_grad = False
                #     agent_list[i].policy_net.c.requires_grad = False
                #     agent_list[i].policy_net.q.requires_grad = True
                #     agent_list[i].policy_net.b.requires_grad = True
                #     agent_list[i].train(replay_buffer=high_buffer_list[i], iterations= i, batch_size=batch_size, 
                #                              policy_noise=0.1, noise_clip=0.3, policy_freq=3)

            if(done):
                logger.success('episode {} done at step {}', episode, step)
                episode_reward += reward  
                break
            else:
                state = np.copy(next_state)
                episode_reward += reward    

        last_action = np.copy(action)

    if keyboard.is_pressed('q') or keyboard.is_pressed('esc'):
        logger.warning("Training process terminated by user!")
        break
    if not done:
        logger.info('episode {} finish with entire step!', episode)
    rewards.append(episode_reward)
    avg_reward = np.mean(rewards[-40:])
    logger.trace(action)
    if(episode%10==0):
        print("Episode * {} * Avg Reward is ==> {}".format(episode, avg_reward))
    
    ### save nn model parameters
    if(episode%100 == 0):
        for i in range(num_agent):
            today = date.today()
            value_pth = f'check_points/value_net/{today}/'
            policy_pth = f'check_points/policy_net/{today}/'
            if not os.path.exists(value_pth): 
                os.makedirs(value_pth)
            if not os.path.exists(policy_pth):
                os.makedirs(policy_pth)

            torch.save(agent_list[i].value_net.state_dict(), value_pth + f'Step_{episode}_Seed_{seed}_a{i}.pth')
            torch.save(agent_list[i].policy_net.state_dict(), policy_pth + f'Step_{episode}_Seed_{seed}_a{i}.pth')
            logger.info('value net parameters had saved to {}', value_pth + f'Step_{episode}_Seed_{seed}_a{i}.pth')
            logger.info('policy_net parameters had saved to {}', policy_pth + f'Step_{episode}_Seed_{seed}_a{i}.pth')
        
    avg_reward_list.append(avg_reward)


# plot the reward 
fig, axs = plt.subplots(1, 1)
plt.plot(range(len(avg_reward_list)), avg_reward_list)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.grid(True)
if not os.path.exists(f'images/reward_img/{today}/'):
    os.makedirs(f'images/reward_img/{today}/')
plt.savefig(f'images/reward_img/{today}/reward_{seed}.png')
plt.show()