from Environment import *
from DDPG import *
from TD3 import *
from NN_Module import *
from Utils import ReplayBuffer
from config import Config
import sys
from loguru import logger
import os
from datetime import date
import keyboard
import shutil

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

### create two new folder to save result
today = date.today()
if not os.path.exists(Config.data_path + f'images/policy_img/{today}/'):
    os.makedirs(Config.data_path + f'images/policy_img/{today}/')
if not os.path.exists(Config.data_path + f'images/reward_img/{today}/'):
    os.makedirs(Config.data_path + f'images/reward_img/{today}/')

### define training algorithm 'DDPG' or 'TD3'
algorithm = 'TD3'

### define which envrionment to use, '56bus' or '123bus'
ENV = '123bus'

### create trainning environment
if ENV == '56bus':
    injection_bus = np.array([18, 21, 30, 45, 53])-1
    pp_net = create_56bus()
    env = VoltageCtrl_Env(pp_net, injection_bus)
elif ENV == '123bus':
    injection_bus = np.array([9, 10, 15, 19, 32, 35, 47, 58, 65, 74, 82, 91, 103, 60]) #11, 36, 75,/ 1,5,9
    pp_net = create_123bus()
    env = Env_123bus(pp_net, injection_bus)

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

save_config = os.path.join(Config.data_path,f'images/reward_img/{today}/', f'config_{seed}.py')
shutil.copy('config.py', save_config)
logger.info(f'config file saved to {save_config}')

num_agent = env.agentnum                # agent number is defined by environment
obs_dim = Config.obs_dim
action_dim = Config.action_dim
topology_hidden_dim = Config.topology_hidden_dim
num_episodes = Config.total_episodes
num_steps = Config.total_steps          # trajetory length each episode
batch_size = Config.batch_size
plot = False                            # if/not plot trained policy every # episodes
if ENV == '56bus':
    maxaction = Config.max_action_56bus
    minaction = -Config.max_action_56bus
elif ENV == '123bus':
    maxaction = Config.max_action
    minaction = -Config.max_action


"""
Create Agent list and replay buffer
"""
torch.manual_seed(seed)

#be careful that this figure is defined galbal but use in function below
if ENV == '56bus':
    fig, axs = plt.subplots(1, 5, figsize=(15,3))
    title = ['Bus 18', 'Bus 21', 'Bus 30', 'Bus 45', 'Bus 53']
elif ENV == '123bus':
    fig, axs = plt.subplots(2, 7, figsize=(15,6))
    title = ['Bus 9', 'Bus 10', 'Bus 15', 'Bus19', 'Bus 32', 'Bus 35', 'Bus 47', 
                'Bus 58', 'Bus 65', 'Bus 74', 'Bus 72', 'Bus 91', 'Bus 103', 'Bus 60']

def plot_policy(agent_list, topology):
    plt.cla()
    for i in range(num_agent):
        if ENV == '123bus':
            axs[i//7][i%7].clear()
        elif ENV == '56bus':
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

        if ENV == '123bus':
            axs[i//7][i%7].plot(12*s_array, 10*a_array_baseline, '-.', label = 'Linear')
            axs[i//7][i%7].plot(12*s_array, a_array, label = 'Flexible-DDPG')
            axs[i//7][i%7].set_title(title[i])
            axs[i//7][i%7].legend(loc='lower left')
        elif ENV == '56bus':
            axs[i].plot(12*s_array, 5*a_array_baseline, '-.', label = 'Linear')
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
        value_net = ValueNetwork(obs_dim=1, action_dim=action_dim, hidden_dim=256).to(device)
        target_value_net = ValueNetwork(obs_dim=1, action_dim=action_dim, hidden_dim=256).to(device)
    if algorithm == 'TD3':
        value_net = Q_Network(obs_dim=1, action_dim=action_dim,hidden_dim=256).to(device)
        target_value_net = Q_Network(obs_dim=1, action_dim=action_dim,hidden_dim=256).to(device)

    # Initialize the actor netowrk
    if ENV == '56bus':
        topology_net = TopologyNet(topology_dim=env.topology_dim, output_dim=1, hidden_dim=topology_hidden_dim)
        policy_net = FlexiblePolicyNet(env=env, topology_net=topology_net, obs_dim=obs_dim, 
                                    action_dim=action_dim, hidden_dim=Config.hidden_dim_56bus).to(device)
        target_policy_net = FlexiblePolicyNet(env=env, topology_net=topology_net, obs_dim=obs_dim, 
                                            action_dim=action_dim, hidden_dim=Config.hidden_dim_56bus).to(device)
    if ENV == '123bus':
        topology_net = TopologyNet(topology_dim=env.topology_dim, output_dim=1, hidden_dim=topology_hidden_dim)
        policy_net = FlexiblePolicyNet(env=env, topology_net=topology_net, obs_dim=obs_dim, 
                                    action_dim=action_dim, hidden_dim=Config.hidden_dim_123bus).to(device)
        target_policy_net = FlexiblePolicyNet(env=env, topology_net=topology_net, obs_dim=obs_dim, 
                                            action_dim=action_dim, hidden_dim=Config.hidden_dim_123bus).to(device)

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(param.data)

    for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(param.data)

    if algorithm == 'DDPG':
        agent = DDPG(policy_net=policy_net, value_net=value_net,
                    target_policy_net=target_policy_net, target_value_net=target_value_net)
    if algorithm == 'TD3':
        agent = TD3(policy_net = policy_net, value_net = value_net, target_policy_net = target_policy_net,
                    target_value_net = target_value_net, value_lr = Config.value_learning_rate,
                    policy_lr=Config.policy_learning_rate, max_action=Config.max_action)
    
    replay_buffer = ReplayBuffer(capacity=1000000)
    high_buffer = ReplayBuffer(capacity=1000000)
    low_buffer = ReplayBuffer(capacity=1000000)
    
    agent_list.append(agent)
    replay_buffer_list.append(replay_buffer)
    high_buffer_list.append(high_buffer)
    low_buffer_list.append(low_buffer)
    

# load nn model parameter from saved model 
# for i in range(num_agent):
#     value_net_dict = torch.load(f'check_points/value_net/2023-07-03/Step_50_Seed_8_a{i}.pth')
#     policy_net_dict = torch.load(f'check_points/policy_net/2023-07-03/Step_50_Seed_8_a{i}.pth')

#     agent_list[i].value_net.load_state_dict(value_net_dict)
#     agent_list[i].policy_net.load_state_dict(policy_net_dict)

rewards_history = []
avg_reward_list = []

for episode in range(num_episodes+1):
    #logger.info('------ now training episode {}  ------', episode)
    state, topology, senario = env.reset_topo(seed = episode)
    #topology = env.network.line.x_ohm_per_km
    # episode_reward = (np.clip(np.max(state)-env.v0, 0, np.inf) + np.clip(env.v0 - np.min(state), 0, np.inf)) * 3000
    # logger.debug('add reward {} to episode', episode_reward)
    episode_reward = 0
    last_action = np.zeros((num_agent,1))
    plot_policy(agent_list,torch.cuda.FloatTensor(topology).unsqueeze(0))
    if episode%50==0:
        fig_path = os.path.join(Config.data_path,f'images/policy_img/{today}/seed{seed}_episode_{episode}.png')
        plt.savefig(fig_path)
        logger.info(f'save policy image to {fig_path}')

    for step in range(num_steps):
        if keyboard.is_pressed('end'):
            break

        action = []
        topology = torch.cuda.FloatTensor(topology).unsqueeze(0)
        for i in range(num_agent):
            # sample action according to the current policy and exploration noise
            state_i = torch.cuda.FloatTensor(state[i].reshape(1,)).unsqueeze(0)
            action_agent = agent_list[i].policy_net(state_i, topology)
            if episode < 30:
                epsilon = np.random.normal(0, 0.5) / (episode+1)
            else:
                epsilon = np.random.normal(0, 0.05)
            epsilon = np.clip(epsilon, -0.5, 0.5)
            action_agent = action_agent.detach().cpu().numpy()[0] + epsilon #exploration
            logger.trace(action_agent)
            action_agent = np.clip(action_agent, minaction, maxaction)
            action.append(action_agent)

        # PI policy    
        action = last_action - np.asarray(action)
        # action = np.clip(action, -5.0, 5.0)

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
                                            Config.r_global_weight*reward+Config.r_local_weight*reward_sep[i],  # reward include two part
                                            next_state_buffer, done)
                
                # update both critic and actor network
                if len(replay_buffer_list[i]) > batch_size:

                    if algorithm == 'DDPG':
                        agent_list[i].train_step_uncertain(replay_buffer=replay_buffer_list[i], batch_size=batch_size)
                    if algorithm == 'TD3':
                        agent_list[i].train(replay_buffer=replay_buffer_list[i], iterations= i, batch_size=batch_size, 
                                            policy_noise=0.03, noise_clip=0.05, policy_freq=3)
                    
                if senario == 0:    # low voltage
                    low_buffer_list[i].push(state_buffer, topology, action_buffer, last_action_buffer,
                                            reward, next_state_buffer, done)
                if senario == 1:    # high voltage
                    high_buffer_list[i].push(state_buffer, topology, action_buffer, last_action_buffer,
                                            reward, next_state_buffer, done)
                    
            if(done):
                logger.success('episode {} done at step {}', episode, step)
                episode_reward += reward  
                break
            else:
                state = np.copy(next_state)
                episode_reward += reward    

        last_action = np.copy(action)

    if keyboard.is_pressed('end'):
        logger.warning("Training process terminated by user!")
        today = date.today()
        value_pth = Config.data_path + f'check_points/value_net/{today}/'
        policy_pth = Config.data_path + f'check_points/policy_net/{today}/'
        for i in range(num_agent):
            torch.save(agent_list[i].value_net.state_dict(), value_pth + f'Step_{episode}_Seed_{seed}_a{i}.pth')
            torch.save(agent_list[i].policy_net.state_dict(), policy_pth + f'Step_{episode}_Seed_{seed}_a{i}.pth')
            logger.info('value net parameters had saved to {}', value_pth + f'Step_{episode}_Seed_{seed}_a{i}.pth')
            logger.info('policy_net parameters had saved to {}', policy_pth + f'Step_{episode}_Seed_{seed}_a{i}.pth')

        plt.savefig(Config.data_path + f'images/policy_img/{today}/seed{seed}_episode_{episode}.png')
        logger.info(f'save policy image to images/policy_img/{today}/seed{seed}_episode_{episode}.png')
        break

    if not done:
        logger.info('episode {} finish with entire step!', episode)
    logger.info('reward of this trojectory was {}', episode_reward)
    rewards_history.append(episode_reward)
    avg_reward = np.mean(rewards_history[-50:])
    logger.trace(action)
    if(episode%10==0):
        print("Episode * {} * Avg Reward is ==> {}".format(episode, avg_reward))
    
    ### save nn model parameters
    if(episode%50 == 0):
        today = date.today()
        value_pth = Config.data_path + f'check_points/value_net/{today}/'
        policy_pth = Config.data_path + f'check_points/policy_net/{today}/'
        if not os.path.exists(value_pth): 
            os.makedirs(value_pth)
        if not os.path.exists(policy_pth):
            os.makedirs(policy_pth)
        for i in range(num_agent):
            torch.save(agent_list[i].value_net.state_dict(), value_pth + f'Step_{episode}_Seed_{seed}_a{i}.pth')
            torch.save(agent_list[i].policy_net.state_dict(), policy_pth + f'Step_{episode}_Seed_{seed}_a{i}.pth')
            logger.info('value net parameters had saved to : {}', value_pth + f'Step_{episode}_Seed_{seed}_a{i}.pth')
            logger.info('policy_net parameters had saved to : {}', policy_pth + f'Step_{episode}_Seed_{seed}_a{i}.pth')
        
    avg_reward_list.append(avg_reward)

# plot the reward 
fig2, axs2 = plt.subplots(1, 1)
plt.plot(range(len(avg_reward_list)), avg_reward_list)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.grid(True)
plt.savefig(Config.data_path + f'images/reward_img/{today}/avg_reward_{seed}.png')
plt.show()