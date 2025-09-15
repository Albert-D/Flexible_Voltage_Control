from Environment import *
from DDPG import *
from TD3 import *
from NN_Module import *
from Utils import ReplayBuffer
from config import Config
from dashboard import PlotStore, start_dashboard
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
ENV = '56bus'

### create trainning environment
if ENV == '56bus':
    injection_bus = np.array([18, 21, 30, 45, 53])-1
    pp_net = create_56bus()
    env = VoltageCtrl_Env(pp_net, injection_bus)
elif ENV == '123bus':
    injection_bus = np.array([9, 10, 15, 19, 32, 35, 47, 58, 65, 74, 82, 91, 103, 60]) #11, 36, 75,/ 1,5,9
    pp_net = create_123bus()
    env = Env_123bus(pp_net, injection_bus)
elif ENV == '56bus_10':
    # injection_bus = np.array([5, 9, 18, 19, 21, 25, 30, 45, 48, 53])-1
    injection_bus = np.array([5, 9, 18, 21, 30, 45, 53])-1
    pp_net = create_56bus_10()
    env = VoltageCtrl_Env(pp_net, injection_bus)

# Read the seed value from the configuration file or initialize it to 0
# 123bus seed 2
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
plot = False                            # if/not plot trained policy every # episodes
if ENV == '56bus' or ENV == '56bus_10':
    maxaction = Config.max_action_56bus
    minaction = -Config.max_action_56bus
    num_steps = Config.total_steps          # trajetory length each episode
    batch_size = Config.batch_size
elif ENV == '123bus':
    maxaction = Config.max_action
    minaction = -Config.max_action
    num_steps = Config.total_steps_123bus          # trajetory length each episode
    batch_size = Config.batch_size_123bus


"""
Create Agent list and replay buffer
"""
torch.manual_seed(seed)

# be careful that this figure is defined galbal but use in function below
if ENV == '56bus':
    # fig, axs = plt.subplots(1, 5, figsize=(15,3))
    title = ['Bus 18', 'Bus 21', 'Bus 30', 'Bus 45', 'Bus 53']
elif ENV == '123bus':
    # fig, axs = plt.subplots(2, 7, figsize=(15,6))
    title = ['Bus 9', 'Bus 10', 'Bus 15', 'Bus19', 'Bus 32', 'Bus 35', 'Bus 47', 
                'Bus 58', 'Bus 65', 'Bus 74', 'Bus 72', 'Bus 91', 'Bus 103', 'Bus 60']
elif ENV == '56bus_10':
    # fig, axs = plt.subplots(2, 5, figsize=(15,6))
    title = ['Bus 5', 'Bus 9', 'Bus 18', 'Bus 19', 'Bus 21', 'Bus 25', 'Bus 30', 'Bus 45', 'Bus 48', 'Bus 53']
    

# def plot_policy(agent_list, topology):
#     plt.cla()
#     for i in range(num_agent):
#         if ENV == '123bus':
#             axs[i//7][i%7].clear()
#         elif ENV == '56bus':
#             axs[i].clear()
#         # plot policy
#         N = 40
#         s_array = np.zeros(N,)
        
#         a_array_baseline = np.zeros(N,)
#         a_array = np.zeros(N,)
        
#         for j in range(N):
#             v = 0.80+0.01*j
#             s_array[j] = v

#             action_baseline = (np.maximum(v-1.05, 0)-np.maximum(0.95-v, 0)).reshape((1,))

#             state = torch.cuda.FloatTensor(s_array[j].reshape(1,)).unsqueeze(0)
#             action = agent_list[i].policy_net(state, topology).detach()
#             action = float(action.view(-1)[0].cpu())
            
#             a_array_baseline[j] = -action_baseline[0]
#             a_array[j] = -action

#         if ENV == '123bus':
#             axs[i//7][i%7].plot(12*s_array, 10*a_array_baseline, '-.', label = 'Linear')
#             axs[i//7][i%7].plot(12*s_array, a_array, label = 'Flexible-DDPG')
#             axs[i//7][i%7].set_title(title[i])
#             axs[i//7][i%7].legend(loc='lower left')
#         elif ENV == '56bus':
#             axs[i].plot(12*s_array, 5*a_array_baseline, '-.', label = 'Linear')
#             axs[i].plot(12*s_array, a_array, label = 'Flexible-DDPG')
#             axs[i].set_title(title[i])
#             axs[i].legend(loc='lower left')

#     plt.pause(0.1)

## Function to compute policy curves for each agent
def compute_policy_curves(agent_list, topology, ENV, num_agent, N=40):
    """
    Compute the policy curves for each agent based on the current policy network.
    """
    try:
        s_array = np.linspace(0.80, 0.80 + 0.01 * (N - 1), N, dtype=np.float32)
        
        over = np.maximum(s_array - 1.05, 0.0)
        under = np.maximum(0.95 - s_array, 0.0)
        a_baseline = -(over - under)
        x = 12.0 * s_array

        y_lin_list, y_RL_list = [], []

        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for i in range(num_agent):
            if ENV == '123bus':
                y_lin = 10.0 * a_baseline
            else:
                y_lin = 5.0 * a_baseline
            
            try:
                policy_net = agent_list[i].policy_net
                was_training = policy_net.training
                
                with torch.no_grad():
                    policy_net.eval()
                    
                    states = torch.from_numpy(s_array).to(
                        device=target_device, 
                        dtype=torch.float32
                    ).view(N, 1)
                    
                    if torch.is_tensor(topology):
                        topo_t = topology.to(device=target_device, dtype=torch.float32)
                    else:
                        topo_t = torch.as_tensor(topology, device=target_device, dtype=torch.float32)
                    
                    if topo_t.dim() == 1:
                        topo_t = topo_t.unsqueeze(0)
                    if topo_t.shape[0] == 1 and N > 1:
                        topo_t = topo_t.expand(N, *topo_t.shape[1:]).contiguous()
                    
                    out = policy_net(states, topo_t).detach()
                    if out.ndim >= 2:
                        out = out.view(out.shape[0], -1)[:, 0]
                    y_RL = -out.cpu().numpy().reshape(-1)
                    
                    policy_net.train(was_training)
                    
            except Exception as e:
                print(f'[Training] Agent {i} inference failed: {e}')
                y_RL = np.zeros(N)
            
            y_lin_list.append(y_lin.tolist())
            y_RL_list.append(y_RL.tolist())

        return {
            'x': x.tolist(),
            'y_lin_list': y_lin_list,
            'y_RL_list': y_RL_list
        }
        
    except Exception as e:
        print(f'[Training] compute_policy_curves failed: {e}')
        return None

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
    if ENV == '56bus' or ENV == '56bus_10':
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

# start dashboard
store = PlotStore(ENV=ENV, title=title, num_agent=num_agent, N=40)
app, th = start_dashboard(store, host="127.0.0.1", port=8050)

rewards_history = []
avg_reward_list = []

for episode in range(num_episodes+1):
    #logger.info('------ now training episode {}  ------', episode)
    state, topology, senario = env.reset_topo(seed = episode)
    episode_reward = 0
    last_action = np.zeros((num_agent,1))
    fig_path = os.path.join(Config.data_path, f'images/policy_img/{today}/seed{seed}_episode_{episode}.png')

    # update policy in dashboard
    plot_data = compute_policy_curves(agent_list, topology, ENV, num_agent)
    if plot_data is not None:
        store.bump_policy(plot_data=plot_data)

    # use PlotStore save figure
    if episode % 50 == 0:
        if plot_data is not None:  # 使用刚计算的数据
            store.bump_policy(plot_data=plot_data)
        # 然后保存
        success = store.save_figure(fig_path, episode, seed)

    for step in range(num_steps):
        if keyboard.is_pressed('end'):
            break

        action = []
        topology = torch.tensor(topology, device='cuda', dtype=torch.float32).unsqueeze(0)

        for i in range(num_agent):
            # sample action according to the current policy and exploration noise
            state_i = torch.tensor(state[i].reshape(1,), device='cuda', dtype=torch.float32).unsqueeze(0)
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

        if store.save_figure(fig_path, episode, seed):
            logger.info(f'save policy image to {fig_path}')
        else:
            logger.warning(f'failed to save policy image to {fig_path}')
        break

    if not done:
        logger.info('episode {} finish with entire step!', episode)
    logger.info('reward of this trojectory was {}', episode_reward)
    rewards_history.append(episode_reward)

    # update dashboard
    store.add_reward(episode, episode_reward)
    store.bump_reward()

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

print("Training completed, saving final reward figure...")
final_reward_path = Config.data_path + f'images/reward_img/{today}/'
success = store.save_reward_figure(final_reward_path, seed=seed)
if success:
    print(f"Final reward figure saved to {final_reward_path}")