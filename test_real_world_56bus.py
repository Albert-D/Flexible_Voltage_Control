from Environment import *
from DDPG import *
from NN_Module import *
from config import Config

import numpy as np
from numpy import linalg as LA
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

from loguru import logger
from scipy.io import loadmat

### a simple logger
logger.remove()
logger.add(sys.stderr, level='DEBUG')

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
logger.info(f"Using {device} device")

### load real world data from mat files
p = loadmat('data/aggr_p.mat')
logger.debug(p)

q = loadmat('data/aggr_q.mat')
logger.debug(q)

pv_p = loadmat('data/PV.mat')
logger.debug(pv_p)



### Create power network and environment
seed = 0
torch.manual_seed(seed)

injection_bus = np.array([18, 21, 30, 45, 53])-1
pp_net = create_56bus()
env = VoltageCtrl_Env(pp_net, injection_bus)
state, topology, senario = env.reset_topo(seed=seed)
topology = torch.cuda.FloatTensor(topology).unsqueeze(0)

agent_num = len(injection_bus)

### load nn model parameter from saved model
agent_policy_net = []
for i in range(agent_num):
    topology_net = TopologyNet(topology_dim=55, output_dim=1, hidden_dim=Config.topology_hidden_dim)
    policy_net = FlexiblePolicyNet(env=env, topology_net=topology_net, obs_dim=1, action_dim=1, hidden_dim=Config.hidden_dim).to(device)
    agent_policy_net.append(policy_net)

for i in range(agent_num):
    #value_net_dict = torch.load(f'check_points/value_net/2023-06-19/Step_200_Seed_12_a{i}.pth')
    #policy_net_dict = torch.load(f'check_points/policy_net/2023-07-05/Step_300_Seed_45_a{i}.pth')
    #policy_net_dict = torch.load(f'check_points/policy_net/2023-08-09/Step_250_Seed_23_a{i}.pth')
    policy_net_dict = torch.load(f'check_points/policy_net/2023-08-15/Step_900_Seed_33_a{i}.pth')

    agent_policy_net[i].load_state_dict(policy_net_dict)


### test with no action
state = env.reset0()
last_action = np.zeros((agent_num,1))
action_list=[]
state_list =[]
state_list.append(state)

logger.info('-----start no action testing-----')

for step in tqdm(range(p.shape[0])):
    action = np.zeros((agent_num,1))
    next_state, reward, reward_sep, done = env.step_load(action, p[step],q[step],pv_p[step])
    action_list.append(action)
    state_list.append(next_state)
    last_action = np.copy(action)
    state = next_state
fig, axs = plt.subplots(1, 3, figsize=(12,4))
plt.gcf().subplots_adjust(wspace=0.4)
plt.gcf().subplots_adjust(bottom=0.18)
axs[0].plot(range(len(action_list)), p[:len(action_list)], label = f'Active Load', linewidth=1.5)
axs[0].plot(range(len(action_list)), q[:len(action_list)], label = f'Reactive Load', linewidth=1.5)
axs[0].plot(range(len(action_list)), pv_p[:len(action_list)], label = f'Solar', linewidth=1.5)

for i in range(agent_num):    
    dps = axs[1].plot(range(len(action_list)), np.array(state_list)[:len(action_list),i], label = f'Bus {injection_bus[i]}', linewidth=1.5)

axs[1].plot(range(len(action_list)), [0.95]*len(action_list), '--', color='k', linewidth=1)
axs[1].plot(range(len(action_list)), [1.05]*len(action_list), '--', color='k', linewidth=1)


### tset with flexible RL controller
state = env.reset0()
last_action = np.zeros((agent_num,1))
action_list=[]
state_list =[]
state_list.append(state)

logger.info('-----start Flexible RL controller testing-----')
for step in tqdm(range(p.shape[0])):
    action = []
    for i in range(agent_num):
        # sample action according to the current policy and exploration noise
        action_agent = agent_policy_net[i](torch.cuda.FloatTensor(state[i].reshape(1,)).unsqueeze(0), topology)
        action_agent = action_agent.detach().cpu().numpy()[0]
        action.append(action_agent)

    # PI policy    
    action = last_action - np.asarray(action)

    # execute action a_t and observe reward r_t and observe next state s_{t+1}
    next_state, reward, reward_sep, done = env.step_load(action, p[step],q[step],pv_p[step])

    action_list.append(action)
    state_list.append(next_state)
    last_action = np.copy(action)
    state = next_state

for i in range(agent_num):    
    safes=axs[2].plot(range(len(action_list)), np.array(state_list)[:len(action_list),i], label = f'Bus {injection_bus[i]}', linewidth=1.5)
axs[2].plot(range(len(action_list)), [0.95]*len(action_list), '--', color='k', linewidth=1)
axs[2].plot(range(len(action_list)), [1.05]*len(action_list), '--', color='k', linewidth=1)

### plot figure
axs[0].legend(loc='upper left', prop={"size":10})

axs[0].set_xlabel('Time (Hour)')   
axs[1].set_xlabel('Time (Hour)')  
axs[2].set_xlabel('Time (Hour)')  
# axs[2].get_yaxis().set_visible(False)
axs[1].set_yticks([0.95,1.00,1.05,1.10])
axs[1].set_yticklabels(['0.95','1.00','1.05','1.10'])
axs[2].set_yticks([0.95,1.00,1.05,1.10])
axs[2].set_yticklabels(['0.95','1.00','1.05','1.10'])
axs[0].set_xticks(np.arange(0,len(action_list),21600))
axs[0].set_xticklabels(['00:00','06:00','12:00','18:00','24:00'], fontsize=13)
axs[1].set_xticks(np.arange(0,len(action_list),21600))
axs[1].set_xticklabels(['00:00','06:00','12:00','18:00','24:00'], fontsize=13)
axs[2].set_xticks(np.arange(0,len(action_list),21600))
axs[2].set_xticklabels(['00:00','06:00','12:00','18:00','24:00'], fontsize=13)
axs[0].set_ylabel('Power (MW/MVar)', fontsize=15)   
axs[1].set_ylabel('Bus voltage (p.u.)', fontsize=15)  
axs[2].set_ylabel('Bus voltage (p.u.)', fontsize=15)  
plt.show()