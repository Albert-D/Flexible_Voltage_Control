from Environment import *
from DDPG import *
from NN_Module import *

import torch
import matplotlib.pyplot as plt

from loguru import logger

### a simple logger
logger.remove()
logger.add(sys.stderr, level='DEBUG')

env_seed = 2        #10-h  5-h 0-l 1-h 2-l 3-l 4l 7h 8h 9l
episode_num = 100   # the total test episode
step_num = 300      # the longest test step

agent_num = 5
agent_policy_net = []
safe_agent_net = []

### create testing environment
injection_bus = np.array([18, 21, 30, 45, 53])-1
pp_net = create_56bus()
env = VoltageCtrl_Env(pp_net, injection_bus)
topology = env.network.line.x_ohm_per_km
topology = torch.cuda.FloatTensor(topology).unsqueeze(0)

# plot policy
def plot_policy(policy_net, topology):
    fig, axs = plt.subplots(1, 5, figsize=(15,3))
    title = ['Bus 18', 'Bus 21', 'Bus 30', 'Bus 45', 'Bus 53']
    for i in range(5):
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
        
            action = policy_net[i](state, topology)
            action = action.detach().cpu().numpy()[0]
            
            a_array_baseline[j] = -action_baseline[0]
            a_array[j] = -action

        axs[i].plot(12*s_array, 3*a_array_baseline, '-.', label = 'Linear')
        axs[i].plot(12*s_array, a_array, label = 'Flexible-DDPG')
        axs[i].set_title(title[i])
        axs[i].legend(loc='lower left')

def plot_safe_net(net):
    fig, axs = plt.subplots(1, 5, figsize=(15,3))
    title = ['Bus 18', 'Bus 21', 'Bus 30', 'Bus 45', 'Bus 53']
    for i in range(agent_num):
        N = 40
        s_array = np.zeros(N,)
        
        a_array_baseline = np.zeros(N,)
        a_array = np.zeros(N,)
        
        for j in range(N):
            state = np.array([0.8+0.01*j])
            s_array[j] = state

            action_baseline = (np.maximum(state-1.05, 0)-np.maximum(0.95-state, 0)).reshape((1,))
        
            action = net[i].get_action([state])
            
            a_array_baseline[j] = -action_baseline[0]
            a_array[j] = -action

        axs[i].plot(12*s_array, 2*a_array_baseline, '-.', label = 'Linear')
        axs[i].plot(12*s_array, a_array, label = 'Stable-DDPG')
        axs[i].legend(loc='lower left')

def plot_x_policy(policy_net, topology):
    fig, axs = plt.subplots()
    for i in range(5):
        # plot policy
        N = 40
        s_array = np.zeros(N,)
        
        a_array_baseline = np.zeros(N,)
        a_array = np.zeros(N,)
        topology = torch.cuda.FloatTensor(env.topology_init * np.random.uniform(0.7,1.3)).unsqueeze(0)
        
        for j in range(N):
            state = torch.tensor([[0.80+0.01*j]])
            s_array[j] = state

            action_baseline = (np.maximum(state.cpu()-1.05, 0)-np.maximum(0.95-state.cpu(), 0)).reshape((1,))
        
            action = policy_net[2](state, topology)
            action = action.detach().cpu().numpy()[0]
            
            a_array_baseline[j] = -action_baseline[0]
            a_array[j] = -action

        axs.plot(12*s_array, a_array_baseline, '-.', label = 'Linear')
        axs.plot(12*s_array, a_array, label = 'Flexible-DDPG')
        axs.legend(loc='lower left')
        plt.pause(0.1)

### load nn model parameter from saved model 
for i in range(agent_num):
    topology_net = TopologyNet(topology_dim=55, output_dim=1, hidden_dim=100)
    policy_net = FlexiblePolicyNet(env=env, topology_net=topology_net, obs_dim=1, action_dim=1, hidden_dim=512).to(device)
    agent_policy_net.append(policy_net)

for i in range(agent_num):
    policy_net = SafePolicyNetwork(env=env, obs_dim=1, action_dim=1, hidden_dim=100).to(device)
    safe_agent_net.append(policy_net)

for i in range(agent_num):
    #value_net_dict = torch.load(f'check_points/value_net/2023-06-19/Step_200_Seed_12_a{i}.pth')
    policy_net_dict = torch.load(f'check_points/policy_net/2023-07-03/Step_200_Seed_8_a{i}.pth')

    agent_policy_net[i].load_state_dict(policy_net_dict)

for i in range(agent_num):
    #value_net_dict = torch.load(f'D:/Code/Python/StableRL_VoltageCtrl-main/saved_models/2023-06-19/SafeDDPG_value_Step_200_a{i}.pth')
    policy_net_dict = torch.load(f'D:/Code/Python/StableRL_VoltageCtrl-main/saved_models/2023-06-19/SafeDDPG_policy_Step_200_a{i}.pth')

    safe_agent_net[i].load_state_dict(policy_net_dict)

### plot the controller
plot_policy(agent_policy_net, topology)
plot_x_policy(agent_policy_net, topology)
plot_safe_net(safe_agent_net)


### test our controller
episode_reward = 0
episode_control = 0
voltage = []
q = []
cost = []
success_list = []
fail_list = []

for episode in range(episode_num):
    state, topology, senario = env.reset(seed=episode)
    topology = torch.cuda.FloatTensor(topology).unsqueeze(0)
    last_action = np.zeros((agent_num,1))

    for step in range(step_num):
        action = []
        for i in range(agent_num):
            action_agent = agent_policy_net[i](torch.cuda.FloatTensor(state[i].reshape(1,)).unsqueeze(0), topology)
            action_agent = action_agent.detach().cpu().numpy()[0]
            action.append(action_agent)

        action = last_action - np.asarray(action)
        
        last_action = np.copy(action)
        
        try:
            next_state, reward, done = env.step(action)
        except:
            logger.error(sys.exc_info())
            logger.error('power flow not converge at epsisode{} step{}', episode, step)
            fail_list.append((episode,step))
            break

        if(np.min(next_state) < 0.75 or np.max(next_state) > 1.25): #if voltage violation > 25%, episode ends.
            logger.warning('episode {} step {} break, min_state is {}, max_state is {}', episode, step, np.min(next_state), np.max(next_state))
            fail_list.append((episode,step))
            break
        if done:
            success_list.append((episode,step))
            logger.success('stable at {}',success_list[-1])
            break

        voltage.append(state)

        q.append(action)

        state = next_state
        
        episode_reward += reward
        
        cost.append(-reward)
        
        episode_control += LA.norm(action, 2)


### test the base line controller
state, topology, senario = env.reset(seed=env_seed)
episode_reward = 0
episode_control = 0
num_agent = 5
voltage = []
q = []
cost = []

last_action = np.zeros((num_agent,1))

for t in range(100):
    state1 = np.asarray(state-env.vmax)
    state2 = np.asarray(env.vmin-state)
    d_v = (np.maximum(state1, 0)-np.maximum(state2, 0)).reshape((num_agent,1))
    
    action = (last_action - 2*d_v)
    
    last_action = np.copy(action)
    
    try:
        next_state, reward, done = env.step(action)
    except:
        logger.error(sys.exc_info())
        logger.error('power flow not converge at {}', i)
        break

    voltage.append(state)

    q.append(action)

    state = next_state
    
    episode_reward += reward
    
    cost.append(-reward)
    
    episode_control += LA.norm(action, 2)**2

voltage_baseline = np.asarray(voltage)
q_baseline =  np.asarray(q)
cost_baseline =  np.asarray(cost)

### test the safe policy net
state,topology,senario = env.reset(seed=env_seed)
episode_reward = 0
episode_control = 0
num_agent = 5
safe_voltage = []
safe_q = []
safe_cost = []

last_action = np.zeros((num_agent,1))

for t in range(100):
    action = []
    for i in range(num_agent):
        action_agent = safe_agent_net[i].get_action(torch.cuda.FloatTensor([state[i]]).float().reshape(1,1))
        action.append(action_agent)
    
    action = last_action - np.asarray(action).reshape((num_agent, 1))
    
    last_action = np.copy(action)
    
    try:
        next_state, reward, done = env.step(action)
    except pp.powerflow.LoadflowNotConverged:
        # logger.error(sys.exc_info())
        print('power flow not converge at step{}', step)
        abnormal_stop = True
        break

    safe_voltage.append(state)

    safe_q.append(action)

    state = next_state
    
    episode_reward += reward
    
    safe_cost.append(-reward)
    
    episode_control += LA.norm(action, 2)

safe_voltage = np.asarray(safe_voltage)
safe_q =  np.asarray(safe_q)
safe_cost =  np.asarray(safe_cost)

