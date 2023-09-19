from Environment import *
from DDPG import *
from NN_Module import *
from config import Config

import torch
import matplotlib.pyplot as plt

from loguru import logger

### a simple logger
logger.remove()
logger.add(sys.stderr, level='DEBUG')

env_seed = 8        #10-h  5-h 0-l 1-h 2-l 3-l 4l 7h 8h 9l

agent_policy_net = []
safe_agent_net = []

### create testing environment
injection_bus = np.array([9, 10, 15, 19, 32, 35, 47, 58, 65, 74, 82, 91, 103, 60]) #11, 36, 75,/ 1,5,9
pp_net = create_123bus()
env = Env_123bus(pp_net, injection_bus)
state, topology, senario = env.reset_topo(seed=env_seed)
topology = torch.cuda.FloatTensor(topology).unsqueeze(0)
agent_num = env.agentnum

# plot policy
def plot_policy(policy_net, topology):
    fig, axs = plt.subplots(2, 7, figsize=(15,6))
    title = ['Bus 9', 'Bus 10', 'Bus 15', 'Bus19', 'Bus 32', 'Bus 35', 'Bus 47', 
                'Bus 58', 'Bus 65', 'Bus 74', 'Bus 72', 'Bus 91', 'Bus 103', 'Bus 60']
    for i in range(agent_num):
        axs[i//7][i%7].clear()
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

        axs[i//7][i%7].plot(12*s_array, 30*a_array_baseline, '-.', label = 'Linear')
        axs[i//7][i%7].plot(12*s_array, a_array, label = 'Flexible-DDPG')
        axs[i//7][i%7].set_title(title[i])
        axs[i//7][i%7].legend(loc='lower left')

def plot_safe_net(net):
    fig, axs = plt.subplots(2, 7, figsize=(15,6))
    title = ['Bus 9', 'Bus 10', 'Bus 15', 'Bus19', 'Bus 32', 'Bus 35', 'Bus 47', 
                'Bus 58', 'Bus 65', 'Bus 74', 'Bus 72', 'Bus 91', 'Bus 103', 'Bus 60']
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

        axs[i//7][i%7].plot(12*s_array, 2*a_array_baseline, '-.', label = 'Linear')
        axs[i//7][i%7].plot(12*s_array, a_array, label = 'Stable-DDPG')
        axs[i//7][i%7].legend(loc='lower left')

def plot_x_policy(policy_net, topology):
    fig, axs = plt.subplots()
    for i in range(5):
        # plot policy
        N = 40
        s_array = np.zeros(N,)
        
        a_array_baseline = np.zeros(N,)
        a_array = np.zeros(N,)
        #topology = torch.cuda.FloatTensor(env.topology_init * np.random.uniform(0.7,1.3)).unsqueeze(0)
        state, topology, senario = env.reset_topo(seed=i)
        topology = torch.cuda.FloatTensor(topology).unsqueeze(0)
        
        for j in range(N):
            state = torch.tensor([[0.80+0.01*j]])
            s_array[j] = state

            action_baseline = (np.maximum(state.cpu()-1.05, 0)-np.maximum(0.95-state.cpu(), 0)).reshape((1,))
        
            action = policy_net[3](state, topology)
            action = action.detach().cpu().numpy()[0]
            
            a_array_baseline[j] = -action_baseline[0]
            a_array[j] = -action

        axs.plot(12*s_array, 30*a_array_baseline, '-.', label = 'Linear')
        axs.plot(12*s_array, a_array, label = 'Flexible-DDPG')
        axs.legend(loc='lower left')
        plt.pause(0.1)

### load nn model parameter from saved model 
for i in range(agent_num):
    topology_net = TopologyNet(topology_dim=env.topology_dim, output_dim=1, hidden_dim=Config.topology_hidden_dim)
    policy_net = FlexiblePolicyNet(env=env, topology_net=topology_net, obs_dim=1, action_dim=1, hidden_dim=Config.hidden_dim).to(device)
    agent_policy_net.append(policy_net)

for i in range(agent_num):
    policy_net = SafePolicyNetwork(env=env, obs_dim=1, action_dim=1, hidden_dim=100).to(device)
    safe_agent_net.append(policy_net)

for i in range(agent_num):
    #value_net_dict = torch.load(f'check_points/value_net/2023-06-19/Step_200_Seed_12_a{i}.pth')
    policy_net_dict = torch.load(f'check_points/policy_net/2023-09-12/Step_700_Seed_18_a{i}.pth')

    agent_policy_net[i].load_state_dict(policy_net_dict)

for i in range(agent_num):
    #value_net_dict = torch.load(f'D:/Code/Python/StableRL_VoltageCtrl-main/saved_models/2023-06-19/SafeDDPG_value_Step_200_a{i}.pth')
    policy_net_dict = torch.load(f'D:/Code/Python/Stable-DDPG-for-voltage-control/checkpoints/single-phase/123bus/safe-ddpg/policy_net_checkpoint_a{i}.pth')

    safe_agent_net[i].load_state_dict(policy_net_dict)

plot_policy(agent_policy_net, topology)
plot_x_policy(agent_policy_net, topology)
plot_safe_net(safe_agent_net)

### test our policy net

logger.success('begin to test flexible RL controller...')

episode_reward = 0
episode_control = 0
voltage = []
q = []
cost = []

last_action = np.zeros((agent_num,1))

done_record = True
state, topology, senario = env.reset_topo(seed=env_seed)
topology = torch.cuda.FloatTensor(topology).unsqueeze(0)
for t in range(100):
    action = []
    for i in range(agent_num):
        action_agent = agent_policy_net[i](torch.cuda.FloatTensor(state[i].reshape(1,)).unsqueeze(0), topology)
        action_agent = action_agent.detach().cpu().numpy()[0]
        action.append(action_agent)

    if np.min(action) < -0.3 or np.max(action) > 0.3:
        logger.warning('control output saturated! min is {}, max is {}', np.min(action), np.max(action))

    action = last_action -np.asarray(action)
    
    last_action = np.copy(action)
    
    try:
        next_state, reward, done = env.step(action)
    except:
        logger.error(sys.exc_info())
        logger.error('power flow not converge at {}', t)
        break

    if done and done_record:
        logger.success('flexible contorller stable at step {}', t)
        logger.info('stable cost is {}', episode_control)
        done_record = False

    voltage.append(state)

    q.append(action)

    state = next_state
    
    episode_reward += reward
    
    cost.append(-reward)
    
    episode_control += LA.norm(action, 2)

    # if done:
    #     break

voltage_RL = np.asarray(voltage)
q_RL =  np.asarray(q)
cost_RL =  np.asarray(cost)
logger.info('control cost of flexible controller is {}',episode_control)

### test the base line controller
logger.success('now begin to test linear controller...')

state, topology, senario = env.reset_topo(seed=env_seed)
episode_reward = 0
episode_control = 0
num_agent = env.agentnum
voltage = []
q = []
cost = []

last_action = np.zeros((num_agent,1))
done_record = True
for t in range(100):
    state1 = np.asarray(state-env.vmax)
    state2 = np.asarray(env.vmin-state)
    d_v = (np.maximum(state1, 0)-np.maximum(state2, 0)).reshape((num_agent,1))
    
    action = (last_action - 30*d_v)
    
    last_action = np.copy(action)
    
    try:
        next_state, reward, done = env.step(action)
    except:
        logger.error(sys.exc_info())
        logger.error('power flow not converge at {}', t)
        break

    if done and done_record:
        logger.success('linear controller stable at step {}', t)
        logger.info('stable cost is {}', episode_control)
        done_record = False

    voltage.append(state)

    q.append(action)

    state = next_state
    
    episode_reward += reward
    
    cost.append(-reward)
    
    episode_control += LA.norm(action, 2)

    # if done:
    #     break

voltage_baseline = np.asarray(voltage)
q_baseline =  np.asarray(q)
cost_baseline =  np.asarray(cost)
logger.info('control cost of linear controller is {}',episode_control)

# ### test the safe policy net
logger.success('now begin to test safe-ddpg controller...')

state,topology,senario = env.reset_topo(seed=env_seed)
episode_reward = 0
episode_control = 0
num_agent = env.agentnum
safe_voltage = []
safe_q = []
safe_cost = []

last_action = np.zeros((num_agent,1))
done_record = True
for t in range(100):
    action = []
    for i in range(num_agent):
        action_agent = safe_agent_net[i].get_action(torch.cuda.FloatTensor([state[i]]).float().reshape(1,1))
        action.append(action_agent)

    if 5*np.min(action) < -0.3 or 5*np.max(action) > 0.3:
        logger.warning('control output saturated! min is {}, max is {}', 5*np.min(action), 5*np.max(action))
    
    action = last_action - 10*np.asarray(action).reshape((num_agent, 1))
    
    last_action = np.copy(action)
    
    try:
        next_state, reward, done = env.step(action)
    except:
        logger.error(sys.exc_info())
        logger.error('power flow not converge at {}', t)
        break

    if done and done_record:
        logger.success('Safe-DDPG stable at step {}', t)
        logger.info('Safe-DDPG stable cost is {}', episode_control)
        done_record = False

    safe_voltage.append(state)

    safe_q.append(action)

    state = next_state
    
    episode_reward += reward
    
    safe_cost.append(-reward)
    
    episode_control += LA.norm(action, 2)

    # if done:
    #     break

safe_voltage = np.asarray(safe_voltage)
safe_q =  np.asarray(safe_q)
safe_cost =  np.asarray(safe_cost)
logger.info('control cost of safe-DDPG is {}',episode_control)


fig, ax = plt.subplots()
title = ['Bus 9', 'Bus 10', 'Bus 15', 'Bus19', 'Bus 32', 'Bus 35', 'Bus 47', 
                'Bus 58', 'Bus 65', 'Bus 74', 'Bus 72', 'Bus 91', 'Bus 103', 'Bus 60']
for i in range(agent_num):
    ax.plot(q_RL[:,i], label = title[i])
    # ax.plot(safe_q[:,i], '-.', label = title[i])
    ax.plot(q_baseline[:,i], '--')
ax.legend(loc = 'upper right')
plt.title('Controller Output')

fig, ax = plt.subplots()
ax.plot(cost_RL, label = 'RL')
ax.plot(cost_baseline, label = 'Linear')
# ax.plot(safe_cost, label = 'SafeRL')
ax.legend(loc = 'upper right')
plt.title('Cost with Voltage and Q')

#plt.show()

index = [1,2,3,4,5,6,7,8,9,10,11,12,13] 
labels = ['Bus 18 (Linear)', 'Bus 9 (Flexible-DDPG)', 'Bus 18 (safe-DDPG)',
          'Bus 21 (Linear)', 'Bus 10 (Flexible-DDPG)', 'Bus 21 (safe-DDPG)',
          'Bus 30 (Linear)', 'Bus 15 (Flexible-DDPG)', 'Bus 30 (safe-DDPG)',
          'Bus 45 (Linear)', 'Bus 19 (Flexible-DDPG)', 'Bus 45 (safe-DDPG)',
          'Bus 53 (Linear)', 'Bus 32 (Flexible-DDPG)', 'Bus 53 (safe-DDPG)',
          'Bus 18 (Linear)', 'Bus 35 (Flexible-DDPG)', 'Bus 18 (safe-DDPG)',
          'Bus 21 (Linear)', 'Bus 47 (Flexible-DDPG)', 'Bus 21 (safe-DDPG)',
          'Bus 30 (Linear)', 'Bus 58 (Flexible-DDPG)', 'Bus 30 (safe-DDPG)',
          'Bus 45 (Linear)', 'Bus 65 (Flexible-DDPG)', 'Bus 45 (safe-DDPG)',
          'Bus 53 (Linear)', 'Bus 74 (Flexible-DDPG)', 'Bus 53 (safe-DDPG)',
          'Bus 18 (Linear)', 'Bus 82 (Flexible-DDPG)', 'Bus 18 (safe-DDPG)',
          'Bus 21 (Linear)', 'Bus 91 (Flexible-DDPG)', 'Bus 21 (safe-DDPG)',
          'Bus 30 (Linear)', 'Bus 103 (Flexible-DDPG)', 'Bus 30 (safe-DDPG)',
          'Bus 45 (Linear)', 'Bus 60 (Flexible-DDPG)', 'Bus 45 (safe-DDPG)',
          ]
colors = ['b', 'g', 'r', 'c','m', 'y', 'darkblue','chocolate','purple',
          'deeppink','gold','lavenderblush','lightseagreen','olive']

f = plt.figure(figsize=(8, 8))
ax = f.add_subplot(111)

for i in range(len(index)):
    ax.plot(12*voltage_RL[:, index[i]], color = colors[i], label = labels[3*i+1])
    ax.plot(12*voltage_baseline[:, index[i]], '-.', color = colors[i], label = labels[3*i])
    ax.plot(12*safe_voltage[:, index[i]], '--', color = colors[i], label = labels[3*i+2])

ax.legend(loc = 'upper right')

state,topology,senario = env.reset_topo(seed=env_seed)
logger.debug(senario)
ax.axhline(y=12.0, color='r')

if senario == 0:
    ax.set_ylim([10, 12.6])
    ax.axhline(y=11.4, color='k', linestyle='--', label = 'Lower bound')
elif senario == 1:
    ax.set_ylim([11.4, 14])
    ax.axhline(y=12.6, color='k', linestyle='--', label = 'Upper bound')
else:
    ax.set_ylim([10, 14])
    ax.axhline(y=12.6, color='k', linestyle='--', label = 'Upper bound')
    ax.axhline(y=11.4, color='k', linestyle='--', label = 'Lower bound')

ax.set_xlim([0, 100])
# ax.set_yticks([10, 11, 12, 13, 14])
ax.set_ylabel('Bus voltage (kV)')
ax.set_xlabel('Iteration Steps')
ax.grid()

plt.show()
