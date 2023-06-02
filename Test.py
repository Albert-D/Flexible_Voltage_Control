from Environment import *
from DDPG import *
from NN_Module import *

import torch
import matplotlib.pyplot as plt

agent_num = 5
agent_policy_net = []

### create testing environment
injection_bus = np.array([18, 21, 30, 45, 53])-1
pp_net = create_56bus()
env = VoltageCtrl_Env(pp_net, injection_bus)

# plot policy
def plot_policy(policy_net, episode):
    s_array = np.zeros(30,)
    topology = env.network.line.x_ohm_per_km
    topology = torch.cuda.FloatTensor(topology).unsqueeze(0)

    a_array_baseline = np.zeros(30,)
    a_array = np.zeros(30,)
    for i in range(30):
        state = torch.tensor([[0.85+0.01*i]])
        s_array[i] = state

        action_baseline = -(np.maximum(state.cpu()-1.05, 0)-np.maximum(0.95-state.cpu(), 0)).reshape((1,))
        action = policy_net(state, topology)

        a_array_baseline[i] = action_baseline[0]
        a_array[i] = action[0]
        
    plt.figure() 
    plt.plot(s_array, a_array_baseline, label = 'Baseline')
    plt.plot(s_array, a_array, label = 'RL')
    plt.savefig('Policy{0}.png'.format(episode), dpi=100)

### load nn model parameter from saved model 
for i in range(agent_num):
    policy_net = FlexiblePolicyNet(env=env, obs_dim=1, action_dim=1, hidden_dim=100).to(device)
    agent_policy_net.append(policy_net)

for i in range(agent_num):
    value_net_dict = torch.load(f'saved_models/value_net/Step_400_Seed_10_a{i}.pth')
    policy_net_dict = torch.load(f'saved_models/policy_net/Step_400_Seed_10_a{i}.pth')

    agent_policy_net[i].load_state_dict(policy_net_dict)

plot_policy(agent_policy_net[1], 100)
plt.show()

state = env.reset(seed=10)
topology = env.network.line.x_ohm_per_km
print(state, topology)
topology = torch.cuda.FloatTensor(topology).unsqueeze(0)
episode_reward = 0
episode_control = 0
voltage = []
q = []
cost = []

last_action = np.zeros((agent_num,1))

for t in range(100):
    action = []
    for i in range(agent_num):
        action_agent = agent_policy_net[i](torch.cuda.FloatTensor(state[i].reshape(1,)).unsqueeze(0), topology)
        action_agent = action_agent.detach().cpu().numpy()[0]
        action.append(action_agent)
    
    print(action)
    action = last_action - np.asarray(action)
    
    last_action = np.copy(action)
    
    next_state, reward, done = env.step(action)

    voltage.append(state)

    q.append(action)

    state = next_state
    
    episode_reward += reward
    
    cost.append(-reward)
    
    episode_control += LA.norm(action, 2)**2

voltage_RL = np.asarray(voltage)
q_RL =  np.asarray(q)
cost_RL =  np.asarray(cost)
print(voltage_RL)
print(q)
print(cost_RL)

index = [0, 1, 3] 
labels = ['Bus 18 (Linear)', 'Bus 18 (Flexible-DDPG)', 
          'Bus 21 (Linear)', 'Bus 21 (Flexible-DDPG)',
          'Bus 45 (Linear)', 'Bus 45 (Flexible-DDPG)']
colors = ['b', 'r', 'c']

f = plt.figure(figsize=(4, 4))
ax = f.add_subplot(111)

for i in range(len(index)):
    ax.plot(12*voltage_RL[:, index[i]], color = colors[i], label = labels[2*i+1])

ax.legend(loc = 'upper right')
ax.axhline(y=12.6, color='k', linestyle='--', label = 'Upper bound')
ax.axhline(y=11.4, color='k', linestyle='--', label = 'Lower bound')
ax.set_xlim([0, 60])
ax.set_ylim([10, 14])
ax.set_yticks([10, 11, 12, 13, 14])
ax.set_ylabel('Bus voltage (kV)')
ax.set_xlabel('Iteration Steps')
ax.grid()

plt.show()
