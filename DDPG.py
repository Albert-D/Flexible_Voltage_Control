import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
print(f"Using {device} device")

# DPPG class
class DDPG:
    def __init__(self, policy_net, value_net, target_policy_net, target_value_net,
                 value_lr=2e-4, policy_lr=1e-4):
        
        self.policy_net = policy_net
        self.value_net = value_net
        self.target_policy_net = target_policy_net
        self.target_value_net = target_value_net
        
        self.value_lr = value_lr
        self.policy_lr = policy_lr
        
        self.value_optimizer = optim.Adam(value_net.parameters(),  lr=value_lr)
        self.policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)
        self.value_criterion = nn.MSELoss()

    def train_step(self, replay_buffer, batch_size,
                   gamma=0.99,
                   soft_tau=1e-2):

        state, action, last_action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        last_action = torch.FloatTensor(last_action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        next_action = action-self.target_policy_net(next_state)    
        target_value = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + gamma*(1.0-done)*target_value
        
        value = self.value_net(state, action)
        value_loss = self.value_criterion(value, expected_value.detach())
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
 
        self.value_optimizer.step()
        
        
        policy_loss = self.value_net(state, last_action-self.policy_net(state))
        policy_loss = -policy_loss.mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data*soft_tau
            )

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )


# value network
class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        self.linear1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class ReplayBufferPI:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, last_action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, last_action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, last_action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, last_action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
    

# standard ddpg policy network
class PolicyNetwork(nn.Module):
    def __init__(self, env, obs_dim, action_dim, hidden_dim, init_w=3e-3):
        super(PolicyNetwork, self).__init__()

        self.env = env
        self.linear1 = nn.Linear(obs_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, action_dim)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        state.requires_grad = True
        x = torch.relu(self.linear1(state))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0]
    

# monotone policy network with dead-band between [v_min, v_max]
class SafePolicyNetwork(nn.Module):
    def __init__(self, env, obs_dim, action_dim, hidden_dim, scale = 0.15, init_w=3e-3):
        super(SafePolicyNetwork, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.env = env
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.scale = scale
        
        #define weight and bias recover matrix
        self.w_recover = torch.ones((self.hidden_dim, self.hidden_dim))
        self.w_recover = -torch.triu(self.w_recover, diagonal=0)\
        +torch.triu(self.w_recover, diagonal=2)+2*torch.eye(self.hidden_dim)
        self.w_recover=self.w_recover.to(self.device)
        
        self.b_recover = torch.ones((self.hidden_dim, self.hidden_dim))
        self.b_recover = torch.triu(self.b_recover, diagonal=0)-torch.eye(self.hidden_dim)
        self.b_recover = self.b_recover.to(self.device)
        
        self.select_w = torch.ones(1, self.hidden_dim).to(self.device)
        self.select_wneg = -torch.ones(1, self.hidden_dim).to(self.device)
        
        # initialization
        self.b = torch.rand(self.hidden_dim).to(self.device)
        self.b = (self.b/torch.sum(self.b))*scale
        self.b = torch.nn.Parameter(self.b, requires_grad=True)
        
        self.c = torch.rand(self.hidden_dim).to(self.device)
        self.c = (self.c/torch.sum(self.c))*scale
        self.c = torch.nn.Parameter(self.c, requires_grad=True)
        
        self.q = torch.nn.Parameter(torch.rand(action_dim, self.hidden_dim).to(self.device), requires_grad=True)
        self.z = torch.nn.Parameter(torch.rand(action_dim, self.hidden_dim).to(self.device), requires_grad=True)
        
    def forward(self, state):
        self.w_plus=torch.matmul(torch.square(self.q), self.w_recover)
        
        self.w_minus=torch.matmul(-torch.square(self.q), self.w_recover)
        
        b = self.b.data
        b = b.clamp(min=0)
        b = self.scale*b/torch.norm(b, 1)
        self.b.data = b
        
        c = self.c.data
        c = c.clamp(min=0)
        c = self.scale*c/torch.norm(c, 1)
        self.c.data = c
        
        self.b_plus=torch.matmul(-self.b, self.b_recover) - torch.tensor(self.env.vmax-0.02)
        self.b_minus=torch.matmul(-self.b, self.b_recover) + torch.tensor(self.env.vmin+0.02)
        
        self.nonlinear_plus = torch.matmul(F.relu(torch.matmul(state, self.select_w)
                                                  + self.b_plus.view(1, self.hidden_dim)),
                                           torch.transpose(self.w_plus, 0, 1))
        
        self.nonlinear_minus = torch.matmul(F.relu(torch.matmul(state, self.select_wneg)
                                                   + self.b_minus.view(1, self.hidden_dim)),
                                            torch.transpose(self.w_minus, 0, 1))
        
        x = (self.nonlinear_plus+self.nonlinear_minus) 
        
        return x
    

# define flexible safety policy network (our policy)
class FlexiblePolicyNet(nn.Module):
    def __init__(self, env, obs_dim, action_dim, hidden_dim, init_w=3e-3):
        super(FlexiblePolicyNet, self).__init__()

        self.env = env
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        #this two upper trianglular matrixes are used to reconstruct parameters
        self.w_triangle = torch.ones((self.hidden_dim, self.hidden_dim))
        self.w_triangle = -torch.triu(self.w_triangle, diagonal=0) + torch.triu(self.w_triangle, diagonal=2)\
                        + 2*torch.eye(self.hidden_dim)

        self.b_triangle = torch.ones((self.hidden_dim, self.hidden_dim))
        self.b_triangle = torch.triu(self.b_triangle, diagonal=1)

        #define parameters of NN
        self.b = torch.rand(self.hidden_dim)
        self.b = (self.b/torch.sum(self.b))
        self.b = torch.nn.Parameter(self.b, requires_grad=True)
        
        self.c = torch.rand(self.hidden_dim)
        self.c = (self.c/torch.sum(self.c))
        self.c = torch.nn.Parameter(self.c, requires_grad=True)
        
        self.q = torch.nn.Parameter(torch.rand(action_dim, self.hidden_dim), requires_grad=True)
        self.z = torch.nn.Parameter(torch.rand(action_dim, self.hidden_dim), requires_grad=True)

    def forward(self, state, X):
        input = torch.cat((state,X), dim=0)
        input_dim = input.size(dim=0)

        self.w_plus=torch.matmul(torch.square(self.q), self.w_triangle)
        self.w_minus=torch.matmul(-torch.square(self.z), self.w_triangle)

        self.b.data = self.b.data.clamp(min=0)/torch.norm(self.b.data, 1)
        self.c.data = self.c.data.clamp(min=0)/torch.norm(self.c.data, 1)

        self.b_plus=torch.matmul(-self.b, self.b_triangle) - torch.tensor(self.env.vmax)
        self.b_minus=torch.matmul(-self.b, self.b_triangle) + torch.tensor(self.env.vmin)
        
        self.nonlinear_plus = F.relu(input @ torch.ones(input_dim, self.hidden_dim) + 
                                self.b_plus.view(1, self.hidden_dim)) @ self.w_plus.t()
        self.nonlinear_minus = F.relu(input @ (-torch.ones(input_dim, self.hidden_dim)) + 
                                self.b_plus.view(1, self.hidden_dim)) @ self.w_minus.t()
        
        y = (self.nonlinear_plus+self.nonlinear_minus) 

        return y
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        return action.detach().numpy()[0] #return action.detach().cpu().numpy()[0] 


if __name__ == "__main__":

    from Environment import *

    injection_bus = np.array([18, 21, 30, 45, 53])-1
    pp_net = create_56bus()
    env = VoltageCtrl_Env(pp_net, injection_bus)
    env.reset()

    net=FlexiblePolicyNet(env=env,action_dim=env.action_dim,obs_dim=env.obs_dim,hidden_dim=10)

    topology = torch.cuda.FloatTensor(pp_net.line.x_ohm_per_km)
    state = torch.cuda.FloatTensor(pp_net.res_bus.vm_pu[1].reshape(1,))
    print(state.shape)
    print(topology.shape)
    input = torch.cat((state, topology), dim=0)
    b_plus = torch.matmul(-net.b, net.b_triangle) - torch.tensor(env.vmax)
    d = input.size(dim=0)
    x = F.relu(input @ torch.ones(d, net.hidden_dim) + b_plus.view(1, net.hidden_dim)) @ torch.matmul(torch.square(net.q), net.w_triangle).t()
    print(x)

