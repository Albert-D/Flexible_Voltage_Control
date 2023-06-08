import torch
import sys
from loguru import logger

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import random
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

logger.remove()
logger.add(sys.stderr, level='TRACE')

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

    def forward(self, state, topolgy):
        logger.trace('input dimansion of nn are: {},{}',state.shape,topolgy.shape)
        topolgy = F.normalize(topolgy, dim=1) * 0.05

        input = torch.cat((state,topolgy), dim=1)

        state.requires_grad = True
        topolgy.requires_grad = True
        x = torch.relu(self.linear1(input))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0]
    

# monotone policy network with dead-band between [v_min, v_max]
class SafePolicyNetwork(nn.Module):
    def __init__(self, env, obs_dim, action_dim, hidden_dim, scale = 0.156, init_w=3e-3):
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
        
        self.w_minus=torch.matmul(-torch.square(self.z), self.w_recover)
        
        b = self.b.data
        b = b.clamp(min=0)
        b = self.scale*b/torch.norm(b, 1)
        self.b.data = b
        
        c = self.c.data
        c = c.clamp(min=0)
        c = self.scale*c/torch.norm(c, 1)
        self.c.data = c
        
        self.b_plus=torch.matmul(-self.b, self.b_recover) 
        self.b_minus=torch.matmul(-self.c, self.b_recover) 
        # self.b_plus=torch.matmul(-self.b, self.b_recover) - torch.tensor(self.env.vmax-0.02)
        # self.b_minus=torch.matmul(-self.c, self.b_recover) + torch.tensor(self.env.vmin+0.02)

        self.nonlinear_plus = torch.matmul(F.relu(torch.matmul(state, self.select_w)
                                                  + self.b_plus.view(1, self.hidden_dim)),
                                           torch.transpose(self.w_plus, 0, 1))
        
        self.nonlinear_minus = torch.matmul(F.relu(torch.matmul(state, self.select_wneg)
                                                   + self.b_minus.view(1, self.hidden_dim)),
                                            torch.transpose(self.w_minus, 0, 1))
        
        x = (self.nonlinear_plus+self.nonlinear_minus) 
        
        return self.nonlinear_minus
    
# define a sub-NN to hanlde topology information
class TopologyNet(nn.Module):
    def __init__(self, topology_dim, output_dim, hidden_dim, init_w=0.1):
        super(TopologyNet, self).__init__()

        self.linear1 = nn.Linear(topology_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

        self.linear1.weight.data.uniform_(0, init_w)
        self.linear2.weight.data.uniform_(0, init_w)
        self.linear3.weight.data.uniform_(0, init_w)

    def forward(self, topology):
        topology.requires_grad = True

        # x = nn.BatchNorm2d(topology)
        x = torch.relu(self.linear1(topology))
        x = torch.relu(self.linear2(x))
        x = F.sigmoid(self.linear3(x))

        return x


# define flexible safety policy network (our policy)
class FlexiblePolicyNet(nn.Module):
    def __init__(self, env, topology_net, obs_dim, action_dim, hidden_dim, scale=0.25, init_w=3e-3):
        super(FlexiblePolicyNet, self).__init__()

        self.env = env
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.topology_net = topology_net
        self.scale = scale

        # this matrix used to guarantee the sum of w matrix is postive
        self.w_triangle = torch.ones((self.hidden_dim, self.hidden_dim))
        self.w_triangle = -torch.triu(self.w_triangle, diagonal=0) + torch.triu(self.w_triangle, diagonal=2)\
                        + 2*torch.eye(self.hidden_dim)

        # this matrix used to guarantee b_i >= b_{i-1}
        self.b_triangle = torch.ones((self.hidden_dim, self.hidden_dim))
        self.b_triangle = torch.triu(self.b_triangle, diagonal=1)

        #define parameters of NN
        self.b = torch.rand(self.hidden_dim)
        self.b = (self.b/torch.sum(self.b))*self.scale
        self.b = torch.nn.Parameter(self.b, requires_grad=True)
        
        self.c = torch.rand(self.hidden_dim)
        self.c = (self.c/torch.sum(self.c))*self.scale
        self.c = torch.nn.Parameter(self.c, requires_grad=True)
        
        self.q = torch.nn.Parameter(torch.rand(action_dim, self.hidden_dim), requires_grad=True)
        self.z = torch.nn.Parameter(torch.rand(action_dim, self.hidden_dim), requires_grad=True)

    def forward(self, state, topology):
        #logger.trace('input dimansion of nn are: {},{}',state.shape,topology.shape)
        topology = F.normalize(topology, dim=1)
        y1 = self.topology_net(topology)

        # input = torch.cat((state,topology), dim=1)
        input = state
        input_dim = input.size(dim=1)

        self.w_plus = torch.square(self.q) @ self.w_triangle
        self.w_minus = -torch.square(self.q) @ self.w_triangle

        # self.b.data = self.b.data.clamp(min=0)
        # self.c.data = self.c.data.clamp(min=0)
        self.b.data = self.b.data.clamp(min=0) / torch.norm(self.b.data, 1) * self.scale
        self.c.data = self.c.data.clamp(min=0) / torch.norm(self.c.data, 1) * self.scale

        self.b_plus=torch.matmul(-self.b, self.b_triangle) - torch.tensor(self.env.vmax - 0.02)
        self.b_minus=torch.matmul(-self.b, self.b_triangle) + torch.tensor(self.env.vmin + 0.02)

        self.nonlinear_plus = F.relu(input @ (torch.ones(input_dim, self.hidden_dim)) + 
                                self.b_plus.view(1, self.hidden_dim)) @ self.w_plus.t()
        self.nonlinear_minus = F.relu(input @ (-torch.ones(input_dim, self.hidden_dim)) + 
                                self.b_minus.view(1, self.hidden_dim)) @ self.w_minus.t()
        
        y2 = self.nonlinear_plus + self.nonlinear_minus

        y = y1 * y2

        return y
    
def plot_net(net, topology):
    fig, ax = plt.subplots()
    N = 40
    s_array = np.zeros(N,)
    a_array_baseline = np.zeros(N,)
    a_array = np.zeros(N,)

    for j in range(N):
        state = torch.tensor([[0.8+0.01*j]])
        s_array[j] = state
        action_baseline = (np.maximum(state.cpu()-1.05, 0)-np.maximum(0.95-state.cpu(), 0)).reshape((1,))

        action = net(state, topology)
        action = action.detach().cpu().numpy()[0]

        a_array_baseline[j] = -action_baseline[0]
        a_array[j] = -action
    
    ax.plot(s_array, a_array_baseline, label = 'Baseline')
    ax.plot(s_array, a_array, label = 'RL')
    plt.show()

def plot_safe_net(net):
    fig, ax = plt.subplots()
    N = 40
    s_array = np.zeros(N,)
    a_array_baseline = np.zeros(N,)
    a_array = np.zeros(N,)

    for j in range(N):
        state = torch.tensor([[0.8+0.01*j]])
        s_array[j] = state
        action_baseline = (np.maximum(state.cpu()-1.05, 0)-np.maximum(0.95-state.cpu(), 0)).reshape((1,))

        action = net(state)
        action = action.detach().cpu().numpy()[0]

        a_array_baseline[j] = -action_baseline[0]
        a_array[j] = -action
    
    ax.plot(s_array, a_array_baseline, label = 'Baseline')
    ax.plot(s_array, a_array, label = 'RL')
    plt.show()

if __name__ == "__main__":

    logger.info(f"Using {device} device")

    from Environment import *

    injection_bus = np.array([18, 21, 30, 45, 53])-1
    pp_net = create_56bus()
    env = VoltageCtrl_Env(pp_net, injection_bus)
    state, topology, senario = env.reset()

    topology = torch.cuda.FloatTensor(topology).unsqueeze(0)
    topology = topology.expand(64,55)
    logger.info(topology.shape)
    state = torch.cuda.FloatTensor(state[0].reshape(1,)).unsqueeze(0)
    state = state.expand(64,1)
    logger.info(state.shape)

    topology_net = TopologyNet(topology_dim=55, output_dim=1, hidden_dim=100)
    net=FlexiblePolicyNet(env=env,topology_net=topology_net,action_dim=env.action_dim,obs_dim=env.obs_dim,hidden_dim=512)
    safe_net=SafePolicyNetwork(env=env,action_dim=env.action_dim,obs_dim=env.obs_dim,hidden_dim=100)

    y = net( torch.cuda.FloatTensor([[1]]), topology)
    
    # logger.info(y)
    #plot_safe_net(safe_net)
    plot_net(net, topology)

    logger.success(y)

    # for i in range(5):
    #     torch.manual_seed(i)
    #     topology_net = TopologyNet(topology_dim=55, output_dim=1, hidden_dim=50)
    #     net = FlexiblePolicyNet(env=env,topology_net=topology_net, action_dim=env.action_dim, obs_dim=env.obs_dim, hidden_dim=100)
    #     plot_net(net, topology)