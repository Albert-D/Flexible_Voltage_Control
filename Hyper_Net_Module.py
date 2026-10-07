import torch
import sys
from loguru import logger
from config import Config

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

class MainPolicyNet(nn.Module):
    def __init__(self, env, obs_dim, action_dim, hidden_dim, scale, init_w):
        super(MainPolicyNet, self).__init__()

        self.env = env
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.scale = scale

        # this matrix used to guarantee the sum of w matrix is postive
        self.w_triangle = torch.ones((self.hidden_dim, self.hidden_dim))
        self.w_triangle = -torch.triu(self.w_triangle, diagonal=0) + torch.triu(self.w_triangle, diagonal=2)\
                        + 2*torch.eye(self.hidden_dim)

        # this matrix used to guarantee b_i >= b_{i-1}
        self.b_triangle = torch.ones((self.hidden_dim, self.hidden_dim))
        self.b_triangle = torch.triu(self.b_triangle, diagonal=1)

        #define parameters of NN
        self.b = torch.nn.Parameter(torch.rand(1,self.hidden_dim), requires_grad=True)
        self.c = torch.nn.Parameter(torch.rand(1,self.hidden_dim), requires_grad=True)
        self.q = torch.nn.Parameter(torch.rand(1,action_dim, self.hidden_dim), requires_grad=True)
        self.z = torch.nn.Parameter(torch.rand(1,action_dim, self.hidden_dim), requires_grad=True)

        self.b.data = (self.b.data / torch.sum(self.b.data)) * self.scale
        self.c.data = (self.c.data / torch.sum(self.c.data)) * self.scale

    def forward(self, state, b,c,q,z):
        # input = torch.cat((state,topology), dim=1)
        input = state
        
        # q_constrain = 0.1 + 10.0 * torch.sigmoid(q)
        # z_constrain = 0.1 + 10.0 * torch.sigmoid(z)

        # self.b.data = self.b.data.clamp(min=0)
        # self.c.data = self.c.data.clamp(min=0)
        self.b.data = self.b.data.clamp(min=0) / torch.norm(self.b.data, 1) * self.scale
        self.c.data = self.c.data.clamp(min=0) / torch.norm(self.c.data, 1) * self.scale
        # b = b.clamp(min=0) / torch.norm(b, 1) * self.scale
        # c = c.clamp(min=0) / torch.norm(c, 1) * self.scale

        self.w_plus = torch.square(self.q) @ self.w_triangle
        self.w_minus = -torch.square(self.z) @ self.w_triangle
        # self.w_plus = torch.square(q) @ self.w_triangle
        # self.w_minus = -torch.square(z) @ self.w_triangle
        # self.w_plus = q_constrain @ self.w_triangle
        # self.w_minus = -q_constrain @ self.w_triangle

        self.b_plus=torch.matmul(-self.b, self.b_triangle) - torch.tensor(self.env.vmax - 0.005)
        self.b_minus=torch.matmul(-self.c, self.b_triangle) + torch.tensor(self.env.vmin + 0.005)
        # self.b_plus=torch.matmul(-b, self.b_triangle) - torch.tensor(self.env.vmax - 0.005)
        # self.b_minus=torch.matmul(-c, self.b_triangle) + torch.tensor(self.env.vmin + 0.005)

        self.nonlinear_plus = F.relu(input.unsqueeze(2) + self.b_plus.unsqueeze(1)) @ self.w_plus.transpose(1,2)
        self.nonlinear_minus = F.relu(-input.unsqueeze(2) + self.b_minus.unsqueeze(1)) @ self.w_minus.transpose(1,2)

        # logger.debug("b={}",b)
        # logger.debug("c={}",self.b_minus.detach())
        # logger.debug("q={}",self.w_plus.detach())
        # logger.debug("z={}",self.w_minus.detach())
        
        y = self.nonlinear_plus + self.nonlinear_minus
        # logger.debug("plus={}, minus={}.",self.nonlinear_plus, self.nonlinear_minus)

        return y.squeeze(-1)
    

class HyperXnet(nn.Module):
    def __init__(self, topology_dim, x_hidden_dim, hidden_dim, action_dim, init_w):
        super(HyperXnet,self).__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.linear1 = nn.Linear(topology_dim, x_hidden_dim)
        self.linear2 = nn.Linear(x_hidden_dim, x_hidden_dim)
        self.linear3 = nn.Linear(x_hidden_dim, hidden_dim * 2 + action_dim * hidden_dim * 2)

        # He initialization for ReLU activations
        nn.init.uniform_(self.linear1.weight, 0, init_w)
        nn.init.uniform_(self.linear2.weight, 0, init_w)
        nn.init.uniform_(self.linear3.weight, 0, init_w)

    def forward(self, topology):
        
        # topology.requires_grad = True

        # x = nn.BatchNorm2d(topology)
        x = F.normalize(topology, p=2, dim=1)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)

        # Split the output into the respective sizes
        b = x[:, :self.hidden_dim]
        c = x[:, self.hidden_dim:2*self.hidden_dim]
        q = x[:, 2*self.hidden_dim:2*self.hidden_dim + self.action_dim*self.hidden_dim]
        z = x[:, 2*self.hidden_dim + self.action_dim*self.hidden_dim:]
        
        # Reshape q and z to match the required dimensions
        q = q.reshape(-1, self.action_dim, self.hidden_dim)
        z = z.reshape(-1, self.action_dim, self.hidden_dim)
        
        return b, c, q, z
    

class HyperFlexibleNet(nn.Module):
    def __init__(self, env, obs_dim, topology_dim, x_hidden_dim, hidden_dim, action_dim, scale=0.25, init_w=0.01):
        super(HyperFlexibleNet,self).__init__()
        self.hyper_x_net = HyperXnet(topology_dim, x_hidden_dim, hidden_dim, action_dim, init_w)
        self.policy_net = MainPolicyNet(env, obs_dim, action_dim, hidden_dim, scale, init_w)

    def forward(self, state, topolgy):
        b, c, q, z = self.hyper_x_net(topolgy)

        # Set the parameters of policy_net dynamically
        # self.policy_net.b = nn.Parameter(b, requires_grad=True)

        # self.policy_net.c = nn.Parameter(c, requires_grad=True)
        # self.policy_net.q = nn.Parameter(q, requires_grad=True)
        # self.policy_net.z = nn.Parameter(z, requires_grad=True)

        y = self.policy_net(state, b, c, q, z)

        return y
    
def plot_hyper_net(net, topology):
    fig, ax = plt.subplots()
    
    s_array = np.arange(0.8,1.21,0.01)
    a_array_baseline = -(np.maximum(s_array-1.05, 0)-np.maximum(0.95-s_array, 0))

    state = torch.cuda.FloatTensor(s_array).unsqueeze(0)

    a_array = -net(state, topology).squeeze(0).detach().cpu().numpy()
    
    ax.plot(s_array, a_array_baseline, label = 'Baseline')
    ax.plot(s_array, a_array, label = 'RL')
    plt.show()



if __name__ == "__main__":

    logger.info(f"Using {device} device")

    from Environment import *

    injection_bus = np.array([9, 10, 15, 19, 32, 35, 47, 58, 65, 74, 82, 91, 103, 60]) #11, 36, 75,/ 1,5,9
    pp_net = create_123bus()
    test_env = Env_123bus(pp_net, injection_bus)
    state, topology, senario = test_env.reset_topo()

    topology = torch.cuda.FloatTensor(topology).unsqueeze(0)
    # topology = topology.expand(64,113)
    logger.info(topology.shape)

    # test_hypernet = HyperXnet(113, 1024, 512, 1,0.01)
    # b,c,q,z = test_hypernet(topology)
    # print(b,c.shape,q,z.shape)

    test_net = HyperFlexibleNet(test_env, test_env.obs_dim, 113, 1024, 50, test_env.action_dim)

    plot_hyper_net(test_net, topology)
    # for i in range(5):
    #     seed = i
    #     random.seed(seed)            # Python random module
    #     np.random.seed(seed)         # NumPy
    #     torch.manual_seed(seed)      # CPU-level seeding for PyTorch
    #     if torch.cuda.is_available():
    #         torch.cuda.manual_seed(seed)       # GPU-level seeding for PyTorch
    #         torch.cuda.manual_seed_all(seed)   # Include this if using more than one GPU
    #         torch.backends.cudnn.deterministic = True  # Use deterministic algorithms
    #         torch.backends.cudnn.benchmark = False     # Disable this if determinism is preferred over speed
    #     test_net = HyperFlexibleNet(test_env, test_env.obs_dim, 113, 1024, 100, test_env.action_dim)
    #     plot_hyper_net(test_net, topology)