## Twin Delay Training

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
from loguru import logger

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)


class TD3(object): 
    def __init__(self, policy_net, value_net, target_policy_net, target_value_net,
                 value_lr=1e-4, policy_lr=1e-4, max_action=0.3):
        self.policy_net = policy_net
        self.target_policy_net = target_policy_net
        # self.target_policy_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net_optimizer = optim.Adam(self.policy_net.parameters(), lr=value_lr)

        self.value_net = value_net
        self.target_value_net = target_value_net
        # self.target_value_net.load_state_dict(self.value_net.state_dict())
        self.value_net_optimizer = optim.Adam(self.value_net.parameters(), lr=policy_lr)

        self.max_action = max_action
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_net(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, \
              tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.total_it += 1

        # Sample replay buffer 
        state, topology, action, last_action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(device)
        topology = torch.FloatTensor(topology).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        last_action = torch.FloatTensor(last_action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        # Select action according to policy and add clipped noise 
        # noise = torch.cuda.FloatTensor(action).data.normal_(0, policy_noise).to(device)   # may change original action value, do not use this
        noise = torch.randn_like(action).data.normal_(0, policy_noise).to(device)
        noise = noise.clamp(-noise_clip, noise_clip)
        next_action = (self.target_policy_net(next_state, topology) + noise).clamp(-self.max_action, self.max_action)
        next_action = action - next_action  # remember that the output of policy_net is /dot(q)

        # Compute the target Q value
        target_Q1, target_Q2 = self.target_value_net(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + ((1.0-done) * discount * target_Q).detach() # be careful that our reward is negative cost

        # Get current Q estimates
        current_Q1, current_Q2 = self.value_net(state, action)

        # Compute value_net loss
        value_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

        # Optimize the value_net
        self.value_net_optimizer.zero_grad()
        value_loss.backward()
        self.value_net_optimizer.step()

        # Delayed policy updates
        if self.total_it % policy_freq == 0:

            # Compute policy_net loss
            policy_loss = -self.value_net.Q1(state, self.policy_net(state, topology)).mean()
            # Optimize the policy_net 
            self.policy_net_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_net_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.value_net.parameters(), self.target_value_net.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.policy_net.parameters(), self.target_policy_net.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    