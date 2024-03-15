import gymnasium as gym
import sys

import numpy as np
from numpy import linalg as LA

import pandapower as pp
import pandapower.networks as pn
import pandas as pd 

import torch
import torch.nn as nn
import torch.nn.functional as F

from gymnasium import spaces
from gymnasium.utils import seeding

from pandapower.plotting.plotly import simple_plotly
from pandapower.plotting.plotly import vlevel_plotly
from pandapower.plotting.plotly import pf_res_plotly

from loguru import logger
logger.remove()
logger.add(sys.stderr, level='DEBUG')

from config import Config

# some hyperparameter in traning
cost_w_a = Config.cost_g_a       # weight of action cost
cost_w_v = Config.cost_g_v     # weight of voltage error cost
cost_l_a = Config.cost_l_a
cost_l_v = Config.cost_l_v

cost_w_a_56bus = Config.cost_g_a_56bus
cost_w_v_56bus = Config.cost_g_v_56bus
cost_l_a_56bus = Config.cost_l_a_56bus
cost_l_v_56bus = Config.cost_l_v_56bus

class VoltageCtrl_Env(gym.Env):
    def __init__(self, pp_net, injection_bus, v0=1, vmax=1.05, vmin=0.95):
        self.network =  pp_net
        self.obs_dim = 1
        self.topology_dim = len(pp_net.line)
        self.action_dim = 1
        self.injection_bus = injection_bus
        self.agentnum = len(injection_bus)
        self.v0 = v0 
        self.vmax = vmax
        self.vmin = vmin
        
        self.load0_p = np.copy(self.network.load['p_mw'])
        self.load0_q = np.copy(self.network.load['q_mvar'])

        self.gen0_p = np.copy(self.network.sgen['p_mw'])
        self.gen0_q = np.copy(self.network.sgen['q_mvar'])
        
        self.state = np.ones(self.agentnum, )
        self.topology_init = pp_net.line.x_ohm_per_km
        self.topology = self.topology_init

    
    #this function is used to test the policy
    def step(self, action):
        
        done = False 
        
        # $-5*|u|^2 -100 * |max(v-v_max,0)|^2 -100 * |max(v_min-v,0)|^2$
        reward = float(-1*LA.norm(action) -100*LA.norm(np.clip(self.state-self.v0, 0, np.inf))
                       - 100*LA.norm(np.clip(self.v0-self.state, 0, np.inf)))
        
        # state-transition dynamics
        for i in range(self.agentnum):
            self.network.sgen.at[i+1, 'q_mvar'] = action[i] 

        pp.runpp(self.network, algorithm='bfsw', init = 'dc')
        
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        
        if(np.min(self.state) > 0.9499 and np.max(self.state)< 1.0501):
            done = True
        
        return self.state, reward, done

    
    def step_Preward(self, action, p_action): 
        
        done = False 
        
        reward = float(-50*LA.norm(p_action) -100*LA.norm(np.clip(self.state-self.v0, 0, np.inf))
                       - 100*LA.norm(np.clip(self.v0-self.state, 0, np.inf)))
        
        # local reward
        agent_num = len(self.injection_bus)
        reward_sep = np.zeros(agent_num, )
        
        for i in range(agent_num):
            reward_sep[i] = float(-50*LA.norm(p_action[i]) -100*LA.norm(np.clip(self.state[i]-self.v0, 0, np.inf))
                           - 100*LA.norm(np.clip(self.v0-self.state[i], 0, np.inf)))              
        
        # state-transition dynamics
        for i in range(len(self.injection_bus)):
            self.network.sgen.at[i+1, 'q_mvar'] = action[i] 

        pp.runpp(self.network, algorithm='bfsw', init = 'dc')
        
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        
        if(np.min(self.state) > 0.95 and np.max(self.state)< 1.05):
            done = True
        
        return self.state, reward, reward_sep, done

    #state-transition with topology or impedance change
    def step_uncertain(self, action):
        
        done = False 
        
        # adjust parameters of the line
        # self.network.line.x_ohm_per_km = self.topology_init * np.random.uniform(0.8,1.2)
           
        #adjust reactive power inj at the PV bus
        for i in range(self.agentnum):
            self.network.sgen.at[i+1, 'q_mvar'] = action[i]     # index start from 1, there is a bug here, they did not delete the original sgen

        pp.runpp(self.network, algorithm='bfsw', init = 'dc')

        reward = float(cost_w_a_56bus * LA.norm(action)**2+ cost_w_v_56bus * LA.norm(np.clip(self.state-(self.v0), 0, np.inf))
            + cost_w_v_56bus * LA.norm(np.clip((self.v0)-self.state, 0, np.inf)))
        
        agent_num = len(self.injection_bus)
        reward_sep = np.zeros(agent_num, )
        for i in range(agent_num):
            reward_sep[i] = float(cost_l_a_56bus*LA.norm(action[i])**2 + cost_l_v_56bus * LA.norm(np.clip(self.state[i]-(self.v0), 0, np.inf))
                           + cost_l_v_56bus * LA.norm(np.clip((self.v0)-self.state[i], 0, np.inf)))    
        
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        state_all = self.network.res_bus.vm_pu.to_numpy()
        topology = self.network.line.x_ohm_per_km
        
        if(np.min(self.state) > 0.9550 and np.max(self.state)< 1.0450):
            done = True
        
        return self.state, topology, reward, reward_sep, done
    
    def topology_change(self, seed=0):
        np.random.seed(seed)

        random_change_lsit = np.random.choice([True,False], size=len(self.network.switch))
        logger.debug(random_change_lsit)

        for i in range(len(random_change_lsit)):
            self.network.switch.at[i, 'closed'] = random_change_lsit[i]
            if not random_change_lsit[i]:
                self.topology[self.network.switch.element[i]] = 0.0

    
    def reset(self, seed=1): #sample different initial volateg conditions during training
        np.random.seed(seed)
        scenario = np.random.choice([0, 1])
        #scenario = 3

        self.network.line.x_ohm_per_km = self.topology_init * np.random.uniform(0.7,1.3)
        #self.network.line.x_ohm_per_km = self.topology_init
        self.topology = self.network.line.x_ohm_per_km
        if(scenario == 0):#low voltage
            # logger.info('this episode is start at low voltage!')
            self.network.sgen['p_mw'] = 0.0
            self.network.sgen['q_mvar'] = 0.0
            self.network.load['p_mw'] = 0.0
            self.network.load['q_mvar'] = 0.0
            
            self.network.sgen.at[1, 'p_mw'] = -0.5*np.random.uniform(2, 5)
            self.network.sgen.at[2, 'p_mw'] = -0.4*np.random.uniform(10, 30)
            self.network.sgen.at[3, 'p_mw'] = -0.3*np.random.uniform(2, 8)
            self.network.sgen.at[4, 'p_mw'] = -0.3*np.random.uniform(2, 8)
            self.network.sgen.at[5, 'p_mw'] = -0.4*np.random.uniform(2, 8)

        elif(scenario == 1): #high voltage 
            # logger.info('this episode is start at high voltage!')
            self.network.sgen['p_mw'] = 0.0
            self.network.sgen['q_mvar'] = 0.0
            self.network.load['p_mw'] = 0.0
            self.network.load['q_mvar'] = 0.0
            
            self.network.sgen.at[1, 'p_mw'] = 0.5*np.random.uniform(2, 10)
            self.network.sgen.at[2, 'p_mw'] = np.random.uniform(5, 40)
            self.network.sgen.at[3, 'p_mw'] = 0.2*np.random.uniform(2, 14)
            self.network.sgen.at[4, 'p_mw'] = 0.4*np.random.uniform(2, 14) 
            self.network.sgen.at[5, 'p_mw'] = 0.4*np.random.uniform(2, 14) 
        
        else: #mixture (this is used only during testing)
            self.network.sgen['p_mw'] = 0.0
            self.network.sgen['q_mvar'] = 0.0
            self.network.load['p_mw'] = 0.0
            self.network.load['q_mvar'] = 0.0
            
            self.network.sgen.at[1, 'p_mw'] = -2*np.random.uniform(2, 3)
            self.network.sgen.at[2, 'p_mw'] = np.random.uniform(15, 35)
            self.network.sgen.at[2, 'q_mvar'] = 0.1*self.network.sgen.at[2, 'p_mw']
            self.network.sgen.at[3, 'p_mw'] = 0.2*np.random.uniform(2, 12)
            self.network.sgen.at[4, 'p_mw'] = -2*np.random.uniform(2, 8) 
            self.network.sgen.at[5, 'p_mw'] = 0.2*np.random.uniform(2, 12) 
            self.network.sgen.at[5, 'q_mvar'] = 0.2*self.network.sgen.at[5, 'p_mw']
            
        
        pp.runpp(self.network, algorithm='bfsw')
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        if np.max(self.state) > self.v0:
            logger.debug('Episode start at high volatage, highest is {} pu...', np.max(self.state))
        if np.min(self.state) < self.v0:
            logger.debug('Episode start at low volatage, lowest is {} pu...', np.min(self.state))

        return self.state, self.topology, scenario
    
    def reset_topo(self, seed=1): #initial volateg conditions with topology change
        np.random.seed(seed)
        scenario = np.random.choice([0, 1])
        #scenario = np.random.choice([0, 1, 3])
        # scenario = 3
        self.network.line.x_ohm_per_km = self.topology_init * np.random.uniform(0.5,1.5)
        self.topology = 1/self.network.line.x_ohm_per_km

        if(scenario == 0):#low voltage
            # logger.info('this episode is start at low voltage!')
            self.network.sgen['p_mw'] = 0.0
            self.network.sgen['q_mvar'] = 0.0
            self.network.load['p_mw'] = 0.0
            self.network.load['q_mvar'] = 0.0
            
            self.network.sgen.at[1, 'p_mw'] = -0.5*np.random.uniform(2, 5)
            self.network.sgen.at[2, 'p_mw'] = -0.4*np.random.uniform(10, 30)
            self.network.sgen.at[3, 'p_mw'] = -0.3*np.random.uniform(2, 8)
            self.network.sgen.at[4, 'p_mw'] = -0.3*np.random.uniform(2, 8)
            self.network.sgen.at[5, 'p_mw'] = -0.4*np.random.uniform(2, 8)

        elif(scenario == 1): #high voltage 
            # logger.info('this episode is start at high voltage!')
            self.network.sgen['p_mw'] = 0.0
            self.network.sgen['q_mvar'] = 0.0
            self.network.load['p_mw'] = 0.0
            self.network.load['q_mvar'] = 0.0
            
            self.network.sgen.at[1, 'p_mw'] = 0.5*np.random.uniform(2, 10)
            self.network.sgen.at[2, 'p_mw'] = np.random.uniform(5, 40)
            self.network.sgen.at[3, 'p_mw'] = 0.2*np.random.uniform(2, 14)
            self.network.sgen.at[4, 'p_mw'] = 0.4*np.random.uniform(2, 14) 
            self.network.sgen.at[5, 'p_mw'] = 0.4*np.random.uniform(2, 14) 
        
        else: #mixture (this is used only during testing)
            self.network.sgen['p_mw'] = 0.0
            self.network.sgen['q_mvar'] = 0.0
            self.network.load['p_mw'] = 0.0
            self.network.load['q_mvar'] = 0.0
            
            self.network.sgen.at[1, 'p_mw'] = -2*np.random.uniform(2, 3)
            self.network.sgen.at[2, 'p_mw'] = np.random.uniform(15, 35)
            self.network.sgen.at[2, 'q_mvar'] = 0.1*self.network.sgen.at[2, 'p_mw']
            self.network.sgen.at[3, 'p_mw'] = 0.2*np.random.uniform(2, 12)
            self.network.sgen.at[4, 'p_mw'] = -2*np.random.uniform(2, 8) 
            self.network.sgen.at[5, 'p_mw'] = 0.2*np.random.uniform(2, 12) 
            self.network.sgen.at[5, 'q_mvar'] = 0.2*self.network.sgen.at[5, 'p_mw']

        random_change_lsit = np.random.choice([True,False], size=len(self.network.switch))
        # logger.debug(random_change_lsit)

        for i in range(len(random_change_lsit)):
            self.network.switch.at[i, 'closed'] = random_change_lsit[i]
            if not random_change_lsit[i]:

                self.topology[self.network.switch.element[i]] = np.random.uniform(-0.01,0.01)

        
        pp.runpp(self.network, algorithm='bfsw')
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()

        return self.state, self.topology, scenario
    
    def reset0(self, seed=1): #reset voltage to nominal value
        
        self.network.load['p_mw'] = 0*self.load0_p
        self.network.load['q_mvar'] = 0*self.load0_q

        self.network.sgen['p_mw'] = 0*self.gen0_p
        self.network.sgen['q_mvar'] = 0*self.gen0_q
        
        pp.runpp(self.network, algorithm='bfsw')
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        return self.state
    

class Env_123bus(gym.Env):
    def __init__(self, pp_net, injection_bus, v0=1.0, vmax=1.05, vmin=0.95):
        self.network =  pp_net
        self.obs_dim = 1
        self.topology_dim = len(pp_net.line)
        self.action_dim = 1
        self.injection_bus = injection_bus
        self.agentnum = len(injection_bus)
        self.v0 = v0 
        self.vmax = vmax
        self.vmin = vmin
        
        self.load0_p = np.copy(self.network.load['p_mw'])
        self.load0_q = np.copy(self.network.load['q_mvar'])

        self.gen0_p = np.copy(self.network.sgen['p_mw'])
        self.gen0_q = np.copy(self.network.sgen['q_mvar'])
        
        self.state = np.ones(self.agentnum, )
        self.topology_init = pp_net.line.x_ohm_per_km
        self.topology = self.topology_init

    
    #this function is used to test the policy
    def step(self, action):
        
        done = False 
        
        # $-5*|u|^2 -100 * |max(v-v_max,0)|^2 -100 * |max(v_min-v,0)|^2$
        reward = float(-50*LA.norm(action) -100*LA.norm(np.clip(self.state-self.vmax, 0, np.inf))
                       - 100*LA.norm(np.clip(self.vmin-self.state, 0, np.inf)))
        
        # state-transition dynamics
        for i in range(self.agentnum):
            self.network.sgen.at[i, 'q_mvar'] = action[i] 

        pp.runpp(self.network, algorithm='bfsw', init = 'dc')
        
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        
        if(np.min(self.state) > 0.9499 and np.max(self.state)< 1.0501):
            done = True
        
        return self.state, reward, done

    #state-transition with topology or impedance change
    def step_uncertain(self, action):
        
        done = False 
        
        # adjust parameters of the line
        # self.network.line.x_ohm_per_km = self.topology_init * np.random.uniform(0.8,1.2)
           
        #adjust reactive power inj at the PV bus
        for i in range(self.agentnum):
            self.network.sgen.at[i, 'q_mvar'] = action[i]     # index start from 0, bus123 system is correct

        pp.runpp(self.network, algorithm='bfsw', init = 'dc')

        reward = float(cost_w_a * LA.norm(action)**2 + cost_w_v * LA.norm(np.clip(self.state-(self.v0), 0, np.inf)**0.5)
            + cost_w_v * LA.norm(np.clip((self.v0)-self.state, 0, np.inf))**0.5)
        
        agent_num = len(self.injection_bus)
        reward_sep = np.zeros(agent_num, )
        for i in range(agent_num):
            reward_sep[i] = float(cost_l_a*LA.norm(action[i])**2 + cost_l_v * LA.norm(np.clip(self.state[i]-(self.v0), 0, np.inf)**0.5)
                           + cost_l_v * LA.norm(np.clip((self.v0)-self.state[i], 0, np.inf))**0.5)
        
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        state_all = self.network.res_bus.vm_pu.to_numpy()
        topology = self.network.line.x_ohm_per_km
        
        if(np.min(self.state) > 0.9550 and np.max(self.state)< 1.0450):
            done = True
        
        return self.state, topology, reward, reward_sep, done
    
    def reset_topo(self, seed=1): #initial volateg conditions with topology change
        np.random.seed(seed)
        scenario = np.random.choice([0, 1])
        # scenario = 3
        self.network.line.x_ohm_per_km = self.topology_init * np.random.uniform(0.5,1.5)
        self.topology = 1/self.network.line.x_ohm_per_km
        #self.topology = self.topology.to_numpy()

        if(scenario == 0):#low voltage 
           # Low voltage
            self.network.sgen['p_mw'] = 0.0
            self.network.sgen['q_mvar'] = 0.0
            self.network.load['p_mw'] = 0.0
            self.network.load['q_mvar'] = 0.0
            
            self.network.sgen.at[0, 'p_mw'] = -0.8*np.random.uniform(15, 60)
            # self.network.sgen.at[0, 'q_mvar'] = -0.8*np.random.uniform(10, 300)
            self.network.sgen.at[1, 'p_mw'] = -0.8*np.random.uniform(10, 45)
            self.network.sgen.at[2, 'p_mw'] = -0.8*np.random.uniform(10, 55)
            self.network.sgen.at[3, 'p_mw'] = -0.8*np.random.uniform(10, 30)
            self.network.sgen.at[4, 'p_mw'] = -0.6*np.random.uniform(1, 35)
            self.network.sgen.at[5, 'p_mw'] = -0.5*np.random.uniform(2, 25)
            self.network.sgen.at[6, 'p_mw'] = -0.8*np.random.uniform(2, 30)
            self.network.sgen.at[7, 'p_mw'] = -0.9*np.random.uniform(1, 10)
            self.network.sgen.at[8, 'p_mw'] = -0.7*np.random.uniform(1, 15)
            self.network.sgen.at[9, 'p_mw'] = -0.5*np.random.uniform(1, 30)
            self.network.sgen.at[10, 'p_mw'] = -0.3*np.random.uniform(1, 20)
            self.network.sgen.at[11, 'p_mw'] = -0.5*np.random.uniform(1, 20)
            self.network.sgen.at[12, 'p_mw'] = -0.4*np.random.uniform(1, 20)
            self.network.sgen.at[13, 'p_mw'] = -0.4*np.random.uniform(2, 10)
            #not real controllers
            self.network.sgen.at[14, 'p_mw'] = -0.4*np.random.uniform(10, 20)
            self.network.sgen.at[15, 'p_mw'] = -0.8*np.random.uniform(10, 20)
            self.network.sgen.at[16, 'p_mw'] = -0.8*np.random.uniform(10, 20)


        elif(scenario == 1): #high voltage 
            self.network.sgen['p_mw'] = 0.0
            self.network.sgen['q_mvar'] = 0.0
            self.network.load['p_mw'] = 0.0
            self.network.load['q_mvar'] = 0.0
            
            self.network.sgen.at[0, 'p_mw'] = 0.8*np.random.uniform(15, 60)
            # self.network.sgen.at[0, 'q_mvar'] = 0.6*np.random.uniform(5, 300)
            self.network.sgen.at[1, 'p_mw'] = 0.8*np.random.uniform(15, 50)
            self.network.sgen.at[2, 'p_mw'] = 0.8*np.random.uniform(20, 60)
            self.network.sgen.at[3, 'p_mw'] = 0.8*np.random.uniform(10, 34)
            self.network.sgen.at[4, 'p_mw'] = 0.8*np.random.uniform(2, 20)
            self.network.sgen.at[5, 'p_mw'] = 0.8*np.random.uniform(2, 80)
            self.network.sgen.at[6, 'p_mw'] = 0.8*np.random.uniform(10, 80)
            self.network.sgen.at[7, 'p_mw'] = 0.8*np.random.uniform(5, 50)
            self.network.sgen.at[8, 'p_mw'] = 0.7*np.random.uniform(2, 30)
            self.network.sgen.at[9, 'p_mw'] = 0.5*np.random.uniform(2, 30)
            self.network.sgen.at[10, 'p_mw'] = 0.4*np.random.uniform(1, 40)
            self.network.sgen.at[11, 'p_mw'] = 0.5*np.random.uniform(1, 30)
            self.network.sgen.at[12, 'p_mw'] = 0.5*np.random.uniform(1, 30)
            self.network.sgen.at[13, 'p_mw'] = 0.5*np.random.uniform(1, 24)
            #not real controllers
            self.network.sgen.at[14, 'p_mw'] = 0.5*np.random.uniform(15, 25)
            self.network.sgen.at[15, 'p_mw'] = 0.8*np.random.uniform(10, 50)
            self.network.sgen.at[16, 'p_mw'] = 0.8*np.random.uniform(10, 20)

        random_change_lsit = np.random.choice([True,False], size=len(self.network.switch))
        #logger.debug(random_change_lsit)
        
        for i in range(len(random_change_lsit)):
            self.network.switch.at[i, 'closed'] = random_change_lsit[i]
            if not random_change_lsit[i]:
                self.topology[self.network.switch.element[i]] = np.random.uniform(-0.02,0.04)

        #logger.debug(self.topology)
        
        pp.runpp(self.network, algorithm='bfsw')
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()

        return self.state, self.topology, scenario
    
    def reset0(self, seed=1): #reset voltage to nominal value
        
        self.network.load['p_mw'] = 0*self.load0_p
        self.network.load['q_mvar'] = 0*self.load0_q

        self.network.sgen['p_mw'] = 0*self.gen0_p
        self.network.sgen['q_mvar'] = 0*self.gen0_q
        
        pp.runpp(self.network, algorithm='bfsw')
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        return self.state

def create_56bus():
    pp_net = pp.converter.from_mpc('data/SCE_56bus.mat', casename_mpc_file='case_mpc')
    pp_net.sgen['p_mw'] = 0.0
    pp_net.sgen['q_mvar'] = 0.0

    pp.create_sgen(pp_net, 17, p_mw = 1.5, q_mvar=0)
    pp.create_sgen(pp_net, 20, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 29, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 44, p_mw = 2, q_mvar=0)
    pp.create_sgen(pp_net, 52, p_mw = 2, q_mvar=0)

    # define which lines can be closed or opened
    switch_lines = [7, 10, 12, 14, 22, 31, 33, 34, 35, 36, 37, 38, 41, 42, 46, 48, 50, 54]
    lines_connect_buses = [7, 10, 12, 14, 22, 31, 33, 33, 35, 33, 37, 38, 41, 41, 46, 48, 50, 52]
    

    for i in range(len(switch_lines)):
        pp.create_switch(pp_net, bus=lines_connect_buses[i], 
                            element=switch_lines[i], et='l', closed=True)
    
    return pp_net

def create_123bus():
    pp_net = pp.converter.from_mpc('data/case_123.mat', casename_mpc_file='case_mpc')
    
    pp_net.sgen['p_mw'] = 0.0
    pp_net.sgen['q_mvar'] = 0.0

    pp.create_sgen(pp_net, 9, p_mw = 1.5, q_mvar=0)
    pp.create_sgen(pp_net, 10, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 15, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 19, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 32, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 35, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 47, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 58, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 65, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 74, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 82, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 91, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 103, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 60, p_mw = 1, q_mvar=0) #node 114 in the png
    
    #only for reset
    pp.create_sgen(pp_net, 13, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 14, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 18, p_mw = 1, q_mvar=0)

    # define which lines can be closed or opened
    switch_lines =        [0, 1, 3, 4, 5, 7,  20, 22, 27, 31, 25, 29, 30, 35, 36, 37, 38, 40, 42, 44, 46, 47, 48, 16, 51, 53, 62, 65, 66, 67, 75, 79, 81, 83, 85, 88, 89, 90, 91, 92, 93, 94,  99, 101, 104]
    lines_connect_buses = [1, 2, 3, 4, 5, 11, 21, 23, 30, 31, 27, 28, 29, 36, 37, 38, 40, 42, 44, 45, 48, 49, 50, 16, 54, 55, 67, 68, 69, 70, 78, 83, 84, 87, 89, 92, 93, 94, 95, 97, 98, 99, 105, 106, 110]

    for i in range(len(switch_lines)):
        pp.create_switch(pp_net, bus=lines_connect_buses[i], 
                            element=switch_lines[i], et='l', closed=True)

    return pp_net

if __name__ == "__main__":

    # injection_bus = np.array([18, 21, 30, 45, 53])-1
    # pp_net = create_56bus()
    # env = VoltageCtrl_Env(pp_net, injection_bus)

    injection_bus = np.array([9, 10, 15, 19, 32, 35, 47, 58, 65, 74, 82, 91, 103, 60]) #11, 36, 75,/ 1,5,9
    pp_net = create_123bus()
    env = Env_123bus(pp_net, injection_bus)

    for i in range(5):
        state, topology, scenario = env.reset_topo(i+5)
        topology = torch.cuda.FloatTensor(topology).unsqueeze(0)
        # topology = topology.expand(64,55)
        topology = F.normalize(topology)
    high_volt = 0
    low_volt = 0
    # print(env.network.line.x_ohm_per_km)
    print(pp_net.sgen)

    simple_plotly(env.network, figsize=2)
    # pf_res_plotly(pp_net)

 
