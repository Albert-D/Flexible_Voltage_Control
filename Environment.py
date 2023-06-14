import gymnasium as gym

import numpy as np
from numpy import linalg as LA

import pandapower as pp
import pandapower.networks as pn
import pandas as pd 

from gymnasium import spaces
from gymnasium.utils import seeding

from pandapower.plotting.plotly import simple_plotly
from pandapower.plotting.plotly import vlevel_plotly
from pandapower.plotting.plotly import pf_res_plotly

from loguru import logger

# some hyperparameter in traning
cost_w_a = -8       # weight of action cost
cost_w_v = -100     # weight of voltage error cost

class VoltageCtrl_Env(gym.Env):
    def __init__(self, pp_net, injection_bus, v0=1, vmax=1.05, vmin=0.95 ,v_std = 1.0):
        self.network =  pp_net
        self.obs_dim = 56
        self.action_dim = 1
        self.injection_bus = injection_bus
        self.agentnum = len(injection_bus)
        self.v0 = v0 
        self.vmax = vmax
        self.vmin = vmin
        self.v_std = v_std
        
        self.load0_p = np.copy(self.network.load['p_mw'])
        self.load0_q = np.copy(self.network.load['q_mvar'])

        self.gen0_p = np.copy(self.network.sgen['p_mw'])
        self.gen0_q = np.copy(self.network.sgen['q_mvar'])
        
        self.state = np.ones(self.agentnum, )
        self.topology_init = pp_net.line.x_ohm_per_km

    
    #this function is used to test the policy
    def step(self, action):
        
        done = False 
        
        # $-50*|u|^2 -100 * |max(v-v_max,0)|^2 -100 * |max(v_min-v,0)|^2$
        reward = float(-50*LA.norm(action)**2 -100*LA.norm(np.clip(self.state-self.vmax, 0, np.inf))**2
                       - 100*LA.norm(np.clip(self.vmin-self.state, 0, np.inf))**2)
        
        # state-transition dynamics
        for i in range(self.agentnum):
            self.network.sgen.at[i, 'q_mvar'] = action[i] 

        pp.runpp(self.network, algorithm='bfsw', init = 'dc')
        
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        
        if(np.min(self.state) > 0.9499 and np.max(self.state)< 1.0501):
            done = True
        
        return self.state, reward, done

    
    def step_Preward(self, action, p_action): 
        
        done = False 
        
        reward = float(-50*LA.norm(p_action) -100*LA.norm(np.clip(self.state-self.vmax, 0, np.inf))
                       - 100*LA.norm(np.clip(self.vmin-self.state, 0, np.inf)))
        
        # local reward
        agent_num = len(self.injection_bus)
        reward_sep = np.zeros(agent_num, )
        
        for i in range(agent_num):
            reward_sep[i] = float(-50*LA.norm(p_action[i])**2 -100*LA.norm(np.clip(self.state[i]-self.vmax, 0, np.inf))**2
                           - 100*LA.norm(np.clip(self.vmin-self.state[i], 0, np.inf))**2)              
        
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
            self.network.sgen.at[i, 'q_mvar'] = action[i] 

        pp.runpp(self.network, algorithm='bfsw', init = 'dc')

        reward = float(cost_w_a * LA.norm(action) + cost_w_v * LA.norm(np.clip(self.state-(self.vmax-0.02), 0, np.inf))
            + cost_w_v * LA.norm(np.clip((self.vmin+0.02)-self.state, 0, np.inf)))      #8/100,
        
        agent_num = len(self.injection_bus)
        reward_sep = np.zeros(agent_num, )
        for i in range(agent_num):
            reward_sep[i] = float(cost_w_a*LA.norm(action[i]) + cost_w_v * LA.norm(np.clip(self.state[i]-(self.vmax-0.02), 0, np.inf))
                           + cost_w_v * LA.norm(np.clip((self.vmin+0.02)-self.state[i], 0, np.inf)))    
        
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        state_all = self.network.res_bus.vm_pu.to_numpy()
        topology = self.network.line.x_ohm_per_km
        
        if(np.min(self.state) > 0.9499 and np.max(self.state)< 1.0501):
            done = True
        
        return self.state, topology, reward, reward_sep, done
    
    def reset(self, seed=1): #sample different initial volateg conditions during training
        np.random.seed(seed)
        senario = np.random.choice([0, 1])
        #senario = 0
        self.network.line.x_ohm_per_km = self.topology_init * np.random.uniform(0.7,1.3)
        topology = self.network.line.x_ohm_per_km
        if(senario == 0):#low voltage
            logger.info('this episode is start at low voltage!')
            self.network.sgen['p_mw'] = 0.0
            self.network.sgen['q_mvar'] = 0.0
            self.network.load['p_mw'] = 0.0
            self.network.load['q_mvar'] = 0.0
            
            self.network.sgen.at[1, 'p_mw'] = -0.5*np.random.uniform(2, 5)
            self.network.sgen.at[2, 'p_mw'] = -0.6*np.random.uniform(10, 30)
            self.network.sgen.at[3, 'p_mw'] = -0.3*np.random.uniform(2, 8)
            self.network.sgen.at[4, 'p_mw'] = -0.3*np.random.uniform(2, 8)
            self.network.sgen.at[5, 'p_mw'] = -0.4*np.random.uniform(2, 8)

        elif(senario == 1): #high voltage 
            logger.info('this episode is start at high voltage!')
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
        return self.state, topology, senario
    
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

    return pp_net

if __name__ == "__main__":

    injection_bus = np.array([18, 21, 30, 45, 53])-1
    pp_net = create_56bus()
    env = VoltageCtrl_Env(pp_net, injection_bus)
    high_volt = 0
    low_volt = 0
    print(env.network.line.x_ohm_per_km)
    for i in range(10):
        #env.network.line.x_ohm_per_km = env.network.line.x_ohm_per_km * np.random.uniform(0.8,1.2)
        state, s = env.reset(i)
        #print(env.network.line.x_ohm_per_km)
        print(s)
        # pf_res_plotly(pp_net)
    #simple_plotly(pp_net)
    # pf_res_plotly(pp_net)
    print(f'total high senario are {high_volt}')
    print(f'total low senario are {low_volt}')
 
