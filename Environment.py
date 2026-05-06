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

        # Save original injection buses and agent number
        self.original_injection_bus = np.copy(injection_bus)
        self.original_agentnum = len(injection_bus)

    
    #this function is used to test the policy
    def step(self, action):
        
        done = False 
        
        # $-5*|u|^2 -100 * |max(v-v_0,0)|^2 -100 * |max(v_0-v,0)|^2$
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

    
    def step_load(self, action, load_p, load_q, pv_p): 
        
        done = False 
        
        reward = float(-1*LA.norm(action) -100*LA.norm(np.clip(self.state-self.v0, 0, np.inf))
                       - 100*LA.norm(np.clip(self.v0-self.state, 0, np.inf)))          
        
        load_bus_list = [5, 8, 10, 12, 14, 19, 22, 33, 37, 38, 41]
        for i in range(len(load_bus_list)):
            self.network.load.at[load_bus_list[i], 'p_mw'] = load_p * 0.08
            self.network.load.at[load_bus_list[i], 'q_mvar'] = load_q * 0.08
        # state-transition dynamics
        for i in range(len(self.injection_bus)):    # only control the first 5 PVs
            if i < 5:
                self.network.sgen.at[i+1, 'p_mw'] = pv_p * 0.35
                self.network.sgen.at[i+1, 'q_mvar'] = action[i]
            else:
                self.network.sgen.at[i+1, 'p_mw'] = pv_p * 0.35
                self.network.sgen.at[i+1, 'q_mvar'] = 0

        pp.runpp(self.network, algorithm='bfsw', init = 'dc')
        
        # Get state only for original buses
        self.state = self.network.res_bus.iloc[self.original_injection_bus].vm_pu.to_numpy()
        state_all = self.network.res_bus.vm_pu.to_numpy()
        topology = self.network.line.x_ohm_per_km
        
        if(np.min(self.state) > 0.95 and np.max(self.state)< 1.05):
            done = True
        
        return self.state, topology, reward, done

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
        self.network.line.x_ohm_per_km = self.topology_init * np.random.uniform(0.5,1.5)
        self.topology = 1/self.network.line.x_ohm_per_km

        random_change_lsit = np.random.choice([True,False], size=len(self.network.switch))
        logger.debug(random_change_lsit)

        for i in range(len(random_change_lsit)):
            self.network.switch.at[i, 'closed'] = random_change_lsit[i]
            if not random_change_lsit[i]:
                self.topology[self.network.switch.element[i]] = np.random.uniform(-0.01,0.01)

        pp.runpp(self.network, algorithm='bfsw')
        #logger.debug(self.topology)

        return self.topology
    
    def get_all_state(self):
        return self.network.res_bus.vm_pu.to_numpy()
    
    def topology_reset(self):
        self.network.line.x_ohm_per_km = self.topology_init
        self.topology = 1/self.network.line.x_ohm_per_km

        pp.runpp(self.network, algorithm='bfsw')
        #logger.debug(self.topology)

        return self.topology
    
    def pv_node_change(self, action='disconnect', pv_index=None, bus_id=None, p_mw=None, seed=None):
        """
        Function to dynamically change PV generation nodes
        
        Parameters:
        -----------
        action : str
            'disconnect' - disconnect a PV node
            'add' - add a new PV node
            'random' - randomly disconnect or connect PV nodes
        pv_index : int, optional
            Index of PV node to disconnect (index in sgen table)
        bus_id : int, optional  
            Bus ID to connect new PV node
        p_mw : float, optional
            Active power output (MW) for new PV node
        seed : int, optional
            Random seed
        
        Returns:
        --------
        dict : Dictionary containing operation results
        """
        if seed is not None:
            np.random.seed(seed)
        
        result = {'action': action, 'success': False, 'details': ''}
        
        if action == 'disconnect':
            # Get currently active PV nodes
            active_pv_indices = self.network.sgen[self.network.sgen.in_service == True].index.tolist()
            
            if len(active_pv_indices) == 0:
                result['details'] = 'No active PV nodes to disconnect'
                return result
            
            # If no specific PV node specified, randomly select one
            if pv_index is None:
                pv_index = np.random.choice(active_pv_indices)
            
            # Check if index is valid
            if pv_index not in active_pv_indices:
                result['details'] = f'PV index {pv_index} is not active or does not exist'
                return result
            
            # Disconnect PV node
            self.network.sgen.at[pv_index, 'in_service'] = False
            
            # Update injection_bus array
            disconnected_bus = self.network.sgen.at[pv_index, 'bus']
            if disconnected_bus in self.injection_bus:
                # Convert to list, remove bus, then convert back to numpy array
                injection_bus_list = self.injection_bus.tolist() if isinstance(self.injection_bus, np.ndarray) else list(self.injection_bus)
                injection_bus_list = [bus for bus in injection_bus_list if bus != disconnected_bus]
                self.injection_bus = np.array(injection_bus_list)
                self.agentnum = len(self.injection_bus)
            
            result['success'] = True
            result['details'] = f'Disconnected PV at index {pv_index}, bus {disconnected_bus}'
            
        elif action == 'add':
            # Check required parameters
            if bus_id is None:
                # Randomly select a bus without PV
                existing_pv_buses = self.network.sgen[self.network.sgen.in_service == True]['bus'].tolist()
                all_buses = self.network.bus.index.tolist()
                available_buses = [bus for bus in all_buses if bus not in existing_pv_buses and bus != 0]  # Exclude slack bus
                
                if len(available_buses) == 0:
                    result['details'] = 'No available buses for new PV'
                    return result
                    
                bus_id = np.random.choice(available_buses)
            
            # Set default power
            if p_mw is None:
                p_mw = np.random.uniform(0.5, 5.0)  # Default 0.5-5MW

            # Convert numpy array to scalar if needed
            if isinstance(p_mw, np.ndarray):
                p_mw = float(p_mw.item()) if p_mw.size == 1 else float(p_mw[0])
            else:
                p_mw = float(p_mw)
            
            # Create new PV node
            new_index = self.network.sgen.index.max() + 1 if len(self.network.sgen) > 0 else 0
            
            pp.create_sgen(self.network, 
                        bus=bus_id,
                        p_mw=p_mw,
                        q_mvar=0,  # Initial reactive power is 0
                        name=f'PV_{new_index}',
                        index=new_index,
                        in_service=True)
            
            # Update injection_bus array
            if bus_id not in self.injection_bus:
                # Convert to list, add bus, sort, then convert back to numpy array
                injection_bus_list = self.injection_bus.tolist() if isinstance(self.injection_bus, np.ndarray) else list(self.injection_bus)
                injection_bus_list.append(bus_id)
                injection_bus_list.sort()
                self.injection_bus = np.array(injection_bus_list)
                self.agentnum = len(self.injection_bus)
            
            result['success'] = True
            result['details'] = f'Added PV at bus {bus_id} with {p_mw:.2f} MW, index {new_index}'
            
        elif action == 'random':
            # Randomly decide to disconnect or add
            random_action = np.random.choice(['disconnect', 'add'])
            return self.pv_node_change(action=random_action, seed=seed)
        
        else:
            result['details'] = f'Unknown action: {action}'
            return result
        
        # Run power flow to update system state
        try:
            pp.runpp(self.network, algorithm='bfsw', init='dc')
            self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        except:
            result['success'] = False
            result['details'] += ' - Power flow failed'
        
        return result


    def pv_node_reset(self):
        """
        Reset all PV nodes to initial state
        """
        # Remove PV nodes added later
        original_indices = list(range(len(self.gen0_p)))
        current_indices = self.network.sgen.index.tolist()
        indices_to_drop = [idx for idx in current_indices if idx >= len(self.gen0_p)]
        
        if indices_to_drop:
            self.network.sgen = self.network.sgen.drop(indices_to_drop)
            # Reset index to ensure consistency
            self.network.sgen.reset_index(drop=True, inplace=True)
        
        # Restore all original PV nodes
        for i in range(len(self.gen0_p)):
            if i < len(self.network.sgen):
                self.network.sgen.at[i, 'in_service'] = True
                self.network.sgen.at[i, 'p_mw'] = self.gen0_p[i]
                self.network.sgen.at[i, 'q_mvar'] = self.gen0_q[i]
        
        # Reset injection_bus to original
        self.injection_bus = np.copy(self.original_injection_bus)
        self.agentnum = self.original_agentnum
        
        # Don't run power flow here to avoid conflicts
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        
        return {'success': True, 'details': 'PV nodes reset to initial state'}


    def get_pv_status(self):
        """
        Get current status of all PV nodes
        """
        pv_status = []
        for idx, row in self.network.sgen.iterrows():
            status = {
                'index': idx,
                'bus': row['bus'],
                'name': row.get('name', f'PV_{idx}'),
                'p_mw': row['p_mw'],
                'q_mvar': row['q_mvar'],
                'in_service': row['in_service']
            }
            pv_status.append(status)
        
        return pv_status
    

    # Get the weighted adjacency matrix based on current line parameters
    def _get_adjacency_matrix(self):
        """
        Constructs the weighted adjacency matrix based on current line parameters.
        Returns:
            adj (np.array): [num_nodes, num_nodes] weighted matrix.
        """
        num_nodes = len(self.network.bus)
        adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        
        # Extract connectivity and weights (admittance = 1/impedance)
        # Assuming topology_init or similar stores the base values, 
        # but here we use the current line parameters directly from the pp_net object 
        # to ensure we capture any dynamic changes made during reset.
        
        # Note: Ensure bus indices are 0-based integers for matrix indexing.
        from_bus = self.network.line.from_bus.values.astype(int)
        to_bus = self.network.line.to_bus.values.astype(int)
        
        # Calculate weights (e.g., admittance magnitude)
        # You can use self.topology vector if it's already updated, or recalculate:
        weights = 1.0 / self.network.line.x_ohm_per_km.values
        
        # Fill the matrix (Undirected graph -> Symmetric)
        adj[from_bus, to_bus] = weights
        adj[to_bus, from_bus] = weights
        
        return adj

    
    def reset(self, seed=1): #sample different initial volateg conditions during training
        np.random.seed(seed)
        scenario = np.random.choice([0, 1])
        #scenario = 3

        self.network.line.x_ohm_per_km = self.topology_init * np.random.uniform(0.7,1.3)
        #self.network.line.x_ohm_per_km = self.topology_init
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
            
        
        pp.runpp(self.network, algorithm='bfsw')
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        if np.max(self.state) > self.v0:
            logger.debug('Episode start at high volatage, highest is {} pu...', np.max(self.state))
        if np.min(self.state) < self.v0:
            logger.debug('Episode start at low volatage, lowest is {} pu...', np.min(self.state))

        return self.state, self.topology, scenario
    
    def reset_topo(self, seed=1, manual_switch=None): #initial volateg conditions with topology change
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
            self.network.sgen.at[2, 'p_mw'] = np.random.uniform(5, 15)
            self.network.sgen.at[2, 'q_mvar'] = 0.1*self.network.sgen.at[2, 'p_mw']
            self.network.sgen.at[3, 'p_mw'] = 0.2*np.random.uniform(2, 12)
            self.network.sgen.at[4, 'p_mw'] = -2*np.random.uniform(2, 8) 
            self.network.sgen.at[5, 'p_mw'] = 0.2*np.random.uniform(2, 12) 
            self.network.sgen.at[5, 'q_mvar'] = 0.1*self.network.sgen.at[5, 'p_mw']

        random_change_lsit = np.random.choice([True,False], size=len(self.network.switch))
        if manual_switch is not None:
            for idx, state in manual_switch.items():
                if 0 <= idx < len(random_change_lsit):
                    random_change_lsit[idx] = state
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
        reward = float(-0.1*LA.norm(action) -100*LA.norm(np.clip(self.state-self.v0, 0, np.inf))
                       - 100*LA.norm(np.clip(self.v0-self.state, 0, np.inf)))
        
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

        reward = float(cost_w_a * LA.norm(action)**2 + cost_w_v * LA.norm(np.clip(self.state-(self.v0), 0, np.inf))
            + cost_w_v * LA.norm(np.clip((self.v0)-self.state, 0, np.inf)))
        
        agent_num = len(self.injection_bus)
        reward_sep = np.zeros(agent_num, )
        for i in range(agent_num):
            reward_sep[i] = float(cost_l_a*LA.norm(action[i])**2 + cost_l_v * LA.norm(np.clip(self.state[i]-(self.v0), 0, np.inf))
                           + cost_l_v * LA.norm(np.clip((self.v0)-self.state[i], 0, np.inf)))
        
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        state_all = self.network.res_bus.vm_pu.to_numpy()
        topology = self.network.line.x_ohm_per_km
        
        if(np.min(self.state) > 0.9550 and np.max(self.state)< 1.0450):
            done = True
        
        return self.state, topology, reward, reward_sep, done
    
    def reset_topo(self, seed=1, manual_switch=None): #initial volateg conditions with topology change
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
        if manual_switch is not None:
            for idx, state in manual_switch.items():
                if 0 <= idx < len(random_change_lsit):
                    random_change_lsit[idx] = state
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

# create a 56 bus system with 10 PV nodes
def create_56bus_10():
    pp_net = pp.converter.from_mpc('data/SCE_56bus.mat', casename_mpc_file='case_mpc')
    pp_net.sgen['p_mw'] = 0.0
    pp_net.sgen['q_mvar'] = 0.0

    pp.create_sgen(pp_net, 17, p_mw = 1.5, q_mvar=0)
    pp.create_sgen(pp_net, 20, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 29, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 44, p_mw = 2, q_mvar=0)
    pp.create_sgen(pp_net, 52, p_mw = 2, q_mvar=0)
    pp.create_sgen(pp_net, 33, p_mw = 1.5, q_mvar=0) #
    pp.create_sgen(pp_net, 14, p_mw = 1, q_mvar=0)  #
    pp.create_sgen(pp_net, 24, p_mw = 1, q_mvar=0)  #
    pp.create_sgen(pp_net, 45, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 18, p_mw = 1, q_mvar=0)

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

    injection_bus = np.array([11, 15, 18, 21, 27, 30, 34, 45, 46, 53])-1
    pp_net = create_56bus_10()
    env = VoltageCtrl_Env(pp_net, injection_bus)

    # injection_bus = np.array([9, 10, 15, 19, 32, 35, 47, 58, 65, 74, 82, 91, 103, 60]) #11, 36, 75,/ 1,5,9
    # pp_net = create_123bus()
    # env = Env_123bus(pp_net, injection_bus)

    for i in range(50):
        state, topology, scenario = env.reset_topo(i+50)
        topology = torch.cuda.FloatTensor(topology).unsqueeze(0)
        # topology = topology.expand(64,55)
        topology = F.normalize(topology)
        env.step_uncertain(np.zeros(env.agentnum,))
        print('episode:', i, 'scenario:', scenario)
    high_volt = 0
    low_volt = 0
    # print(env.network.line.x_ohm_per_km)
    print(pp_net.sgen)

    simple_plotly(env.network, figsize=2)
    pf_res_plotly(pp_net)



 
