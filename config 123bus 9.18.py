### this file is used to define some hyperparameter in module and traning

class Config:

    # used in reward funtion
    cost_g_a = -0       # weight of global action cost
    cost_g_v = -50     # weight of global voltage error cost
    cost_l_a = -5       # weight of local action cost
    cost_l_v = -100     # weight of local voltage error cost

    # traning golbal reward and local reward
    r_global_weight = 0.5
    r_local_weight = 0.5

    # dead-zone offset, default deadzone is 0.05, use this offset to reduce dead-zon6
    dz_offset = 0.02
    # tarning rate
    policy_learning_rate = 2e-4
    value_learning_rate = 1e-3
    lr_discount = 0.5
    policy_milestones = [1000,2000,4000]       # change learning rate at specific steps
    value_milestones = [1500, 3000, 5000, 9000]

    # traing parameter
    agent_num = 5           # agent number is equal to the controllable buses
    obs_dim = 56            # obs_dim = the dimensions of voltage state + the dimensions of topology matrix
    state_dim = 1           # voltage state
    topology_dim = 55       # topology matrix dimensions
    action_dim = 1
    hidden_dim = 1024
    hidden_dim_56bus = 2048
    topology_hidden_dim = 256   # hidden neurons in topology nn module
    total_episodes = 1000
    total_steps = 60        # trajetory length each episode
    batch_size = 256

    max_action = 25        #maximum output of q_dot

    # nn module parameter
    topology_net_init_w = 0.03      # the range of uniform initial weight, [0, topology_net_init_w]


