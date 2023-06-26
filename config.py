### this file is used to define some hyperparameter in module and traning

class Config:

    # used in reward funtion
    cost_w_a = -6       # weight of action cost
    cost_w_v = -80     # weight of voltage error cost

    # traning golbal reward and local reward
    r_global_weight = 0.25
    r_local_weight = 0.6

    # dead-zone offset, default deadzone is 0.05, use this offset to reduce dead-zon6
    dz_offset = 0.02
    # tarning rate
    policy_learning_rate = 2e-4
    value_learning_rate = 1e-3
    lr_discount = 0.5
    policy_milestones = [1000, 2000, 4000]       # change learning rate at specific steps
    value_milestones = [1500, 3000, 5000, 9000]

    # traing parameter
    agent_num = 5           # agent number is equal to the controllable buses
    obs_dim = 56            # obs_dim = the dimensions of voltage state + the dimensions of topology matrix
    state_dim = 1           # voltage state
    topology_dim = 55       # topology matrix dimensions
    action_dim = 1
    hidden_dim = 512
    topology_hidden_dim = 256   # hidden neurons in topology nn module
    total_episodes = 300    
    total_steps = 64        # trajetory length each episode
    batch_size = 128

    # nn module parameter
    topology_net_init_w = 0.05      # the range of uniform initial weight, [0, topology_net_init_w]


