from Environment import *
from DDPG import *
from NN_Module import *
from config_56bus import Config
import os

import torch
import matplotlib.pyplot as plt
import scienceplots
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc
from scipy import interpolate

from loguru import logger
import os
import numpy as np

# plt.style.use('ieee')
# Update global plot parameters

# 方案1 - 柔和专业的配色
# colors = ['#4477AA', '#66CCEE', '#228833', '#CCBB44', '#EE6677', '#AA3377']

# 方案2 - 高对比度但不刺眼的配色
colors = ['#0077BB', '#EE7733', '#009988', '#CC3311', '#33BBEE', '#EE3377']

# 更清晰的线型设置
plt.rcParams.update({
    "figure.figsize": (3.5, 2.5),
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6,
    "lines.linewidth": 0.7,  # 增加线宽使图像更清晰
    "figure.dpi": 200,      # 提高分辨率
    "figure.autolayout": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    # 简化线型，使用更清晰的样式
    "axes.prop_cycle": plt.cycler('color', colors) + 
                      plt.cycler('linestyle', ['-', '--', ':', '-.', (0, (3, 1)), (0, (1, 1))])
})

NATURE_CONFIG = {
    "width": 1800,
    "height": 900,
    "font_base": 28,
    "font_title": 32,
    "font_axis": 24,
    "font_legend": 24,
    "dpi": 300
}

# 添加网格使图表更清晰
plt.grid(True, linestyle='--', alpha=0.2)

# 设置背景色
plt.gca().set_facecolor('#f8f9fa')

### a simple logger
logger.remove()
logger.add(sys.stderr, level='DEBUG')

env_seed = 56        #10-h  5-h 0-l 1-h 2-l 3-l 4l 7h 8h 9l 6l

agent_num = 5
agent_policy_net = []
safe_agent_net = []

### create testing environment
injection_bus = np.array([18, 21, 30, 45, 53])-1
pp_net = create_56bus()
env = VoltageCtrl_Env(pp_net, injection_bus)
# state, topology, senario = env.reset_topo(seed=env_seed)      #change topology
state, topology, senario = env.reset(seed=env_seed)             #not change
topology = torch.cuda.FloatTensor(topology).unsqueeze(0)

def moving_average(a, n=3):
    # Padding the array to maintain the length after convolution.
    pad = np.pad(a, (n//2, n-1-n//2), mode='edge')
    ret = np.convolve(pad, np.ones(n), mode='valid') / n
    return ret

# plot policy
# def policy_plotly(policy_net, topology):
#     """
#     用 Plotly 绘制各母线的策略曲线，每个子图显示一个母线的 RLC-FT 策略与基线（Linear）策略比较，
#     """
#     default_colors = pc.qualitative.Plotly  # Plotly 默认颜色序列
#     color_linear = default_colors[0]
#     color_rlc = default_colors[1]
#     fig = make_subplots(rows=1, cols=5,
#                         subplot_titles=['Bus 18', 'Bus 21', 'Bus 30', 'Bus 45', 'Bus 53'])
#     N = 400
#     for i in range(5):
#         baseline_vals = []
#         policy_vals = []
#         for j in range(N):
#             # 计算基线控制值：baseline = max(state-1.05, 0) - max(0.95-state, 0)
#             state_val = 0.80 + 0.001 * j
#             base = np.maximum(state_val - 1.05, 0) - np.maximum(0.95 - state_val, 0)
#             baseline_vals.append(-base)  # 取负值
#             state_tensor = torch.tensor([[state_val]])
#             action_tensor = policy_net[i](state_tensor, topology)
#             policy_vals.append(float(-action_tensor.detach().cpu().numpy()[0]))

#         baseline_vals = np.array(baseline_vals)
#         policy_vals_smoothed = moving_average(np.array(policy_vals), n=20)
#         baseline_vals_scaled = 5 * baseline_vals
        
#         x_vals = np.linspace(10, 14, N)
        
#         # 仅在第一列显示图例，其余子图同组 trace 设为不显示图例
#         showlegend = True if i == 0 else False

#         fig.add_trace(go.Scatter(
#             x=x_vals,
#             y=baseline_vals_scaled,
#             mode='lines',
#             name='Linear',
#             legendgroup='Linear',
#             showlegend=showlegend,
#             line=dict(dash='dash', color=color_linear)
#         ), row=1, col=i+1)

#         fig.add_trace(go.Scatter(
#             x=x_vals,
#             y=policy_vals_smoothed,
#             mode='lines',
#             name='RLC-FT',
#             legendgroup='RLC-FT',
#             showlegend=showlegend,
#             line=dict(color=color_rlc)
#         ), row=1, col=i+1)

#     # 保证仅在第一个子图显示y轴标题，第三个子图显示x轴标题
#     fig.update_yaxes(title_text="Q (MVar)", row=1, col=1)
#     fig.update_xaxes(title=dict(text="Voltage (kV)", standoff=25), row=1, col=3)
#     fig.update_layout(
#         width=1400,
#         height=500,
#         showlegend=True,
#         font=dict(size=16),
#         xaxis=dict(
#             tickfont=dict(size=12),
#             showline=True,
#             mirror=True,
#             showgrid=True,
#         ),
#         yaxis=dict(
#             tickfont=dict(size=12),
#             showline=True,
#             mirror=True,
#             showgrid=True,
#         ),
#     )
    
#     output_path = os.path.join(Config.data_path, 'images', '56bus', 'policy_plot.pdf')
#     import plotly.io as pio
#     pio.kaleido.scope.mathjax = None
#     fig.write_image(output_path)
#     fig.show()

# plot policy
def enhanced_policy_plot(policy_net, topology):
    """
    Enhanced policy visualization with consistent styling with trajectory plot
    Plots are arranged horizontally (1×5 layout)
    
    Args:
        policy_net: The policy network models for different buses
        topology: The power system topology indicator
        
    Returns:
        fig: Plotly figure object with enhanced styling
    """
    # Create 1×5 subplot layout (horizontal arrangement)
    fig = make_subplots(
        rows=1, cols=5,
        subplot_titles=['Bus 18', 'Bus 21', 'Bus 30', 'Bus 45', 'Bus 53'],
        horizontal_spacing=0.05  # Reduce spacing between subplots
    )
    
    # Use consistent color scheme with trajectory plot
    method_colors = {
        'RLC-FT': '#0072B2',  # Deep blue
        'Linear': '#999999'   # Gray
    }
    
    # Line styles and widths
    line_width = {
        'RLC-FT': 3,
        'Linear': 2
    }
    
    N = 400
    x_vals = np.linspace(10, 14, N)
    
    # Adjusted safe voltage range
    safe_range = [11.5, 12.5]  # Adjusted per requirements
    
    # Store all traces to ensure correct rendering order
    all_traces = {i: {'background': [], 'lines': []} for i in range(5)}
    
    # Create subplots for each bus
    bus_indices = [0, 1, 2, 3, 4]
    for idx, i in enumerate(bus_indices):
        # In horizontal layout, all plots are in row 1, column is the index+1
        row = 1
        col = idx + 1
        
        baseline_vals = []
        policy_vals = []
        
        # Calculate output values
        for j in range(N):
            state_val = 0.80 + 0.001 * j
            base = np.maximum(state_val - 1.05, 0) - np.maximum(0.95 - state_val, 0)
            baseline_vals.append(-base)
            
            state_tensor = torch.tensor([[state_val]])
            action_tensor = policy_net[i](state_tensor, topology)
            policy_vals.append(float(-action_tensor.detach().cpu().numpy()[0]))

        baseline_vals = np.array(baseline_vals)
        
        # Apply PCHIP smoothing for better curve appearance
        window_size = 20
        policy_vals_smoothed = moving_average(policy_vals, n=window_size)

        
        # Scale linear values for consistency
        baseline_vals_scaled = 5 * baseline_vals
        
        # Add safe voltage range area
        fig.add_shape(
            type="rect",
            x0=safe_range[0], 
            x1=safe_range[1], 
            y0=-10,  # Using very wide y-range
            y1=10,   # that will be auto-scaled later
            fillcolor="rgba(144, 238, 144, 0.2)",
            line=dict(width=0),
            layer="below",
            row=row, col=col
        )
        
        # Add Safe Range to legend (only in first subplot)
        if idx == 0:
            all_traces[idx]['background'].append(
                go.Scatter(
                    x=[None], y=[None],
                    mode='lines',
                    fill='toself',
                    fillcolor="rgba(144, 238, 144, 0.2)",
                    line=dict(width=0),
                    name="Safe Range",
                    showlegend=True
                )
            )
        
        # Add zero line for reference
        fig.add_hline(
            y=0,
            line=dict(color="black", width=1, dash="dot"),
            row=row, col=col
        )
        
        # Add Linear and RLC-FT curves
        all_traces[idx]['lines'].append(
            go.Scatter(
                x=x_vals,
                y=baseline_vals_scaled,
                mode='lines',
                name='Linear',
                line=dict(
                    dash='dash', 
                    color=method_colors['Linear'],
                    width=line_width['Linear']
                ),
                showlegend=(idx == 0)
            )
        )

        all_traces[idx]['lines'].append(
            go.Scatter(
                x=x_vals,
                y=policy_vals_smoothed,
                mode='lines',
                name='RLC-FT',
                line=dict(
                    color=method_colors['RLC-FT'],
                    width=line_width['RLC-FT']
                ),
                showlegend=(idx == 0)
            )
        )
        
        # Calculate y-axis range based on data
        y_min = min(min(baseline_vals_scaled), min(policy_vals_smoothed)) * 1.1
        y_max = max(max(baseline_vals_scaled), max(policy_vals_smoothed)) * 1.1
        
        # Set axis styles
        fig.update_xaxes(
            title_text="Voltage (kV)" if col == 3 else None,  # Only middle plot gets x-axis title
            title_font=dict(size=14),
            range=[10, 14],
            showgrid=True,
            gridwidth=0.5,
            gridcolor='#E5E5E5',
            zeroline=False,
            row=row, col=col
        )
        
        # Set y-axis styles - only first plot gets y-axis title
        fig.update_yaxes(
            title_text="Q (MVar)" if col == 1 else None,
            title_font=dict(size=14),
            range=[y_min, y_max],  # Dynamically scale y-axis for each subplot
            showgrid=True,
            gridwidth=0.5,
            gridcolor='#E5E5E5',
            zeroline=False,
            row=row, col=col
        )
    
    # Add all traces in the correct order
    for idx in range(5):
        row = 1  # All in row 1 for horizontal layout
        col = idx + 1
        
        # Add background elements first
        for trace in all_traces[idx]['background']:
            fig.add_trace(trace, row=row, col=col)
        
        # Add lines next
        for trace in all_traces[idx]['lines']:
            fig.add_trace(trace, row=row, col=col)

    # Overall layout settings - adjust width/height for horizontal layout
    fig.update_layout(
        font=dict(family='Arial', size=14),
        width=1400,  # Wider for horizontal layout
        height=500,  # Lower height
        margin=dict(l=60, r=30, t=100, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.10,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.8)',
            font=dict(size=14),
            itemsizing='constant'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # First remove any existing legend entries by setting showlegend=False for all traces
    for trace in fig.data:
        trace.showlegend = False

    # Then add explicit legend-only traces in the desired order
    # 1. Linear (first)
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(
                color=method_colors['Linear'],
                width=line_width['Linear'],
                dash='dash'
            ),
            name="Linear",
            showlegend=True
        )
    )

    # 2. RLC-FT (second)
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(
                color=method_colors['RLC-FT'],
                width=line_width['RLC-FT']
            ),
            name="RLC-FT",
            showlegend=True
        )
    )

    # 3. Safe Range (third)
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='lines',
            fill='toself',
            fillcolor="rgba(144, 238, 144, 0.2)",
            line=dict(width=0),
            name="Safe Range",
            showlegend=True
        )
    )

    # Configure export settings for better PDF quality
    import plotly.io as pio
    pio.kaleido.scope.mathjax = None
    
    # Save high-resolution images
    output_path = os.path.join(Config.data_path, 'images', '56bus', 'enhanced_policy_plot.pdf')
    fig.write_image(output_path, scale=2)
    
    # Also save PNG version
    fig.write_image(os.path.join(Config.data_path, 'images', '56bus', 'enhanced_policy_plot.png'), scale=2)
    
    # Display the plot
    fig.show()
    
    return fig


def safe_net_plotly(safe_net):
    """
    用 Plotly 绘制 safe network 策略曲线，每个子图显示一个母线的 Stable-DDPG 与 Linear 比较
    """
    default_colors = pc.qualitative.Plotly  # Plotly 默认颜色序列
    color_linear = default_colors[0]
    color_safe = default_colors[1]
    fig = make_subplots(rows=1, cols=5,
                        subplot_titles=['Bus 18', 'Bus 21', 'Bus 30', 'Bus 45', 'Bus 53'])
    N = 400
    for i in range(len(safe_net)):
        baseline_vals = []
        safe_vals = []
        for j in range(N):
            state_val = 0.80 + 0.001 * j
            base = np.maximum(state_val - 1.05, 0) - np.maximum(0.95 - state_val, 0)
            baseline_vals.append(-base)
            # safe_net[i].get_action 接受列表输入，返回单个数值
            action = safe_net[i].get_action([state_val])
            safe_vals.append(-float(action))
        baseline_vals = np.array(baseline_vals)
        baseline_vals_scaled = 2 * baseline_vals
        x_vals = np.linspace(10, 14, N)
        # 仅在第一列显示图例，其余子图同组 trace 设为不显示图例
        showlegend = True if i == 0 else False

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=baseline_vals_scaled,
            mode='lines',
            name='Linear',
            showlegend=showlegend,
            line=dict(dash='dash', color=color_linear)
        ), row=1, col=i+1)

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=safe_vals,
            mode='lines',
            name='Safe-DDPG',
            showlegend=showlegend,
            line=dict(color=color_safe)
        ), row=1, col=i+1)

    # 保证仅在第一个子图显示y轴标题，第三个子图显示x轴标题
    fig.update_yaxes(title_text="Q (MVar)", row=1, col=1)
    fig.update_xaxes(title=dict(text="Voltage (kV)", standoff=25), row=1, col=3)
    fig.update_layout(
        width=1400,
        height=500,
        showlegend=True,
        xaxis=dict(
            showline=True,
            mirror=True,
            showgrid=True,
        ),
        yaxis=dict(
            showline=True,
            mirror=True,
            showgrid=True,
        ),
    )
    output_path = os.path.join(Config.data_path, 'images', '56bus', 'safe_net_plot.pdf')
    import plotly.io as pio
    pio.kaleido.scope.mathjax = None
    fig.write_image(output_path)
    fig.show()


def x_policy_plotly(policy_net):
    """
    用 Plotly 绘制不同拓扑下的 RLC-FT 策略曲线，所有情形绘制在单个图中
    """
    import plotly.graph_objects as go
    fig = go.Figure()
    N = 400
    for i in range(5):
        policy_vals = []
        # 对于每个拓扑情形，通过 reset_topo 获得对应拓扑设定
        state, topo, senario = env.reset_topo(seed=i)
        topo_tensor = torch.cuda.FloatTensor(topo).unsqueeze(0)
        for j in range(N):
            state_tensor = torch.tensor([[0.80 + 0.001 * j]])
            action_tensor = policy_net[2](state_tensor, topo_tensor)
            policy_vals.append(float(-action_tensor.detach().cpu().numpy()[0]))
        policy_vals_smoothed = moving_average(np.array(policy_vals), n=20)
        x_vals = np.linspace(10, 14, N)
        fig.add_trace(go.Scatter(x=x_vals, y=policy_vals_smoothed,
                                 mode='lines',
                                 name=f'Topology {i}'))
    fig.update_layout(
        font=dict(size=16),
        width=700,
        height=500,
        # margin=dict(l=30, r=30, t=30, b=30),   # Uncomment to adjust margins
        margin=dict(r=30,t=30,b=60),
        xaxis_title='Voltage (kV)',
        yaxis_title='Q (MVar)',
        xaxis=dict(
            showgrid=True,
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            tickfont=dict(size=12),
            showgrid=True,
            zeroline=True,
            zerolinecolor='lightgray'
        ),
        legend=dict(
            x=1,
            y=1,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255,255,255,1.0)'
        ),
    )
    output_path = os.path.join(Config.data_path, 'images', '56bus', 'x_policy_plot.pdf')
    import plotly.io as pio
    pio.kaleido.scope.mathjax = None
    fig.write_image(output_path)
    fig.show()
    

### load nn model parameter from saved model 
for i in range(agent_num):
    topology_net = TopologyNet(topology_dim=55, output_dim=1, hidden_dim=Config.topology_hidden_dim)
    policy_net = FlexiblePolicyNet(env=env, topology_net=topology_net, obs_dim=1, action_dim=1, hidden_dim=Config.hidden_dim_56bus).to(device)
    agent_policy_net.append(policy_net)

for i in range(agent_num):
    policy_net = SafePolicyNetwork(env=env, obs_dim=1, action_dim=1, hidden_dim=100).to(device)
    safe_agent_net.append(policy_net)

for i in range(agent_num):
    #value_net_dict = torch.load(f'check_points/value_net/2023-06-19/Step_200_Seed_12_a{i}.pth')
    #policy_net_dict = torch.load(f'check_points/policy_net/2023-08-09/Step_250_Seed_23_a{i}.pth')
    #policy_net_dict = torch.load(f'check_points/policy_net/2023-08-15/Step_900_Seed_33_a{i}.pth')
    #policy_net_dict = torch.load(os.path.join(Config.data_path,f'check_points/policy_net/2023-09-21/Step_900_Seed_10_a{i}.pth'))
    policy_net_dict = torch.load(os.path.join(Config.data_path,f'check_points/policy_net/2025-02-18/Step_500_Seed_4_a{i}.pth'))

    agent_policy_net[i].load_state_dict(policy_net_dict)

for i in range(agent_num):
    #value_net_dict = torch.load(f'D:/Code/Python/StableRL_VoltageCtrl-main/saved_models/2023-06-19/SafeDDPG_value_Step_200_a{i}.pth')
    policy_net_dict = torch.load(f'D:/Code/Python/StableRL_VoltageCtrl-main/saved_models/stable_ddpg/policy_net_checkpoint_a{i}.pth')

    safe_agent_net[i].load_state_dict(policy_net_dict)

#enhanced_policy_plot(agent_policy_net, topology)
#x_policy_plotly(agent_policy_net)
#safe_net_plotly(safe_agent_net)

episode_reward = 0
episode_control = 0
voltage = []
q = []
cost = []

last_action = np.zeros((agent_num,1))

done_record = True
state, topology, senario = env.reset_topo(seed=env_seed)      #change topology
# state, topology, senario = env.reset(seed=env_seed)             #not change
topology = torch.cuda.FloatTensor(topology).unsqueeze(0)
for t in range(100):
    action = []
    for i in range(agent_num):
        action_agent = agent_policy_net[i](torch.cuda.FloatTensor(state[i].reshape(1,)).unsqueeze(0), topology)
        action_agent = action_agent.detach().cpu().numpy()[0]
        action.append(action_agent)

    if np.min(action) < -1.0 or np.max(action) > 1.0:
        logger.warning('control output saturated! min is {}, max is {}', np.min(action), np.max(action))

    action = last_action -np.asarray(action)
    
    last_action = np.copy(action)
    
    try:
        next_state, reward, done = env.step(action)
    except:
        logger.error(sys.exc_info())
        logger.error('power flow not converge at {}', t)
        break

    if done and done_record:
        logger.info('RLC-FT stable at step {}', t)
        logger.info('RLC-FT stable cost is {}', episode_control)
        done_record = False

    voltage.append(state)

    q.append(action)

    state = next_state
    
    episode_reward += reward
    
    cost.append(-reward)
    
    episode_control += LA.norm(action, 2)

    # if done:
    #     break

voltage_RL = np.asarray(voltage)
q_RL =  np.asarray(q)
cost_RL =  np.asarray(cost)
logger.info('control cost of flexible controller is {}',episode_control)
logger.info('objective of flexible controller is {}', np.sum(cost_RL))

### test the base line controller
state, topology, senario = env.reset_topo(seed=env_seed)      #change topology
# state, topology, senario = env.reset(seed=env_seed)             #not change
episode_reward = 0
episode_control = 0
num_agent = 5
voltage = []
q = []
cost = []

last_action = np.zeros((num_agent,1))
done_record = True
for t in range(100):
    state1 = np.asarray(state-env.vmax)
    state2 = np.asarray(env.vmin-state)
    d_v = (np.maximum(state1, 0)-np.maximum(state2, 0)).reshape((num_agent,1))
    
    action = (last_action - 10*d_v)
    
    last_action = np.copy(action)
    
    try:
        next_state, reward, done = env.step(action)
    except:
        logger.error(sys.exc_info())
        logger.error('power flow not converge at {}', t)
        break

    if done and done_record:
        logger.info('Linear stable at step {}', t)
        logger.info('Linear stable cost is {}', episode_control)
        done_record = False

    voltage.append(state)

    q.append(action)

    state = next_state
    
    episode_reward += reward
    
    cost.append(-reward)
    
    episode_control += LA.norm(action, 2)

    # if done:
    #     break

voltage_baseline = np.asarray(voltage)
q_baseline =  np.asarray(q)
cost_baseline =  np.asarray(cost)
logger.info('control cost of linear controller is {}',episode_control)
logger.info('objective of linear controller is {}', np.sum(cost_baseline))

### test the safe policy net
state, topology, senario = env.reset_topo(seed=env_seed)      #change topology
# state, topology, senario = env.reset(seed=env_seed)             #not change
episode_reward = 0
episode_control = 0
num_agent = 5
safe_voltage = []
safe_q = []
safe_cost = []

last_action = np.zeros((num_agent,1))
done_record = True
for t in range(100):
    action = []
    for i in range(num_agent):
        action_agent = safe_agent_net[i].get_action(torch.cuda.FloatTensor([state[i]]).float().reshape(1,1))
        action.append(action_agent)

    if 5*np.min(action) < -1.0 or 5*np.max(action) > 1.0:
        logger.warning('control output saturated! min is {}, max is {}', 5*np.min(action), 5*np.max(action))
    
    action = last_action - 5*np.asarray(action).reshape((num_agent, 1))
    
    last_action = np.copy(action)
    
    try:
        next_state, reward, done = env.step(action)
    except:
        logger.error(sys.exc_info())
        logger.error('power flow not converge at {}', t)
        break

    if done and done_record:
        logger.info('Safe-DDPG stable at step {}', t)
        logger.info('Safe-DDPG stable cost is {}', episode_control)
        done_record = False

    safe_voltage.append(state)

    safe_q.append(action)

    state = next_state
    
    episode_reward += reward
    
    safe_cost.append(-reward)
    
    episode_control += LA.norm(action, 2)

    # if done:
    #     break

safe_voltage = np.asarray(safe_voltage)
safe_q =  np.asarray(safe_q)
safe_cost =  np.asarray(safe_cost)
logger.info('control cost of safe-DDPG is {}',episode_control)
logger.info('objective of linear controller is {}', np.sum(safe_cost))


bus_titles = ['Bus 18', 'Bus 21', 'Bus 30', 'Bus 45', 'Bus 53']
default_colors = pc.qualitative.Plotly

# ---------------------------
# Figure 1: Control Output (Q values) & cost
# ---------------------------

# Create subplots: 1 row, 2 columns
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Controller Output", "Cost"],
    horizontal_spacing=0.12
)
# Update the title annotations using NATURE_CONFIG
for i, annotation in enumerate(fig.layout.annotations[:2]):
    annotation.font.size = NATURE_CONFIG["font_title"]
    annotation.font.family = "Arial"

# ---------------------------
# Left subplot: Controller Output (Q values)
# ---------------------------
iterations = np.arange(q_RL.shape[0])
selected_buses = [0, 2, 4]

# Method colors (consistent across all plots) - Nature-friendly option 1
method_colors = {
    'RLC-FT': "#0072B2",    # Vibrant blue
    'Safe-DDPG': "#9D9204", # Brick red (better for colorblind visibility)
    'Linear': '#555555'     # Darker gray (better contrast)
}

# Method line styles
method_styles = {
    'RLC-FT': dict(dash='solid', width=5),
    'Safe-DDPG': dict(dash='dashdot', width=3),
    'Linear': dict(dash='dash', width=2.5)
}

# Bus brightness variations
bus_brightness = {
    0: 1.2,  # Bus 18: 20% brighter
    2: 1.0,  # Bus 30: base brightness
    4: 0.8   # Bus 53: 20% darker
}

# Function to adjust color brightness
def adjust_brightness(hex_color, factor):
    # Convert hex to RGB
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Adjust brightness
    new_rgb = [min(255, int(c * factor)) for c in rgb]
    
    # Convert back to hex
    return f'#{new_rgb[0]:02x}{new_rgb[1]:02x}{new_rgb[2]:02x}'

# Add traces for each bus and method
for i, bus_idx in enumerate(selected_buses):
    for method in ['RLC-FT', 'Safe-DDPG', 'Linear']:
        # Adjust color based on bus
        base_color = method_colors[method]
        adjusted_color = adjust_brightness(base_color, bus_brightness[bus_idx])
        
        # Get data based on method
        if method == 'RLC-FT':
            y_data = q_RL[:, bus_idx, 0]
        elif method == 'Safe-DDPG':
            y_data = safe_q[:, bus_idx, 0]
        else:  # Linear
            y_data = q_baseline[:, bus_idx, 0]
        
        # Add trace
        fig.add_trace(go.Scatter(
            x=iterations,
            y=y_data,
            mode='lines',
            name=f"{bus_titles[bus_idx]} ({method})",
            line=dict(color=adjusted_color, **method_styles[method]),
            legendgroup=f"bus{bus_idx}_{method}",
        ), row=1, col=1)

# ---------------------------
# Right subplot: Cost Plot with Custom Legend (Vertical in Upper Right)
# ---------------------------
iterations_cost = np.arange(len(cost_RL))

# RLC-FT
fig.add_trace(go.Scatter(
    x=iterations_cost,
    y=cost_RL,
    mode='lines',
    name="RLC-FT (Cost)",
    line=dict(color=method_colors['RLC-FT'], **method_styles['RLC-FT']),
), row=1, col=2)

# Linear
fig.add_trace(go.Scatter(
    x=iterations_cost,
    y=cost_baseline,
    mode='lines',
    name="Linear (Cost)",
    line=dict(color=method_colors['Linear'], **method_styles['Linear']),
), row=1, col=2)

# Safe-DDPG
fig.add_trace(go.Scatter(
    x=iterations_cost,
    y=safe_cost,
    mode='lines',
    name="Safe-DDPG (Cost)",
    line=dict(color=method_colors['Safe-DDPG'], **method_styles['Safe-DDPG']),
), row=1, col=2)


# Update x-axis properties for both subplots
fig.update_xaxes(
    title_text="Iteration Steps",
    range=[0, 15],
    showgrid=True,
    gridwidth=0.5,
    gridcolor='#E5E5E5',
    zeroline=False,
    tickfont=dict(size=NATURE_CONFIG["font_axis"]),
    title_font=dict(size=NATURE_CONFIG["font_base"]),
    row=1, col=1
)

fig.update_xaxes(
    title_text="Iteration Steps",
    range=[0, 20],
    showgrid=True,
    gridwidth=0.5,
    gridcolor='#E5E5E5',
    zeroline=False,
    tickfont=dict(size=NATURE_CONFIG["font_axis"]),
    title_font=dict(size=NATURE_CONFIG["font_base"]),
    row=1, col=2
)

# Update y-axis properties for both subplots
fig.update_yaxes(
    title_text="Q (MVar)",
    showgrid=True,
    gridwidth=0.5,
    gridcolor='#E5E5E5',
    zeroline=False,
    tickfont=dict(size=NATURE_CONFIG["font_axis"]),
    title_font=dict(size=NATURE_CONFIG["font_base"]),
    row=1, col=1
)

fig.update_yaxes(
    title_text="Cost",
    showgrid=True,
    gridwidth=0.5,
    gridcolor='#E5E5E5',
    zeroline=False,
    tickfont=dict(size=NATURE_CONFIG["font_axis"]),
    title_font=dict(size=NATURE_CONFIG["font_base"]),
    row=1, col=2
)

# Add subplot labels (a) and (b) OUTSIDE the plots
fig.add_annotation(
    text="<b>(a)</b>",
    x=-0.01,  # Position in paper coordinates
    y=1.10,  # Position above the subplot
    xref="paper",
    yref="paper",
    showarrow=False,
    font=dict(size=NATURE_CONFIG["font_title"]),
)

fig.add_annotation(
    text="<b>(b)</b>",
    x=0.57,  # Position in paper coordinates - adjusted for second subplot
    y=1.10,  # Position above the subplot
    xref="paper",
    yref="paper",
    showarrow=False,
    font=dict(size=NATURE_CONFIG["font_title"]),
)

# Update layout properties 
fig.update_layout(
    font=dict(family='Arial', size=NATURE_CONFIG["font_base"]),
    width=NATURE_CONFIG["width"],
    height=800,
    margin=dict(l=60, r=30, t=80, b=150),  # Increased top margin for subplot labels
    plot_bgcolor='white',
    paper_bgcolor='white',
    legend=dict(
        orientation="h",
        y=-0.25,        
        x=0.5,
        xanchor="center",
        yanchor="top",
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='lightgray',
        borderwidth=0.5,
        font=dict(size=NATURE_CONFIG["font_legend"]),
        #entrywidth=250,
        traceorder='grouped',
    ),
)

# Export high-resolution image
import plotly.io as pio
pio.kaleido.scope.mathjax = None
output_path = os.path.join(Config.data_path, "images","56bus", "combined_plots_nature.pdf")
fig.write_image(output_path, scale=2)

# Also save as PNG
fig.write_image(os.path.join(Config.data_path, "images","56bus", "combined_plots_nature.png"), scale=2)

# Display the figure
fig.show()


# ---------------------------
# Figure: Enhanced Voltage Trajectory Comparison
# ---------------------------

# Select 4 buses to display 
index = [0, 1, 2, 3]
bus_titles = ['Bus 18', 'Bus 21', 'Bus 30', 'Bus 45', 'Bus 53']

# Create 2×2 subplot layout with optimized spacing
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[f'(a) {bus_titles[0]}', f'(b) {bus_titles[1]}', 
                    f'(c) {bus_titles[2]}', f'(d) {bus_titles[3]}'],
    vertical_spacing=0.15,
    horizontal_spacing=0.10
)

# Define consistent color scheme and line styles
method_colors = {
    'RLC-FT': '#0072B2',      # Deep blue for primary method
    'Linear': '#999999',      # Gray for baseline method
    'Safe-DDPG': '#D55E00'    # Orange-red for comparison method
}
method_lines = {
    'RLC-FT': 'solid',
    'Linear': 'dash',
    'Safe-DDPG': 'dashdot'
}
line_width = {
    'RLC-FT': 3,
    'Linear': 2,
    'Safe-DDPG': 2
}

# Define method names
method_names = {
    'RLC-FT': 'RLC-FT*',
    'Linear': 'Linear',
    'Safe-DDPG': 'Safe-DDPG'
}

steps = list(range(voltage_RL.shape[0]))

# Function to create smooth interpolated curve - using more sophisticated smoothing
def smooth_curve(x, y, num_points=200):
    """Create a smoother curve by interpolating between points"""
    x_new = np.linspace(min(x), max(x), num_points)
    
    # Try to use a more sophisticated smoothing method when possible
    try:
        # Use PCHIP interpolation for smoother curves without overshooting
        from scipy.interpolate import PchipInterpolator
        pchip = PchipInterpolator(x, y)
        y_new = pchip(x_new)
    except:
        # Fall back to cubic spline if PCHIP isn't available
        f = interpolate.interp1d(x, y, kind='cubic')
        y_new = f(x_new)
        
    return x_new, y_new

# Store all traces to add them in the correct order later
all_traces = {i: {'background': [], 'lines': [], 'markers': []} for i in range(4)}

# Process all subplots and prepare traces
for i, bus_idx in enumerate(index):
    row, col = (i // 2) + 1, (i % 2) + 1
    
    # Add semi-transparent safe range area - using shape instead of scatter to avoid corner points
    fig.add_shape(
        type="rect",
        x0=0, y0=12.0,
        x1=30, y1=12.6,
        fillcolor="rgba(144, 238, 144, 0.2)",
        line=dict(width=0),
        layer="below",
        row=row, col=col
    )
    
    # Add "Safe Range" to legend only once using a separate invisible trace for the legend
    if i == 0:
        all_traces[i]['background'].append(
            go.Scatter(
                x=[None], y=[None],
                mode='lines',
                fill='toself',
                fillcolor="rgba(144, 238, 144, 0.2)",
                line=dict(width=0),
                name="Safe Range",
                showlegend=True
            )
        )
    
    # Add target voltage line
    all_traces[i]['background'].append(
        go.Scatter(
            x=[0, 30],
            y=[12.0, 12.0],
            mode='lines',
            line=dict(color='#E69F00', width=1.5),
            name="Nominal Voltage",
            showlegend=False
        )
    )
    
    # Add upper limit line
    all_traces[i]['background'].append(
        go.Scatter(
            x=[0, 30],
            y=[12.6, 12.6],
            mode='lines',
            line=dict(color='#56B4E9', width=1.5),
            name="Voltage Upper Limit",
            showlegend=False
        )
    )
    
    # Create containers for each method's data
    method_data = {}
    
    # First process and collect all method data
    for method_key, method_name in method_names.items():
        data = None
        if method_key == 'RLC-FT':
            data = voltage_RL
        elif method_key == 'Linear':
            data = voltage_baseline
        else:  # Safe-DDPG
            data = safe_voltage
            
        y_values = (12*data[:, bus_idx]).tolist()
        
        # Create smooth interpolated curves with more points (200)
        x_smooth, y_smooth = smooth_curve(steps, y_values, num_points=200)
        
        # Store data for later plotting (to control layer order)
        method_data[method_key] = {
            'x': x_smooth, 
            'y': y_smooth, 
            'raw_y': y_values
        }
    
    # Add curves in specific order (baselines first, RLC-FT last to be on top)
    # Add Linear and Safe-DDPG first
    for method_key in ['Linear', 'Safe-DDPG']:
        all_traces[i]['lines'].append(
            go.Scatter(
                x=method_data[method_key]['x'],
                y=method_data[method_key]['y'],
                mode='lines',
                name=method_names[method_key],
                line=dict(
                    color=method_colors[method_key], 
                    dash=method_lines[method_key],
                    width=line_width[method_key]
                ),
                showlegend=(i == 0)  # Only show legend in first subplot
            )
        )
    
    # Now add RLC-FT (so it's on top of other method lines)
    all_traces[i]['lines'].append(
        go.Scatter(
            x=method_data['RLC-FT']['x'],
            y=method_data['RLC-FT']['y'],
            mode='lines',
            name=method_names['RLC-FT'],
            line=dict(
                color=method_colors['RLC-FT'], 
                dash=method_lines['RLC-FT'],
                width=line_width['RLC-FT']
            ),
            showlegend=(i == 0)  # Only show legend in first subplot
        )
    )
    
    # Find intersection with upper limit line for RLC-FT
    # Use interpolated data for more precise intersection
    x_smooth = method_data['RLC-FT']['x']
    y_smooth = method_data['RLC-FT']['y']
    
    # Find where curve crosses from above to below the boundary
    crossings = []
    for j in range(1, len(y_smooth)):
        # Check if current point crosses the upper boundary from above
        if y_smooth[j-1] > 12.6 and y_smooth[j] <= 12.6:
            # Linear interpolation to find exact crossing point
            x1, y1 = x_smooth[j-1], y_smooth[j-1]
            x2, y2 = x_smooth[j], y_smooth[j]
            
            if y1 != y2:  # Avoid division by zero
                x_intersect = x1 + (12.6 - y1) * (x2 - x1) / (y2 - y1)
                crossings.append((x_intersect, 12.6))
    
    # If we found an intersection point, use it; otherwise try finding closest point
    if crossings:
        x_mark, y_mark = crossings[0]  # Use first crossing
    else:
        # Find point closest to upper boundary
        diffs = [abs(y - 12.6) for y in y_smooth]
        min_idx = diffs.index(min(diffs))
        x_mark, y_mark = x_smooth[min_idx], 12.6  # Force y to be on boundary
    
    # Add marker at intersection point (will be added last in the rendering process)
    all_traces[i]['markers'].append(
        go.Scatter(
            x=[x_mark],
            y=[y_mark],
            mode='markers',
            marker=dict(
                symbol='circle',
                size=10,  # Increased size for better visibility
                color=method_colors['RLC-FT'],
                line=dict(color='white', width=2)  # Increased border for visibility
            ),
            name='Entry to Safe Range',
            showlegend=False
        )
    )
    
    # Add annotations - using more standard terminology
    if i == 0:  # Add to second subplot
        fig.add_annotation(
            x=22, y=12.02,
            text="Nominal Voltage",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.7)",
            font=dict(size=10),
            row=row, col=col
        )
    
    if i == 0:  # Add to first subplot
        fig.add_annotation(
            x=22, y=12.62,
            text="Voltage Upper Limit",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.7)",
            font=dict(size=10),
            row=row, col=col
        )
    
    # Set axis ranges and labels with improved grid setting for PDF export
    fig.update_xaxes(
        range=[0, 30],
        showgrid=True,
        gridwidth=0.5,
        gridcolor='#E5E5E5',  # Lighter solid color for PDF compatibility
        zeroline=False,
        title_text="Iteration Steps" if row == 2 else None,
        title_font=dict(size=14),
        row=row, col=col
    )
    
    fig.update_yaxes(
        range=[11.8, 13.2],
        showgrid=True,
        gridwidth=0.5,
        gridcolor='#E5E5E5',  # Lighter solid color for PDF compatibility
        zeroline=False,
        title_text="Bus Voltage (kV)" if col == 1 else None,
        title_font=dict(size=14),
        row=row, col=col
    )

# Add all traces in the correct order to ensure proper layering
for i in range(4):
    row, col = (i // 2) + 1, (i % 2) + 1
    
    # Add background elements first
    for trace in all_traces[i]['background']:
        fig.add_trace(trace, row=row, col=col)
    
    # Add lines next
    for trace in all_traces[i]['lines']:
        fig.add_trace(trace, row=row, col=col)
    
    # Add markers absolutely last
    for trace in all_traces[i]['markers']:
        fig.add_trace(trace, row=row, col=col)

# Overall layout settings
fig.update_layout(
    font=dict(family='Arial', size=14),
    width=900,
    height=700,
    margin=dict(l=60, r=30, t=100, b=60),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.10,
        xanchor="center",
        x=0.5,
        bgcolor='rgba(255,255,255,0.8)',
        font=dict(size=14),
        itemsizing='constant',
        traceorder='normal'  # Use normal trace order in legend
    ),
    plot_bgcolor='white',
    paper_bgcolor='white',
    # Title can be easily commented out for journal submission
    title=dict(
        text="Voltage Response Comparison of Different Control Methods",
        font=dict(size=16),
        y=0.99,
        x=0.5,  # Center the title
        xanchor='center'  # Ensure title is properly centered
    )
)


# Configure export settings for better PDF quality
pio.kaleido.scope.mathjax = None

# Save images with high DPI
fig.update_layout(title=None)  # Remove title for image export
fig.write_image(os.path.join(Config.data_path, 'images','56bus', 'enhanced_trajectory_plot.pdf'), scale=2)
fig.write_image(os.path.join(Config.data_path, 'images','56bus', 'enhanced_trajectory_plot.png'), scale=2)

fig.show()