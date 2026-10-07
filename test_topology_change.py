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
from pandapower.plotting.plotly import pf_res_plotly

from loguru import logger

# plt.style.use('ieee')
# Update global plot parameters
plt.rcParams.update({
    "figure.figsize": (3.5, 2.5),   # default figure size for ieee
    "axes.labelsize": 8,  # Font size for axis labels
    "axes.titlesize": 8,  # Font size for titles
    "xtick.labelsize": 7,  # Font size for x-axis tick labels
    "ytick.labelsize": 7,  # Font size for y-axis tick labels
    "legend.fontsize": 6,  # Font size for legend
    "lines.linewidth": 0.8,  # Line width
    # "savefig.dpi": 300,  # Resolution for saving figures
    "figure.dpi": 200,
    "figure.autolayout": True,  # Automatic layout to prevent clipping
    "font.family": "serif",  # Font family
    "font.serif": ["Times New Roman"],  # Use Times New Roman font
    "axes.prop_cycle": plt.cycler('color', [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'
        ]) + plt.cycler('linestyle', [
        '-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (3, 5, 1, 5))
])

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

### a simple logger
logger.remove()
logger.add(sys.stderr, level='DEBUG')

env_seed = 1        #10-h  5-h 0-l 1-h 2-l 3-l 4l 7h 8h 9l 6l 11h

agent_num = 5
agent_policy_net = []
safe_agent_net = []

### create testing environment
injection_bus = np.array([18, 21, 30, 45, 53])-1
pp_net = create_56bus()
env = VoltageCtrl_Env(pp_net, injection_bus)
state, topology, senario = env.reset_topo(seed=env_seed)      #change topology
# state, topology, senario = env.reset(seed=env_seed)             #not change
topology = torch.cuda.FloatTensor(topology).unsqueeze(0)

def moving_average(a, n=3):
    # Padding the array to maintain the length after convolution.
    pad = np.pad(a, (n//2, n-1-n//2), mode='edge')
    ret = np.convolve(pad, np.ones(n), mode='valid') / n
    return ret
# 使用双重移动平均进行更强的平滑
def double_moving_average(data, n=15):
    """先应用一次移动平均，然后对结果再应用一次移动平均"""
    # 第一次移动平均
    smoothed1 = np.convolve(data, np.ones(n)/n, mode='valid')
    # 对结果进行填充，保持长度一致
    pad_size = len(data) - len(smoothed1)
    smoothed1 = np.pad(smoothed1, (pad_size//2, pad_size - pad_size//2), mode='edge')
    # 第二次移动平均
    smoothed2 = np.convolve(smoothed1, np.ones(n)/n, mode='valid')
    # 对结果进行填充，保持长度一致
    pad_size = len(data) - len(smoothed2)
    smoothed2 = np.pad(smoothed2, (pad_size//2, pad_size - pad_size//2), mode='edge')
    return smoothed2

def enhanced_policy_plot(policy_net, topology):
    """
    Enhanced policy visualization with consistent styling with trajectory plot
    
    Args:
        policy_net: The policy network models for different buses
        topology: The power system topology indicator
        
    Returns:
        fig: Plotly figure object with enhanced styling
    """
    # Create 2×3 subplot layout (more balanced layout, last subplot empty)
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f'(a) Bus 18', f'(b) Bus 21', f'(c) Bus 30', 
                        f'(d) Bus 45', f'(e) Bus 53', ''],
        vertical_spacing=0.15,
        horizontal_spacing=0.10
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
        row = (idx // 3) + 1
        col = (idx % 3) + 1
        
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
        window_size = 10  # 窗口大小，越大越平滑
        smoothed = moving_average(policy_vals, n=window_size)
        # 再用PCHIP插值确保点数一致
        try:
            from scipy.interpolate import PchipInterpolator
            pchip = PchipInterpolator(range(len(smoothed)), smoothed)
            policy_vals_smoothed = pchip(np.linspace(0, len(smoothed)-1, N))
        except:
            # 如果PCHIP不可用，直接使用平滑结果
            policy_vals_smoothed = smoothed
        
        # Scale linear values for consistency
        baseline_vals_scaled = 5 * baseline_vals
        
        # Add safe voltage range area
        fig.add_shape(
            type="rect",
            x0=safe_range[0], y0=-10,  # Set large enough y range
            x1=safe_range[1], y1=10,
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
        fig.add_shape(
            type="line",
            x0=10, y0=0,
            x1=14, y1=0,
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
        
        # Set axis ranges
        y_min = min(min(baseline_vals_scaled), min(policy_vals_smoothed)) * 1.1
        y_max = max(max(baseline_vals_scaled), max(policy_vals_smoothed)) * 1.1
        
        # Set axis styles
        fig.update_xaxes(
            title_text="Voltage (kV)" if (row == 2 or (row == 1 and col == 3)) else None,
            title_font=dict(size=14),
            range=[10, 14],
            showgrid=True,
            gridwidth=0.5,
            gridcolor='#E5E5E5',
            zeroline=False,
            row=row, col=col
)
        
        # Set axis styles
        fig.update_yaxes(
            title_text="Q (MVar)" if col == 1 else None,
            title_font=dict(size=14),
            range=[y_min, y_max],
            showgrid=True,
            gridwidth=0.5,
            gridcolor='#E5E5E5',
            zeroline=False,
            row=row, col=col
        )
    
    # Add all traces in the correct order
    for idx in range(5):
        row = (idx // 3) + 1
        col = (idx % 3) + 1
        
        # Add background elements first
        for trace in all_traces[idx]['background']:
            fig.add_trace(trace, row=row, col=col)
        
        # Add lines next
        for trace in all_traces[idx]['lines']:
            fig.add_trace(trace, row=row, col=col)

    # Overall layout settings
    fig.update_layout(
        font=dict(family='Arial', size=14),
        width=1000,
        height=700,
        margin=dict(l=60, r=30, t=100, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.10,  # Adjusted higher per your request
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.8)',
            font=dict(size=14),
            itemsizing='constant'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
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


def plot_safe_net(net):
    fig, axs = plt.subplots(1, 5, figsize=(7.16,2.5))
    title = ['Bus 18', 'Bus 21', 'Bus 30', 'Bus 45', 'Bus 53']
    for i in range(agent_num):
        N = 400
        s_array = np.zeros(N,)
        
        a_array_baseline = np.zeros(N,)
        a_array = np.zeros(N,)
        
        for j in range(N):
            state = np.array([0.8+0.001*j])
            s_array[j] = state

            action_baseline = (np.maximum(state-1.05, 0)-np.maximum(0.95-state, 0)).reshape((1,))
        
            action = net[i].get_action([state])
            
            a_array_baseline[j] = -action_baseline[0]
            a_array[j] = -action

        axs[i].plot(12*s_array, 2*a_array_baseline, '-.', label = 'Linear')
        axs[i].plot(12*s_array, a_array, label = 'Stable-DDPG')
        axs[i].legend(loc='lower left')
        axs[i].grid(True)

def plot_x_policy(policy_net, topology):
    fig, axs = plt.subplots()
    axs.set_ylabel('Q(MVar)')
    axs.set_xlabel('Voltage(kV)')
    for i in range(5):
        # plot policy
        N = 400
        s_array = np.zeros(N,)
        
        a_array_baseline = np.zeros(N,)
        a_array = np.zeros(N,)
        #topology = torch.cuda.FloatTensor(env.topology_init * np.random.uniform(0.7,1.3)).unsqueeze(0)
        state, topology, senario = env.reset_topo(seed=i)
        topology = torch.cuda.FloatTensor(topology).unsqueeze(0)
        
        for j in range(N):
            state = torch.tensor([[0.80+0.001*j]])
            s_array[j] = state

            action_baseline = (np.maximum(state.cpu()-1.05, 0)-np.maximum(0.95-state.cpu(), 0)).reshape((1,))
        
            action = policy_net[0](state, topology)
            action = action.detach().cpu().numpy()[0]
            
            a_array_baseline[j] = -action_baseline[0]
            a_array[j] = -action

        a_array_s = moving_average(a_array, n = 20)
        axs.plot(12*s_array, a_array_s, label = f'Topology {i}')
        axs.legend(loc='best')
        plt.pause(0.1)

    axs.grid(True)
    fig.savefig(Config.data_path+'images/'+'topology.eps', format='eps', bbox_inches='tight')

def policy_plotly(policy_net, topology):
    """
    用 Plotly 绘制各母线的策略曲线，每个子图显示一个母线的 RLC-FT 策略与基线（Linear）策略比较，
    """
    default_colors = pc.qualitative.Plotly  # Plotly 默认颜色序列
    color_linear = default_colors[0]
    color_rlc = default_colors[1]
    fig = make_subplots(rows=1, cols=5,
                        subplot_titles=['Bus 18', 'Bus 21', 'Bus 30', 'Bus 45', 'Bus 53'])
    N = 400
    for i in range(5):
        baseline_vals = []
        policy_vals = []
        for j in range(N):
            # 计算基线控制值：baseline = max(state-1.05, 0) - max(0.95-state, 0)
            state_val = 0.80 + 0.001 * j
            base = np.maximum(state_val - 1.05, 0) - np.maximum(0.95 - state_val, 0)
            baseline_vals.append(-base)  # 取负值
            state_tensor = torch.tensor([[state_val]])
            action_tensor = policy_net[i](state_tensor, topology)
            policy_vals.append(float(-action_tensor.detach().cpu().numpy()[0]))

        baseline_vals = np.array(baseline_vals)
        policy_vals_smoothed = moving_average(np.array(policy_vals), n=20)
        baseline_vals_scaled = 5 * baseline_vals
        
        x_vals = np.linspace(10, 14, N)
        
        # 仅在第一列显示图例，其余子图同组 trace 设为不显示图例
        showlegend = True if i == 0 else False

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=baseline_vals_scaled,
            mode='lines',
            name='Linear',
            legendgroup='Linear',
            showlegend=showlegend,
            line=dict(dash='dash', color=color_linear)
        ), row=1, col=i+1)

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=policy_vals_smoothed,
            mode='lines',
            name='RLC-FT',
            legendgroup='RLC-FT',
            showlegend=showlegend,
            line=dict(color=color_rlc)
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
    
    output_path = os.path.join(Config.data_path, 'images', '56bus', 'policy_plot.pdf')
    import plotly.io as pio
    pio.kaleido.scope.mathjax = None
    fig.write_image(output_path)
    fig.show()


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
            action_tensor = policy_net[0](state_tensor, topo_tensor)
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

def improved_x_policy_plotly(policy_net):
    """
    Plot the improved single-graph comparison of policies across different topologies,
    with optimized x-axis range to highlight differences.
    """
    import plotly.graph_objects as go
    
    # Create figure with better aspect ratio
    fig = go.Figure()
    
    # Use a colorblind-friendly palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Define the safe voltage range
    safe_range = [11.48, 12.52]
    
    # Add safe range area
    fig.add_shape(
        type="rect",
        x0=safe_range[0], y0=-1.2,
        x1=safe_range[1], y1=1.2,
        fillcolor="rgba(144, 238, 144, 0.2)",
        line=dict(width=0),
        layer="below"
    )
    
    # Add zero line
    fig.add_shape(
        type="line",
        x0=10.5, y0=0,
        x1=13.5, y1=0,
        line=dict(color="black", width=1, dash="dash")
    )
    
    # Plot policy curves for different topologies
    N = 400
    x_vals = np.linspace(10, 14, N)
    
    for i in range(5):
        policy_vals = []
        state, topo, senario = env.reset_topo(seed=i)
        topo_tensor = torch.cuda.FloatTensor(topo).unsqueeze(0)
        
        for j in range(N):
            state_tensor = torch.tensor([[0.80 + 0.001 * j]])
            action_tensor = policy_net[2](state_tensor, topo_tensor)
            policy_vals.append(float(-action_tensor.detach().cpu().numpy()[0]))
        
        # Apply Savitzky-Golay filtering for smoother curves
        try:
            from scipy.signal import savgol_filter
            window_size = 25  # Must be odd
            poly_order = 3    # Polynomial order
            policy_vals_smoothed = savgol_filter(policy_vals, window_size, poly_order)
        except:
            # Fallback to moving average if scipy is not available
            policy_vals_smoothed = moving_average(np.array(policy_vals), n=20)
        
        # Add trace with clear labeling
        fig.add_trace(go.Scatter(
            x=x_vals, 
            y=policy_vals_smoothed,
            mode='lines',
            name=f'Topology {i}',
            line=dict(color=colors[i], width=3)
        ))
    
    # Update layout with improved aspect ratio and focused x-axis range
    fig.update_layout(
        font=dict(family='Arial', size=16),
        width=800,
        height=600,
        margin=dict(l=60, r=30, t=30, b=60),
        xaxis_title='Voltage (kV)',
        yaxis_title='Q (MVar)',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            # Focused x-axis range to highlight differences
            range=[10.0, 14.0],
            showgrid=True,
            gridwidth=0.5,
            gridcolor='#E5E5E5',
            zeroline=False,
            tickfont=dict(size=14),
        ),
        yaxis=dict(
            range=[-1.0, 1.0],
            showgrid=True,
            gridwidth=0.5,
            gridcolor='#E5E5E5',
            zeroline=False,
            tickfont=dict(size=14),
        ),
        legend=dict(
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='lightgray',
            borderwidth=1
        ),
    )
    
    # Add "Safe Range" to the legend
    fig.add_trace(go.Scatter(
        x=[None], 
        y=[None], 
        mode='lines',
        fill='toself',
        fillcolor="rgba(144, 238, 144, 0.2)",
        line=dict(width=0),
        name="Safe Range"
    ))
    
    # Export high-resolution images
    output_path = os.path.join(Config.data_path, 'images', '56bus', 'x_policy_plot_improved.pdf')
    import plotly.io as pio
    pio.kaleido.scope.mathjax = None
    fig.write_image(output_path, scale=2)
    fig.write_image(os.path.join(Config.data_path, 'images', '56bus', 'x_policy_plot_improved.png'), scale=2)
    fig.show()
    
    return fig

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

enhanced_policy_plot(agent_policy_net, topology)
improved_x_policy_plotly(agent_policy_net)
# safe_net_plotly(safe_agent_net)


### test the flexible controller
episode_reward = 0
episode_control = 0
voltage = []
q = []
cost = []

last_action = np.zeros((agent_num,1))

done_record = True
state, topology, senario = env.reset(seed=env_seed)
topology = torch.cuda.FloatTensor(topology).unsqueeze(0)
for t in range(50):
    # 在第10次调用 topology_change
    if t == 20:
        logger.info("Changing topology at step {}", t)
        new_topology = env.topology_change(seed=1)
        topology = torch.cuda.FloatTensor(new_topology).unsqueeze(0)
    # 在第20次调用 topology_reset
    if t == 10:
        logger.info("Resetting topology at step {}", t)
        new_topology = env.topology_reset()
        topology = torch.cuda.FloatTensor(new_topology).unsqueeze(0)
    
    action = []
    for i in range(agent_num):
        action_agent = agent_policy_net[i](torch.cuda.FloatTensor(state[i].reshape(1,)).unsqueeze(0), topology)
        action_agent = action_agent.detach().cpu().numpy()[0]
        action.append(action_agent)

    if np.min(action) < -0.3 or np.max(action) > 0.3:
        logger.warning('control output saturated! min is {}, max is {}', np.min(action), np.max(action))

    action = last_action - np.asarray(action)
    last_action = np.copy(action)
    
    try:
        next_state, reward, done = env.step(action)
    except:
        logger.error(sys.exc_info())
        logger.error('power flow not converge at {}', t)
        break

    if done and done_record:
        logger.info('stable at step {}', t)
        logger.info('stable cost is {}', episode_control)
        done_record = False

    voltage.append(state)
    q.append(action)
    state = next_state
    episode_reward += reward
    cost.append(-reward)
    episode_control += LA.norm(action, 2)

voltage_RL = np.asarray(voltage)
q_RL =  np.asarray(q)
cost_RL =  np.asarray(cost)
logger.info('control cost of flexible controller is {}',episode_control)

def plot_voltage_trajectory(voltage_RL, scenario, topology_change_steps=None):
    """
    Create a professional voltage trajectory plot with topology change indicators.
    """
    import plotly.graph_objects as go
    
    # Select bus indices to display
    bus_indices = [0, 1, 2, 3, 4]
    bus_names = ['Bus 18', 'Bus 21', 'Bus 30', 'Bus 45', 'Bus 53']
    
    # Initialize figure
    fig = go.Figure()
    
    # Colorblind-friendly palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Add voltage trajectories
    steps = list(range(voltage_RL.shape[0]))
    for i, idx in enumerate(bus_indices):
        fig.add_trace(go.Scatter(
            x=steps, 
            y=(12*voltage_RL[:, idx]).tolist(),
            mode='lines',
            name=f'{bus_names[i]} (RLC-FT)',
            line=dict(color=colors[i], width=2.5)
        ))
    
    # Add safety limits based on scenario
    if scenario == 0:  # Undervoltage scenario
        y_range = [10, 12.6]
        fig.add_hline(
            y=11.4, 
            line=dict(color='green', dash='dash', width=2),
            annotation=dict(
                text='Voltage Lower Limit',
                xref='paper', yref='y',
                x=1.02, y=11.4,
                showarrow=False,
                font=dict(size=14)
            )
        )
    elif scenario == 1:  # Overvoltage scenario
        y_range = [11.8, 13.5]
        fig.add_hline(
            y=12.6, 
            line=dict(color='green', width=2),
            annotation=dict(
                text='Voltage Upper Limit',
                xref='paper', yref='y',
                x=1.02, y=12.6,
                showarrow=False,
                font=dict(size=14)
            )
        )
    else:  # Both limits scenario
        y_range = [10.8, 13.2]
        fig.add_hline(
            y=12.6, 
            line=dict(color='green', dash='dash', width=2),
            annotation=dict(
                text='Voltage Upper Limit',
                xref='paper', yref='y',
                x=1.02, y=12.6,
                showarrow=False,
                font=dict(size=14)
            )
        )
        fig.add_hline(
            y=11.4, 
            line=dict(color='green', dash='dash', width=2),
            annotation=dict(
                text='Voltage Lower Limit',
                xref='paper', yref='y',
                x=1.02, y=11.4,
                showarrow=False,
                font=dict(size=14)
            )
        )
    
    # Add target voltage line (properly labeled on the right side)
    fig.add_hline(
        y=12.0, 
        line=dict(color='red', width=1.5),
        annotation=dict(
            text='Nominal Voltage',
            xref='paper', yref='y',
            x=1.02, y=12.0,
            showarrow=False,
            font=dict(size=14)
        )
    )
    
    # Handle multiple topology change points
    if topology_change_steps is not None:
        if isinstance(topology_change_steps, (int, float)):
            topology_change_steps = [topology_change_steps]  # Convert single value to list
            
        for i, step in enumerate(topology_change_steps):
            fig.add_vline(
                x=step,
                line=dict(color='rgba(0,0,0,0.5)', dash='dot', width=2),
                annotation=dict(
                    text=f'Topology Change {i+1}' if len(topology_change_steps) > 1 else 'Topology Change',
                    xref='x', yref='paper',
                    x=step, y=1.0,
                    showarrow=False,
                    font=dict(size=14)
                )
            )
    
    # Update layout with professional styling
    fig.update_layout(
        font=dict(family='Arial', size=16),
        width=800,
        height=500,
        margin=dict(l=60, r=100, t=30, b=60),
        xaxis_title='Iteration Steps',
        yaxis_title='Bus voltage (kV)',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            range=[0, 30],  # Shortened range as requested
            showgrid=True,
            gridwidth=0.5,
            gridcolor='#E5E5E5',
            zeroline=False,
            tickfont=dict(size=14),
        ),
        yaxis=dict(
            range=y_range,
            showgrid=True,
            gridwidth=0.5,
            gridcolor='#E5E5E5',
            zeroline=False,
            tickfont=dict(size=14),
        ),
        legend=dict(
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='lightgray',
            borderwidth=1,
            font=dict(size=12)
        ),
    )
    
    # Export high-resolution images
    import plotly.io as pio
    pio.kaleido.scope.mathjax = None
    output_path = os.path.join(Config.data_path, 'images', '56bus', 'voltage_plot.pdf')
    fig.write_image(output_path, scale=2)
    fig.write_image(os.path.join(Config.data_path, 'images', '56bus', 'voltage_plot.png'), scale=2)
    fig.show()
    
    return fig

# Example usage:
state, topology, scenario = env.reset(seed=env_seed)
plot_voltage_trajectory(voltage_RL, scenario=1, topology_change_steps=[10, 20])


