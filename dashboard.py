# dashboard.py
import math
import threading
from typing import List, Dict, Any

import numpy as np
import torch
from dash import Dash, dcc, html, Input, Output, State, no_update, exceptions
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import jsonify

class PlotStore:
    def __init__(self, ENV, title, num_agent, N=40):
        self.ENV = ENV
        self.title = title
        self.num_agent = num_agent
        self.N = N

        # Store precomputed plot data (atomic update)
        self._plot_data = {
            'x': [],
            'y_lin_list': [],
            'y_RL_list': [],
            'valid': False  # Flag indicating whether the data is valid
        }

        # Reward data storage
        self._reward_data = {
            'episodes': [],
            'raw_rewards': [],
            'smoothed_rewards': [],
            'std_rewards': []
        }

        self._policy_version = 0
        self._reward_version = 0

    def bump_policy(self, plot_data=None):
        """Policy-related update, training side passes in precomputed plot data"""
        if plot_data is not None:
            # ✅ Atomic update: prepare complete data first, then replace at once
            new_plot_data = {
                'x': plot_data.get('x', []),
                'y_lin_list': plot_data.get('y_lin_list', []),
                'y_RL_list': plot_data.get('y_RL_list', []),
                'valid': True
            }
            self._plot_data = new_plot_data  # Atomic replacement
        
        self._policy_version += 1
        return self._policy_version
    
    def bump_reward(self):
        """Reward-related update (only triggers reward plot update)"""
        self._reward_version += 1
        return self._reward_version
    
    def get_policy_version(self):
        return self._policy_version
            
    def get_reward_version(self):
        return self._reward_version
        
    def add_reward(self, episode, reward):
        """Called by training side, add new reward data"""
        # First update local data
        episodes = self._reward_data['episodes'] + [episode]
        raw_rewards = self._reward_data['raw_rewards'] + [reward]
        
        # Calculate smoothed reward (moving average, window size 50)
        window = min(50, len(raw_rewards))
        if len(raw_rewards) >= window:
            smooth_reward = np.mean(raw_rewards[-window:])
            std_reward = np.std(raw_rewards[-window:])
        else:
            smooth_reward = reward
            std_reward = 0
            
        smoothed_rewards = self._reward_data['smoothed_rewards'] + [smooth_reward]
        std_rewards = self._reward_data['std_rewards'] + [std_reward]
        
        # ✅ Atomic update of reward data
        self._reward_data = {
            'episodes': episodes,
            'raw_rewards': raw_rewards,
            'smoothed_rewards': smoothed_rewards,
            'std_rewards': std_rewards
        }
    
    def get_plot_data(self):
        """Dashboard fetches precomputed plot data"""
        # ✅ Return a copy to avoid read/write conflicts
        return self._plot_data.copy()
    
    def get_reward_data(self):
        """Get reward curve data"""
        data = self._reward_data.copy()
        return (data['episodes'], data['raw_rewards'], 
                data['smoothed_rewards'], data['std_rewards'])

    def save_figure(self, save_path, episode, seed):
        """Save the current policy Plotly figure to a PNG file"""
        try:
            plot_data = self.get_plot_data()
            if not plot_data.get('valid', False):
                print(f"[PlotStore] No valid plot data for saving")
                return False
                
            x = plot_data['x']
            y_lin_list = plot_data['y_lin_list']
            y_RL_list = plot_data['y_RL_list']
            
            # Create Plotly figure
            rows, cols = _compute_grid(self.num_agent, self.ENV)
            fig = make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=self.title,
                horizontal_spacing=0.03,
                vertical_spacing=0.06,
            )
            
            # Add data to figure
            for i in range(self.num_agent):
                r, c = divmod(i, cols)
                r += 1
                c += 1
                show_legend = (i == 0)
                # Fixed colors
                LINEAR_COLOR = '#1f77b4'  # Blue
                RL_COLOR = '#ff7f0e'    # Orange

                # Linear line
                fig.add_trace(
                    go.Scattergl(mode='lines', name='Linear',
                                line=dict(dash='dashdot', width=1.5, color=LINEAR_COLOR),
                                x=x, y=y_lin_list[i], showlegend=show_legend),
                    row=r, col=c
                )
                # RL line
                fig.add_trace(
                    go.Scattergl(mode='lines', name='Flexible-RL',
                                line=dict(width=2, color=RL_COLOR),
                                x=x, y=y_RL_list[i], showlegend=show_legend),
                    row=r, col=c
                )
            
            # Hide extra subplots
            for k in range(self.num_agent, rows * cols):
                r, c = divmod(k, cols)
                fig.update_xaxes(visible=False, row=r + 1, col=c + 1)
                fig.update_yaxes(visible=False, row=r + 1, col=c + 1)
            
            # Set layout
            height = max(400, rows * 220)
            fig.update_layout(
                height=height,
                width=1200,
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
                template='plotly_white',
                title=f"Policy Episode {episode} (Seed {seed})"
            )
            
            # Ensure directory exists
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save as PNG file
            fig.write_image(save_path, width=1200, height=height)
            
            return True
        except Exception as e:
            print(f"[PlotStore] Failed to save figure: {e}")
            return False
        
    def save_reward_figure(self, save_path, seed=None):
        """保存当前reward的Plotly图形到PNG文件"""
        try:
            episodes, raw_rewards, smoothed_rewards, std_rewards = self.get_reward_data()
            
            if not episodes:
                print(f"[PlotStore] No reward data for saving")
                return False
            
            MAIN_COLOR = '#228B22'      # Forest Green - 森林绿
            FILL_COLOR = 'rgba(34,139,34,0.2)'
            
            # 创建reward图形
            fig = go.Figure()
            
            # 计算上下边界
            upper_bound = [s + std for s, std in zip(smoothed_rewards, std_rewards)]
            lower_bound = [s - std for s, std in zip(smoothed_rewards, std_rewards)]
            
            # 添加标准差填充区域（先添加，这样在图例中排在后面）
            fig.add_trace(go.Scatter(
                x=episodes + episodes[::-1],
                y=upper_bound + lower_bound[::-1],
                fill='toself',
                fillcolor=FILL_COLOR,
                line=dict(color='rgba(255,255,255,0)'),
                name='±1 Std Dev',
                showlegend=True
            ))
            
            # 添加平滑奖励曲线
            fig.add_trace(go.Scatter(
                x=episodes,
                y=smoothed_rewards,
                mode='lines',
                name='Average Reward',
                line=dict(color=MAIN_COLOR, width=3),
                showlegend=True
            ))
            
            # 可选：添加原始奖励点
            # fig.add_trace(go.Scatter(
            #     x=episodes[::max(1, len(episodes)//500)],  # 采样显示，避免过密
            #     y=raw_rewards[::max(1, len(raw_rewards)//500)],
            #     mode='markers',
            #     name='Raw Reward',
            #     marker=dict(color='lightgreen', size=2, opacity=0.5),
            #     showlegend=True
            # ))

                
            fig.update_layout(
                xaxis_title='Episode',
                yaxis_title='Reward',
                template='plotly_white',
                width=600,
                height=600,
                margin=dict(l=60, r=20, t=60, b=60),
                # legend=dict(
                #     orientation='h',
                #     yanchor='bottom',
                #     y=1.02,
                #     xanchor='left',
                #     x=0
                # ),
                showlegend=False,
                # 添加网格
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
            )
            
            # 确保目录存在
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存为PNG文件
            fig.write_image(os.path.join(save_path, f"avg_reward_{seed}.png"), width=600, height=600, scale=2)
            fig.write_image(os.path.join(save_path, f"avg_reward_{seed}.svg"), width=600, height=600, scale=2)
            fig.write_image(os.path.join(save_path, f"avg_reward_{seed}.pdf"), width=600, height=600)
            
            return True
        except Exception as e:
            print(f"[PlotStore] Failed to save reward figure: {e}")
            return False


def _compute_grid(num_agent: int, ENV: str):
    if ENV == '123bus':
        cols = 7
    else:
        cols = min(7, num_agent)
    rows = math.ceil(num_agent / cols)
    return rows, cols


def _make_initial_figure(store: PlotStore):
    rows, cols = _compute_grid(store.num_agent, store.ENV)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=store.title,
        horizontal_spacing=0.03,
        vertical_spacing=0.06,
    )

    # Two empty lines for each agent
    for i in range(store.num_agent):
        r, c = divmod(i, cols)
        r += 1
        c += 1
        show_legend = (i == 0)
        # Fixed colors
        LINEAR_COLOR = '#1f77b4'  # Blue
        RL_COLOR = '#ff7f0e'    # Orange

        fig.add_trace(
            go.Scattergl(mode='lines', name='Linear',
                        line=dict(dash='dashdot', width=1.5, color=LINEAR_COLOR),
                        x=[], y=[], showlegend=show_legend),
            row=r, col=c
        )
        fig.add_trace(
            go.Scattergl(mode='lines', name='Flexible-RL',
                        line=dict(width=2, color=RL_COLOR),
                        x=[], y=[], showlegend=show_legend),
            row=r, col=c
        )

    # Hide extra subplots
    for k in range(store.num_agent, rows * cols):
        r, c = divmod(k, cols)
        fig.update_xaxes(visible=False, row=r + 1, col=c + 1)
        fig.update_yaxes(visible=False, row=r + 1, col=c + 1)

    height = max(400, rows * 220)
    fig.update_layout(
        height=height,
        width=None,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        template='plotly_white'
    )
    return fig

def _make_initial_reward_figure():
    fig = go.Figure()
    MAIN_COLOR = '#228B22'      # Forest Green - 森林绿
    FILL_COLOR = 'rgba(34,139,34,0.2)'
    fig.add_trace(go.Scatter(
        x=[], y=[], 
        mode='lines', 
        name='Average Reward',
        line=dict(color=MAIN_COLOR, width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[], y=[], 
        mode='lines', 
        name='±1 Std Dev',
        fill='toself',
        fillcolor=FILL_COLOR,
        line=dict(color='rgba(255,255,255,0)')
    ))
    
    fig.update_layout(
        title='Training Reward',
        xaxis_title='Episode',
        yaxis_title='Reward',
        template='plotly_white',
        margin=dict(l=50, r=20, t=40, b=40)
    )
    return fig


def create_app(store: PlotStore) -> Dash:
    app = Dash(__name__)
    fig = _make_initial_figure(store)

    app.layout = html.Div(
        style={"padding": "8px"},
        children=[
            html.Div(
                style={"display": "flex", "gap": "12px", "alignItems": "center", "flexWrap": "wrap"},
                children=[
                    html.H3("RLC-FT Policy Monitor", style={"margin": "4px 0"}),
                    dcc.Slider(id="interval-ms", min=100, max=5000, step=50, value=1000,
                            marks=None, tooltip={"placement": "bottom", "always_visible": True},
                            updatemode="drag"),
                    html.Span("Polling interval (ms)"),
                ],
            ),

            html.Div([
                # Policy chart
                dcc.Graph(
                    id="policy-figure",
                    figure=fig,
                    style={"height": "60vh", "width": "100%"}
                ),
                # Reward chart
                dcc.Graph(
                    id="reward-figure",
                    figure=_make_initial_reward_figure(),
                    style={"height": "60vh", "width": "60%"}
                )
            ]),

            # Lightweight polling
            dcc.Interval(id="timer", interval=1000, n_intervals=0),
            
            # Store components
            dcc.Store(id="policy-version-store", data={"version": 0}),
            dcc.Store(id="reward-version-store", data={"version": 0}),
        ]
    )

    # Adjust polling frequency
    @app.callback(
        Output("timer", "interval"),
        Input("interval-ms", "value"),
        prevent_initial_call=False
    )
    def _set_interval(v):
        return int(v or 1000)

    @app.callback(
        Output("reward-figure", "figure"),
        Output("reward-version-store", "data"),
        Input("timer", "n_intervals"),
        State("reward-figure", "figure"),
        State("reward-version-store", "data"),
        prevent_initial_call=False
    )
    def _update_reward_figure(_n, fig_state, version_state):
        last_v = (version_state or {}).get("version", 0)
        cur_v = store.get_reward_version()
        if cur_v == last_v:
            return no_update, no_update
        
        episodes, raw_rewards, smoothed_rewards, std_rewards = store.get_reward_data()
        
        if not episodes:
            return no_update, no_update
        
        # Calculate upper and lower bounds
        upper_bound = [s + std for s, std in zip(smoothed_rewards, std_rewards)]
        lower_bound = [s - std for s, std in zip(smoothed_rewards, std_rewards)]
        
        # Update data
        fig_state["data"][0]["x"] = episodes
        fig_state["data"][0]["y"] = smoothed_rewards
        fig_state["data"][1]["x"] = episodes + episodes[::-1]
        fig_state["data"][1]["y"] = upper_bound + lower_bound[::-1]
        
        return fig_state, {"version": cur_v}

    @app.callback(
        Output("policy-figure", "figure"),
        Output("policy-version-store", "data"),
        Input("timer", "n_intervals"),
        State("policy-figure", "figure"),
        State("policy-version-store", "data"),
        prevent_initial_call=False
    )
    def _update_policy_figure(_n, fig_state, version_state):
        last_v = (version_state or {}).get("version", 0)
        cur_v = store.get_policy_version()
        if cur_v == last_v:
            return no_update, no_update

        try:
            plot_data = store.get_plot_data()
            
            if not plot_data.get('valid', False):
                return no_update, no_update
                
            x = plot_data.get('x', [])
            y_lin_list = plot_data.get('y_lin_list', [])
            y_RL_list = plot_data.get('y_RL_list', [])
            
            if not x or len(y_lin_list) != store.num_agent or len(y_RL_list) != store.num_agent:
                return no_update, no_update
        
        except Exception as e:
            print(f"[Dash] get_plot_data failed: {e}")
            return no_update, no_update

        data = fig_state.get("data", [])
        expected_traces = store.num_agent * 2
        if len(data) < expected_traces:
            return no_update, no_update

        for i in range(store.num_agent):
            lin_idx = 2 * i
            RL_idx = 2 * i + 1
            data[lin_idx]["x"] = x
            data[lin_idx]["y"] = y_lin_list[i]
            data[RL_idx]["x"] = x
            data[RL_idx]["y"] = y_RL_list[i]

        fig_state["data"] = data
        return fig_state, {"version": cur_v}

    return app


def start_dashboard(store: PlotStore, host: str = "127.0.0.1", port: int = 8050):
    """
    Start Dash service in a background thread. Returns (app, thread).
    """
    app = create_app(store)

    def _run():
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        if hasattr(app, "run"):
            try:
                app.run(host=host, port=port, debug=False)
                return
            except TypeError:
                pass
        if hasattr(app, "run_server"):
            app.run_server(host=host, port=port, debug=False, use_reloader=False)
        else:
            raise RuntimeError("Dash app has neither .run nor .run_server")

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    print(f"[Dash] dashboard running at http://{host}:{port}")
    return app, th