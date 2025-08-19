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

        # 存储预计算的图表数据（原子性更新）
        self._plot_data = {
            'x': [],
            'y_lin_list': [],
            'y_RL_list': [],
            'valid': False  # 标记数据是否有效
        }

        # reward数据存储
        self._reward_data = {
            'episodes': [],
            'raw_rewards': [],
            'smoothed_rewards': [],
            'std_rewards': []
        }

        self._policy_version = 0
        self._reward_version = 0

    def bump_policy(self, plot_data=None):
        """策略相关更新，训练端传入预计算的图表数据"""
        if plot_data is not None:
            # ✅ 原子性更新：先准备完整数据，再一次性替换
            new_plot_data = {
                'x': plot_data.get('x', []),
                'y_lin_list': plot_data.get('y_lin_list', []),
                'y_RL_list': plot_data.get('y_RL_list', []),
                'valid': True
            }
            self._plot_data = new_plot_data  # 原子替换
        
        self._policy_version += 1
        return self._policy_version
    
    def bump_reward(self):
        """reward相关更新（只触发reward图更新）"""
        self._reward_version += 1
        return self._reward_version
    
    def get_policy_version(self):
        return self._policy_version
            
    def get_reward_version(self):
        return self._reward_version
        
    def add_reward(self, episode, reward):
        """训练端调用，添加新的reward数据"""
        # 先更新本地数据
        episodes = self._reward_data['episodes'] + [episode]
        raw_rewards = self._reward_data['raw_rewards'] + [reward]
        
        # 计算平滑reward（滑动平均，窗口大小100）
        window = min(100, len(raw_rewards))
        if len(raw_rewards) >= window:
            smooth_reward = np.mean(raw_rewards[-window:])
            std_reward = np.std(raw_rewards[-window:])
        else:
            smooth_reward = reward
            std_reward = 0
            
        smoothed_rewards = self._reward_data['smoothed_rewards'] + [smooth_reward]
        std_rewards = self._reward_data['std_rewards'] + [std_reward]
        
        # ✅ 原子性更新reward数据
        self._reward_data = {
            'episodes': episodes,
            'raw_rewards': raw_rewards,
            'smoothed_rewards': smoothed_rewards,
            'std_rewards': std_rewards
        }
    
    def get_plot_data(self):
        """Dashboard获取预计算的图表数据"""
        # ✅ 返回数据副本，避免读写冲突
        return self._plot_data.copy()
    
    def get_reward_data(self):
        """获取reward曲线数据"""
        data = self._reward_data.copy()
        return (data['episodes'], data['raw_rewards'], 
                data['smoothed_rewards'], data['std_rewards'])

    def save_figure(self, save_path, episode, seed):
        """保存当前策略的Plotly图形到PNG文件"""
        try:
            plot_data = self.get_plot_data()
            if not plot_data.get('valid', False):
                print(f"[PlotStore] No valid plot data for saving")
                return False
                
            x = plot_data['x']
            y_lin_list = plot_data['y_lin_list']
            y_RL_list = plot_data['y_RL_list']
            
            # 创建Plotly图形
            rows, cols = _compute_grid(self.num_agent, self.ENV)
            fig = make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=self.title,
                horizontal_spacing=0.03,
                vertical_spacing=0.06,
            )
            
            # 添加数据到图形
            for i in range(self.num_agent):
                r, c = divmod(i, cols)
                r += 1
                c += 1
                show_legend = (i == 0)
                # 定义固定颜色
                LINEAR_COLOR = '#1f77b4'  # 蓝色
                RL_COLOR = '#ff7f0e'    # 橙色

                # Linear线
                fig.add_trace(
                    go.Scattergl(mode='lines', name='Linear',
                                line=dict(dash='dashdot', width=1.5, color=LINEAR_COLOR),
                                x=x, y=y_lin_list[i], showlegend=show_legend),
                    row=r, col=c
                )
                # RL线
                fig.add_trace(
                    go.Scattergl(mode='lines', name='Flexible-RL',
                                line=dict(width=2, color=RL_COLOR),
                                x=x, y=y_RL_list[i], showlegend=show_legend),
                    row=r, col=c
                )
            
            # 隐藏多余子图
            for k in range(self.num_agent, rows * cols):
                r, c = divmod(k, cols)
                fig.update_xaxes(visible=False, row=r + 1, col=c + 1)
                fig.update_yaxes(visible=False, row=r + 1, col=c + 1)
            
            # 设置布局
            height = max(400, rows * 220)
            fig.update_layout(
                height=height,
                width=1200,
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
                template='plotly_white',
                title=f"Policy Episode {episode} (Seed {seed})"
            )
            
            # 确保目录存在
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存为PNG文件
            fig.write_image(save_path, width=1200, height=height)
            
            return True
        except Exception as e:
            print(f"[PlotStore] Failed to save figure: {e}")
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

    # 每个 agent 两条空线
    for i in range(store.num_agent):
        r, c = divmod(i, cols)
        r += 1
        c += 1
        show_legend = (i == 0)
        # 定义固定颜色
        LINEAR_COLOR = '#1f77b4'  # 蓝色
        RL_COLOR = '#ff7f0e'    # 橙色

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

    # 隐藏多余子图
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
    fig.add_trace(go.Scatter(
        x=[], y=[], 
        mode='lines', 
        name='Smoothed Mean Reward',
        line=dict(color='green', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[], y=[], 
        mode='lines', 
        name='±1 Std Dev',
        fill='tonexty',
        fillcolor='rgba(0,128,0,0.2)',
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
                    html.Span("轮询间隔(ms)"),
                ],
            ),

            html.Div([
                # Policy图表
                dcc.Graph(
                    id="policy-figure",
                    figure=fig,
                    style={"height": "60vh", "width": "100%"}
                ),
                # Reward图表
                dcc.Graph(
                    id="reward-figure",
                    figure=_make_initial_reward_figure(),
                    style={"height": "35vh", "width": "100%"}
                )
            ]),

            # 轻量轮询
            dcc.Interval(id="timer", interval=1000, n_intervals=0),
            
            # Store 组件
            dcc.Store(id="policy-version-store", data={"version": 0}),
            dcc.Store(id="reward-version-store", data={"version": 0}),
        ]
    )

    # 调节轮询频率
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
        
        # 计算上下边界
        upper_bound = [s + std for s, std in zip(smoothed_rewards, std_rewards)]
        lower_bound = [s - std for s, std in zip(smoothed_rewards, std_rewards)]
        
        # 更新数据
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
    在后台线程启动 Dash 服务。返回 (app, thread)。
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