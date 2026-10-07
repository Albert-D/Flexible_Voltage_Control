"""Same-profile staged operational pilot; imports, but does not edit, old code."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandapower as pp
import torch

import run_real_world_pv_outage_stress as old


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".pending")
    with gzip.open(temp, "wb") as stream:
        pickle.dump(obj, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temp.replace(path)


def make_env(runtime):
    env, state, buses = old.build_environment(*runtime[:2])
    # An ideal terminal breaker makes the PV connection explicit without
    # inventing feeder impedance or changing the five measured grid buses.
    terminal = pp.create_bus(env.network, vn_kv=float(env.network.bus.at[52, "vn_kv"]),
                             name="PV53 terminal (no customer load)")
    breaker = pp.create_switch(env.network, bus=52, element=terminal, et="b", closed=True)
    env.network.sgen.at[5, "bus"] = terminal
    pp.runpp(env.network, algorithm="bfsw", init="dc")
    return env, state, buses, terminal, breaker


def run_method(method, profiles, disconnect, reconnect, runtime, data_root, event, max_steps=None):
    torch.set_num_threads(1)
    env, state, buses, terminal, breaker = make_env(runtime)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policies = old.load_policy_set(method, env, data_root, *runtime[2:5], runtime[5], device)
    topology = torch.as_tensor(np.asarray(env.topology_init), dtype=torch.float32, device=device)[None]
    n = len(profiles["p"]) if max_steps is None else max_steps
    states = np.empty((n + 1, 5))
    all_states = np.empty((n, len(env.network.bus)))
    actions = np.empty((n, 5))
    rewards = np.empty(n)
    availability = np.ones(n, dtype=bool)
    saturation = np.zeros(n, dtype=bool)
    states[0] = state
    last = np.zeros(5)
    events = []
    started = time.perf_counter()
    print(f"{method}: {n} steps, {device}", flush=True)
    with torch.inference_mode():
        for k in range(n):
            online = not (event and disconnect <= k < reconnect)
            if event and k in (disconnect, reconnect):
                env.network.switch.at[breaker, "closed"] = online
                env.network.sgen.at[5, "in_service"] = online
                env.network.sgen.at[5, "q_mvar"] = 0.0
                last[4] = 0.0
                events.append({"step": k, "connected": online, "breaker": int(breaker)})
            requested = old.controller_action(method, state, last, policies, topology, device)
            action = np.clip(requested, -25.0, 25.0)
            if not online:
                action[4] = 0.0
            saturation[k] = np.any(np.abs(requested) > 25.0)
            next_state, returned_topology, reward, _ = env.step_load(
                action.reshape(5, 1), profiles["p"][k], profiles["q"][k], profiles["pv_p"][k])
            topology = torch.as_tensor(np.asarray(returned_topology), dtype=torch.float32, device=device)[None]
            state = np.asarray(next_state, dtype=float)
            if not np.all(np.isfinite(state)):
                raise RuntimeError(f"Nonfinite monitored voltage: {method}, step {k}")
            states[k + 1] = state
            all_states[k] = env.network.res_bus.vm_pu.to_numpy()
            actions[k], rewards[k], availability[k] = action, reward, online
            last = action.copy()
            if (k + 1) % 2400 == 0:
                print(f"{method} {k+1}/{n}, {(time.perf_counter()-started)/60:.1f} min", flush=True)
    return dict(method=method, states=states, all_states=all_states, actions=actions,
                rewards=rewards, availability=availability, saturated=saturation,
                events=events, injection_bus=buses, terminal_bus=int(terminal),
                elapsed_sec=time.perf_counter()-started)


def metrics(result, start=0, stop=None):
    # Each post-control voltage is held for the following six-second interval.
    states = result["states"][1:][start:stop]
    outside = (states < .95) | (states > 1.05)
    indicator = np.any(outside, axis=1)
    delta = np.maximum(.95-states, 0) + np.maximum(states-1.05, 0)
    return dict(method=result["method"], peak_monitored_voltage_pu=float(states.max()),
                minimum_monitored_voltage_pu=float(states.min()),
                longest_sustained_violation_min=old.longest_true_run(indicator)*.1,
                longest_single_bus_violation_min=max(old.longest_true_run(c) for c in outside.T)*.1,
                unsafe_duration_min=float(indicator.sum())*.1,
                time_integrated_violation_pu_min=float(delta.sum())*.1,
                objective_cost=float(-result["rewards"][start:stop].sum()),
                max_abs_q_mvar=float(np.abs(result["actions"][start:stop]).max()),
                saturation_fraction=float(result["saturated"][start:stop].mean()),
                violation_at_window_end=bool(indicator[-1]),
                passes_voltage_criterion=bool(states.max() <= 1.10 and old.longest_true_run(indicator)*.1 < 5))


def plot(payload, folder, case):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family":"Arial", "font.size":9, "axes.labelsize":9,
                         "axes.titlesize":9, "xtick.labelsize":8, "ytick.labelsize":8,
                         "pdf.fonttype":42})
    results = {r["method"]:r for r in payload["results"]}
    p = payload["profiles"]
    n = len(p["p"])
    hours = np.arange(n)/600
    settings = payload["settings"]
    for window in ("full", "restoration_60min"):
        mm = {r["method"]:r for r in payload["metrics"][window]}
        fig, axes = plt.subplots(2,3,figsize=(8.4,5.9))
        fig.subplots_adjust(left=.085,right=.985,bottom=.13,top=.84,wspace=.48,hspace=.62)
        a,b,c,d,e,f = axes.flat
        for key,label,color in (("p","Load P",old.POWER_COLORS["Active load"]),
                                ("q","Load Q",old.POWER_COLORS["Reactive load"]),
                                ("pv_p","Available PV",old.POWER_COLORS["Solar"])):
            # Actual total assignments from Environment.step_load, before outage mask.
            mult=.88 if key in ("p","q") else 1.75
            a.plot(hours,p[key]*mult,color=color,lw=.8,label=label)
        a.set_ylabel("Power (MW / Mvar)")
        a.set_title("Operating profiles",pad=9)
        a.legend(loc="lower left",bbox_to_anchor=(-.08,1.20),ncol=3,frameon=False,fontsize=7.5,
                 handlelength=1.2,columnspacing=.7)
        high = max(float(results[m]["states"].max()) for m in ("No control","RLC-FT"))
        low = min(float(results[m]["states"].min()) for m in ("No control","RLC-FT"))
        for ax,method in ((b,"No control"),(c,"RLC-FT")):
            for j,col in enumerate(old.BUS_COLORS):
                ax.plot(hours,results[method]["states"][1:,j],color=col,lw=.7)
            for lim in (.95,1.05): ax.axhline(lim,color="#777777",ls="--",lw=.65)
            ax.set_ylim(min(.94,low-.006),max(1.08,high+.006))
            ax.set_title(method,pad=9)
        b.set_ylabel("Bus voltage (p.u.)")
        handles=[Line2D([0],[0],color=col,lw=1.3,label=f"Bus {bus}")
                 for bus,col in zip((18,21,30,45,53),old.BUS_COLORS)]
        fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.66,.975),ncol=5,
                   frameon=False,fontsize=7.5,handlelength=1.3,columnspacing=.8)
        for ax in (a,b,c):
            if settings["event"]:
                ax.axvline(settings["disconnect_step"]/600,color="#999999",ls="--",lw=.65)
                ax.axvline(settings["reconnect_step"]/600,color="#B2182B",ls=":",lw=.8)
            ax.set_xlim(0,n/600)
            ax.set_xticks(np.linspace(0,n/600,3 if n<=14400 else 5))
            ax.set_xlabel("Time (h)")
        specs=((d,"peak_monitored_voltage_pu","Peak voltage","Voltage (p.u.)",1.10),
               (e,"longest_sustained_violation_min","Longest voltage violation","Duration (min)",5.0),
               (f,"objective_cost","Total objective cost","Normalized cost",None))
        for ax,key,title,ylabel,limit in specs:
            vals=np.array([mm[m][key] for m in old.CONTROL_METHODS])
            if key=="objective_cost": vals=vals/vals[0]
            floor=(min(1.,vals.min()-.01) if key.startswith("peak") else
                   min(.9,vals.min()-.025) if key=="objective_cost" else 0.)
            top=max(vals.max(),limit or vals.max())
            span=max(top-floor,.03)
            ax.set_ylim(floor,top+.23*span)
            for j,(m,v) in enumerate(zip(old.CONTROL_METHODS,vals)):
                ax.vlines(j,floor,v,color=old.METHOD_COLORS[m],lw=1.2)
                ax.scatter(j,v,s=48,color=old.METHOD_FILLS[m],edgecolor=old.METHOD_COLORS[m],zorder=4)
                ax.annotate(f"{v:.3f}" if key!="longest_sustained_violation_min" else f"{v:.1f}",
                            (j,v),xytext=(0,6),textcoords="offset points",ha="center",fontsize=8)
            if limit:
                ax.axhline(limit,color="#B2182B",ls="--",lw=.8,zorder=2)
                ax.text(.98 if key.startswith("peak") else .02,limit,"1.10 p.u. limit" if key.startswith("peak") else "5-min limit",
                        transform=ax.get_yaxis_transform(),ha="right" if key.startswith("peak") else "left",va="bottom",
                        fontsize=7,color="#B2182B")
            else:
                ax.axhline(1,color="#999999",ls="--",lw=.7,zorder=0)
            ax.set_xticks(range(3),old.CONTROL_METHODS,rotation=15,ha="right")
            ax.set_xlim(-.38,2.38)
            ax.set_title(title,pad=9)
            ax.set_ylabel(ylabel)
        for label,ax in zip("abcdef",axes.flat):
            ax.text(-.22,1.08,label,transform=ax.transAxes,fontweight="bold",fontsize=12)
            ax.spines[["top","right"]].set_visible(False)
            ax.set_axisbelow(True)
            ax.grid(axis="y",color="#e4e4e4",lw=.5)
        text="Full-period metrics" if window=="full" else "Metrics: first 60 min after reconnection"
        if settings["event"]:
            text += " | Dashed event line: disconnection; dotted event line: reconnection"
        fig.text(.5,.015,text,ha="center",fontsize=8,color="#555555")
        folder.mkdir(parents=True,exist_ok=True)
        for ext in ("png","pdf"):
            fig.savefig(folder/f"{case}_{window}_2x3.{ext}",dpi=220,facecolor="white")
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.1,2.8),layout="constrained")
    k=settings["reconnect_step"]
    for m,ls in zip(old.CONTROL_METHODS,("--","-.","-")):
        s=results[m]["states"][1:]
        v=np.maximum(.95-s,0)+np.maximum(s-1.05,0)
        start=max(0,k-100); stop=min(n,k+600)
        ax.plot((np.arange(start,stop)-k)/10,v[start:stop].max(axis=1),
                label=m,color=old.METHOD_COLORS[m],ls=ls,lw=1.1)
    ax.set(xlabel="Time from reconnection (min)",ylabel="Maximum voltage violation (p.u.)")
    ax.legend(frameon=False); ax.spines[["top","right"]].set_visible(False)
    fig.savefig(folder/f"{case}_event_diagnostic.png",dpi=220)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--load-scale",type=float,default=1.)
    parser.add_argument("--days",type=int,choices=(1,2),default=1)
    parser.add_argument("--no-event",action="store_true")
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--plot-only",action="store_true")
    parser.add_argument("--workers",type=int,choices=(1,2),default=1)
    args=parser.parse_args()
    torch.set_num_threads(1)
    repo=Path(__file__).resolve().parent
    runtime=old.load_runtime(repo)
    data=Path(runtime[-1].data_path)
    root=data/"cache"/"operational_priority_2026-09-15"
    images=data/"images"/"operational_priority_2026-09-15"
    case=f"d{args.days}_load{args.load_scale:g}"+("_noevent" if args.no_event else "")
    if args.smoke: case+="_smoke"
    root.mkdir(parents=True,exist_ok=True)
    combined=root/f"{case}.pkl.gz"
    if args.plot_only:
        with gzip.open(combined,"rb") as f: payload=pickle.load(f)
        plot(payload,images,case)
        return
    source=data/"realworld_results.pkl.gz"
    profiles,source_steps=old.load_profiles(data,"one-day",None)
    # Use the published six-second sampling interval; trim the 21 extra samples
    # rather than compressing all 14421 samples into exactly 24 hours.
    profiles={k:np.tile(v[:14400],args.days) for k,v in profiles.items()}
    for key in ("p","q"): profiles[key]*=args.load_scale
    begin,end=6900,7800
    net=1.75*profiles["pv_p"][:14400]-.88*profiles["p"][:14400]
    reconnect=int(begin+np.argmax(net[begin:end+1]))+(args.days-1)*14400
    disconnect=5400
    if args.smoke: disconnect,reconnect=10,20
    settings=dict(case=case,source=str(source),source_sha256=digest(source),source_steps=source_steps,
                  day_steps=14400,dt_seconds=6,days=args.days,load_scale=args.load_scale,pv_scale=1.,
                  disconnect_step=disconnect,reconnect_step=reconnect,event=not args.no_event,
                  q_limit_mvar=25.,limit_status="common experimental cap; not a verified nameplate",
                  linear_gain=10.,safe_scale=10.,rlcft_scale=.7,
                  connection="ideal PV terminal breaker; original 55 feeder impedances unchanged",
                  implication="availability/topology event, not evidence of changing effective feeder X",
                  scripts={f:digest(repo/f) for f in ("Environment.py","NN_Module.py",old.__file__)})
    (root/f"{case}_settings.json").write_text(json.dumps(settings,indent=2),encoding="utf-8")
    results=[]
    missing=[]
    for method in old.METHODS:
        path=root/f"{case}_{method.replace(' ','_')}.pkl.gz"
        if path.exists():
            with gzip.open(path,"rb") as f: cached=pickle.load(f)
            if cached["settings"]!=settings: raise RuntimeError("Cached protocol differs")
            result=cached["result"]
        else:
            missing.append((method,path))
            continue
        results.append(result)
        print(json.dumps(metrics(result)),flush=True)
    if args.workers == 1:
        for method,path in missing:
            result=run_method(method,profiles,disconnect,reconnect,runtime,data,
                              not args.no_event,40 if args.smoke else None)
            dump(path,dict(settings=settings,result=result))
            results.append(result)
            print(json.dumps(metrics(result)),flush=True)
    elif missing:
        # Each process owns its environment and random seed; never share
        # mutable feeder state between controllers.
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs={pool.submit(run_method,method,profiles,disconnect,reconnect,runtime,data,
                              not args.no_event,40 if args.smoke else None):path
                  for method,path in missing}
            for future in as_completed(jobs):
                result=future.result()
                dump(jobs[future],dict(settings=settings,result=result))
                results.append(result)
                print(json.dumps(metrics(result)),flush=True)
    results.sort(key=lambda r: old.METHODS.index(r['method']))
    payload=dict(settings=settings,profiles=profiles,results=results,
                 metrics={"full":[metrics(r) for r in results],
                          "restoration_60min":[metrics(r,reconnect,min(reconnect+600,len(r['rewards']))) for r in results]})
    dump(combined,payload)
    (root/f"{case}_summary.json").write_text(json.dumps(payload["metrics"],indent=2),encoding="utf-8")
    if not args.smoke: plot(payload,images,case)
    print(f"Saved {combined}",flush=True)


if __name__=="__main__":
    main()
