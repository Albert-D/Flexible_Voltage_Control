"""123-bus continuing-task TD3: bootstrap collection time limits; true terminal masks only.

Pilot predecessor remains frozen. Model/evaluation/runtime utilities are reused.
"""
from runtime import *
OUT=OUT.parent/'123bus_training_efficiency_2026-09-16'
PROTOCOL=OUT/'protocol.json'
def train(args):
    if not args.run_id.replace('_','').isalnum():raise ValueError('ASCII run ID required')
    run=OUT/args.run_id;started=time.perf_counter()
    protocol=read(PROTOCOL);protocol_hash=sha(PROTOCOL);excluded=set(protocol['excluded_graphs'])
    source_paths=[Path(__file__),Path(__file__).with_name('runtime.py'),SPEED/'fast_environment.py',SPEED/'fast_runtime.py',FEEDER/'environment123.py',FEEDER/'model123.py',FEEDER/'evaluation123.py',FEEDER/'protocol123.py',
                  FEEDER/'learner.py',SPEED/'support.py',ROOT/'src/softplus_controller.py',ROOT/'data/case_123.mat']
    config=dict(args={k:v for k,v in vars(args).items() if k not in ('resume','stop_at')},protocol_sha256=protocol_hash,
                sources={str(p):sha(p) for p in source_paths},from_scratch=True,output_scale=1.,
                runtime=dict(powerflow='lightsim_recycle',adam_fused=True,actor_inference='synchronized_cpu',replay='cached_severity',threads=1),
                reward=dict(band=1.,deviation=.001,delta=args.delta_weight,q=0.),critic_lr=args.critic_lr,
                terminal_semantics='bootstrap time-limit/stage truncations; terminal only on recovery or hard failure',
                acceptance='engineering + learning trend; usable control separately relative to 123 reference',audit_performed=False)
    if args.resume:
        if read(run/'config.json')!=config:raise ValueError('Source/config changed; resume blocked')
    else:
        if run.exists():raise FileExistsError(run)
        run.mkdir(parents=True)
        save_json(run/'config.json',config)
        for src in source_paths:
            dest=run/'source'/src.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    device=torch.device('cuda');learner=Learner(device,seed=args.seed)
    for group in learner.critic_opt.param_groups:group['lr']=args.critic_lr;group['initial_lr']=args.critic_lr
    learner.critic_scheduler.base_lrs=[args.critic_lr];learner.critic_scheduler._last_lr=[args.critic_lr]
    if args.initial_factor!=1:initialize_weak_actor(learner.actor,args.initial_factor)
    learner.target.load_state_dict(learner.actor.state_dict())
    for group in learner.actor_opt.param_groups:group['lr']=args.actor_lr;group['initial_lr']=args.actor_lr
    learner.actor_scheduler.base_lrs=[args.actor_lr];learner.actor_scheduler._last_lr=[args.actor_lr]
    configure_adam(learner,True)
    replay=CachedReplay(seed=args.seed+3000);rng=np.random.default_rng(args.seed+1000);noise_rng=np.random.default_rng(args.seed+2000)
    env=Feeder();used=set();records=[];episodes=[];solver_checks=[]
    counts=dict(interactions=0,successful_steps=0,failed_steps=0,graph_draws=0,reset_attempts=0,reset_failures=0)
    timing={k:0. for k in ('reset','action','powerflow','replay_reward','sample','update','actor_sync','evaluation','checkpoint','solver_check','prior_elapsed')}
    if args.resume:
        latest=read(run/'latest_checkpoint.json');assert sha(latest['path'])==latest['sha256']
        ck=torch.load(latest['path'],map_location='cpu',weights_only=False)
        learner.load(ck['learner']);configure_adam(learner,True)
        replay.rows=ck['replay'];replay.pos=ck['replay_pos'];replay.rng.bit_generator.state=ck['replay_rng'];replay.rebuild_severity()
        rng.bit_generator.state=ck['rng'];noise_rng.bit_generator.state=ck['noise_rng'];torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
        counts=ck['counts'];timing=ck['timing'];timing['prior_elapsed']=ck['elapsed'];used=set(ck['used']);records=ck['records'];episodes=ck['episodes'];solver_checks=ck['solver_checks']
        (run/'training.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in episodes),encoding='utf-8')
        save_json(run/'restore_check.json',dict(interactions=counts['interactions'],updates=learner.updates,replay_rows=len(replay.rows),cache_rebuilt=True,
                  actor_optimizer_devices=sorted({str(v.device) for s in learner.actor_opt.state.values() for v in s.values() if torch.is_tensor(v)})))
    mirror=ActorMirror(learner.actor)
    def elapsed():return timing['prior_elapsed']+time.perf_counter()-started
    def checkpoint():
        t=time.perf_counter();path=run/f'checkpoint_{counts["interactions"]:06d}.pt';temp=path.with_suffix('.tmp')
        torch.save(dict(learner=learner.state(),replay=replay.rows,replay_pos=replay.pos,replay_rng=replay.rng.bit_generator.state,
                        rng=rng.bit_generator.state,noise_rng=noise_rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                        counts=counts,timing=timing.copy(),used=sorted(used),records=records,episodes=episodes,solver_checks=solver_checks,elapsed=elapsed(),protocol_sha256=protocol_hash),temp)
        temp.replace(path);save_json(run/'latest_checkpoint.json',dict(path=str(path),sha256=sha(path)));timing['checkpoint']+=time.perf_counter()-t
    anchor=protocol['groups']['anchor']
    reference=read(OUT/'reference_anchor.json');assert reference['protocol_sha256']==protocol_hash
    reference_anchor=reference['rows'];anchor_count=len(anchor)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
        def assess():
            t=time.perf_counter();index=len(records);rotation=protocol['groups']['rotation'][index*16:(index+1)*16];assert len(rotation)==16
            scene_batch=anchor+rotation
            rows=evaluate(pool,mirror.actor,scene_batch);refs=evaluate(pool,None,rotation,reference=True)
            a=metrics(rows[:anchor_count]);r=metrics(rows[anchor_count:]);refa=metrics(reference_anchor);refr=metrics(refs)
            passed=bool(a['failures']==0 and a['success_rate']>=.95 and .8*refa['recovery_median']<=a['recovery_median']<=1.2*refa['recovery_median'] and a['recovery_p90']<=max(30,refa['recovery_p90']*1.25))
            timing['evaluation']+=time.perf_counter()-t
            row=dict(interactions=counts['interactions'],num_topologies_seen=len(used),updates=learner.updates,anchor=a,rotation=r,reference_anchor=refa,reference_rotation=refr,
                     topology=gains(mirror.actor,anchor+protocol['groups']['rotation'][:16]),usable_development_pass=passed,
                     rows=rows,rotation_reference_rows=refs,timing=timing.copy(),elapsed=elapsed())
            records.append(row);save_json(run/'curve_data.json',records);checkpoint()
            print(json.dumps(dict(run=args.run_id,interactions=counts['interactions'],topologies=len(used),anchor=a,rotation=r,topology=row['topology'],usable_pass=passed,elapsed=elapsed())),flush=True)
        if not records:assess()
        reason='stage_budget'
        stop=min(args.max_interactions,args.stop_at)
        while counts['interactions']<stop:
            if (run/'STOP').exists():reason='requested_stage_stop';break
            if elapsed()>args.wall_minutes*60:reason='wall_budget';break
            if len(records)>=3 and all(r['anchor']['failures']>=4 for r in records[-2:]):reason='evaluation_failure_stop';break
            t=time.perf_counter();topology=draw_graph(env,rng,used|excluded,counts);scene,state=scene_for(env,topology,rng,90000000+len(used),counts)
            timing['reset']+=time.perf_counter()-t;used.add(topology['id']);assert not used&excluded
            x=env.features.copy();previous=np.zeros(14,dtype=np.float32);reward_total=0.;losses={}
            budget=min(60,stop-counts['interactions']);actual_steps=0
            for step in range(budget):
                t=time.perf_counter()
                with torch.no_grad():delta=mirror.actor(torch.as_tensor(state).view(1,14),torch.as_tensor(x).view(1,113)).numpy().reshape(14)
                noise=noise_rng.normal(0,args.warm_noise if counts['interactions']<512 else args.noise,14)
                q=np.clip(previous-np.clip(delta+noise,-5,5),-50,50).astype(np.float32);timing['action']+=time.perf_counter()-t
                if not np.isfinite(q).all():raise ValueError('Nonfinite action')
                counts['interactions']+=1;actual_steps+=1;terminal=False;t=time.perf_counter()
                try:
                    nxt=env.step(q)
                    if not np.isfinite(nxt).all() or nxt.min()<.75 or nxt.max()>1.25:raise ValueError('Hard voltage bound')
                    local=np.maximum(np.abs(nxt-1)-.045,0)+.001*np.abs(nxt-1)+args.delta_weight*(q-previous)**2
                    reward=-(.5*local+.5*local.mean());counts['successful_steps']+=1
                    terminal=terminal or bool(np.all((nxt>.955)&(nxt<1.045)))
                except (ValueError,pp.powerflow.LoadflowNotConverged):
                    counts['failed_steps']+=1;nxt=state.copy();reward=np.full(14,-100.,dtype=np.float32);terminal=True
                timing['powerflow']+=time.perf_counter()-t
                t=time.perf_counter();replay.push(state,x,previous,q,reward,nxt,x,[float(terminal)]);timing['replay_reward']+=time.perf_counter()-t
                if counts['interactions']>=512 and len(replay.rows)>=128:
                    t=time.perf_counter();batch=sample_replay_fast(replay,128,.5,.75);timing['sample']+=time.perf_counter()-t
                    t=time.perf_counter();losses=learner.update(batch);torch.cuda.synchronize();timing['update']+=time.perf_counter()-t
                    if any(v is not None and not np.isfinite(v) for v in losses.values()):raise ValueError('Nonfinite loss')
                    if learner.updates%3==0:
                        if any(not bool(torch.isfinite(v).all()) for v in learner.actor.parameters()):raise ValueError('Nonfinite actor parameter')
                        t=time.perf_counter();mirror.sync(learner.actor);timing['actor_sync']+=time.perf_counter()-t
                reward_total+=float(reward.sum());state,previous=nxt,q
                if terminal:break
            if (len(used) in (1,10,30,50) or len(used)%200==0) and counts['failed_steps']==0:
                t=time.perf_counter();baseline=OriginalFeeder();baseline.reset(scene);baseline.step(previous)
                error=float(np.nanmax(np.abs(baseline.voltage-env.voltage)))
                solver_checks.append(dict(interactions=counts['interactions'],topology=topology['id'],max_voltage_abs=error,passed=error<=1e-6));assert error<=1e-6
                timing['solver_check']+=time.perf_counter()-t
            episode=dict(interactions=counts['interactions'],num_topologies_seen=len(used),episode_steps=actual_steps,terminal=bool(terminal),truncated=bool(actual_steps==budget and not terminal),reward_per_step=reward_total/actual_steps,losses=losses,scene=scene,counts=counts.copy())
            episodes.append(episode);append(run/'training.jsonl',episode)
            save_json(run/'progress.json',dict(interactions=counts['interactions'],topologies=len(used),updates=learner.updates,counts=counts,timing=timing,elapsed=elapsed()))
            if counts['failed_steps']>=5 and counts['failed_steps']/counts['interactions']>.01:reason='training_failure_stop';break
            if counts['interactions']-records[-1]['interactions']>=args.eval_every:assess()
        if records[-1]['interactions']!=counts['interactions']:assess()
    save_json(run/'stage_result.json',dict(reason=reason,counts=counts,topologies=len(used),timing=timing,elapsed=elapsed(),solver_checks=solver_checks,
              train_evaluation_graph_overlap=len(used&excluded),latest=read(run/'latest_checkpoint.json'),final=records[-1],audit_performed=False,
              gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,gpu_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
    print(json.dumps(dict(stage_complete=args.run_id,interactions=counts['interactions'],reason=reason)),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True);parser.add_argument('--actor-lr',type=float,default=1e-4)
    parser.add_argument('--critic-lr',type=float,default=3e-4)
    parser.add_argument('--delta-weight',type=float,default=.05);parser.add_argument('--initial-factor',type=float,default=1.)
    parser.add_argument('--noise',type=float,default=.08);parser.add_argument('--warm-noise',type=float,default=.25)
    parser.add_argument('--seed',type=int,default=2601);parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--max-interactions',type=int,default=30000);parser.add_argument('--stop-at',type=int,default=6000)
    parser.add_argument('--eval-every',type=int,default=1000);parser.add_argument('--wall-minutes',type=float,default=180.)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args();train(args)
if __name__=='__main__':main()
