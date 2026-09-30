"""Reproduce historical code, phase ablations and equal-criteria before/after.

No checkout/reset: read the two immutable Git revisions into isolated modules.
Run: .venv/bin/python -m orbit_study.physics_display_audit
"""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np

from orbit_study.drag_model import acceleration, integrate
from orbit_study.first_iteration_comparison import longest
from orbit_study.model import reconstruct
from orbit_study.orbit_direction import audit_orbit_normals
from orbit_study.passive_optimizer import (configuration,
    neighboring_seed, neighboring_candidate, load_cached, verify, _candidate,
    inverse_refine)
from square_demo.core import (hcw_free, zenith_orbit, observer_eci, project,
                              square_geometry)
from square_demo.orbit_j2 import orbital_basis
from square_demo.drag_provider import calculate_drag_case

ROOT=Path(__file__).resolve().parent.parent
OLD='9a1675c93ffadd91a52dc00544d96b44cda3027f'
BEFORE='ba62a774dc2506b080e655be9d898ae34b528dd9'
OUT=ROOT/'orbit_study'/'audit_results'
CASES=((2,1.818302964,630.),(2,20.,700.),(10,20.,700.))


def git_read(revision,path):
    return subprocess.check_output(['git','show',f'{revision}:{path}'],cwd=ROOT,text=True)


def historical_provider():
    package=types.ModuleType('_historical_square');package.__path__=[]
    sys.modules[package.__name__]=package
    for name in ('core','providers'):
        fullname=f'{package.__name__}.{name}'
        module=types.ModuleType(fullname)
        module.__file__=str(ROOT/'square_demo'/f'{name}.py')
        sys.modules[fullname]=module
        exec(compile(git_read(OLD,f'square_demo/{name}.py'),f'{OLD}:{name}.py','exec'),module.__dict__)
    return module


def duration(t,mask):
    return (longest(t,mask) or {}).get('duration_s',0.)


def pair_normal_angle(states):
    h=np.cross(states[:,:,0],states[:,:,1]);unit=h/np.linalg.norm(h,axis=2)[...,None]
    maximum=0.
    for j in range(states.shape[1]-1):
        dot=np.sum(unit[:,j:j+1]*unit[:,j+1:],axis=2)
        cross=np.linalg.norm(np.cross(unit[:,j:j+1],unit[:,j+1:]),axis=2)
        maximum=max(maximum,float(np.rad2deg(np.arctan2(cross,dot)).max()))
    return maximum


def criteria(r,robs,basis,t,cfg,n):
    """Original outer-shape/visibility, then size, then full grid, then 2°."""
    from square_demo.grid import grid_metrics, grid_acceptance, corner_indices
    image,valid,elev=project(r,robs,basis,cfg['orbit']['earth_radius_km'])
    corners=corner_indices(n)
    sides,_,_,errors=square_geometry(image[:,corners],valid[:,corners])
    vis=cfg['visibility']
    old=(valid.all(axis=1)&(elev.min(axis=1)>=vis['min_elevation_deg'])
         &(errors[:,0]<=vis['max_side_error'])&(errors[:,1]<=vis['max_diagonal_error'])
         &(errors[:,2]<=np.deg2rad(vis['max_angle_error_deg'])))
    size_ok=abs(np.nanmean(sides,axis=1)/np.tan(np.deg2rad(cfg['angular_side_deg']))-1)<=cfg['grid_tolerances']['max_outer_side_relative_error']
    gm=grid_metrics(image,valid,r,robs,n)
    grid_ok,_=grid_acceptance(gm,n,dict(cfg['grid_tolerances'],min_neighbor_angle_deg=0.))
    square=old&size_ok&grid_ok
    resolved=square&(gm['min_true_neighbor_angle_deg']>=2.)
    return dict(shape_visibility_s=duration(t,old),with_outer_size_s=duration(t,old&size_ok),
        square_grid_s=duration(t,square),with_resolution_2deg_s=duration(t,resolved),
        square_interval=longest(t,square),time_step_s=float(np.max(np.diff(t))))


def independent_row(name,candidate,cfg):
    n=candidate['N'];t=np.arange(-450.,451.)
    states=integrate(candidate['states'],t,cfg,1.)
    obs=observer_eci(t,0.,cfg['orbit']['earth_radius_km'],cfg['observer'])
    row=dict(stage=name,N=n,side_deg=cfg['angular_side_deg'],altitude_km=cfg['orbit']['altitude_km'],
        **criteria(states[:,:,0],obs,orbital_basis(states[:,0]),t,cfg,n))
    checked=verify(candidate,cfg)
    row.update(max_normal_to_chief_deg=checked['orbital_direction']['full_orbit']['max_angle_deg'],
        min_pair_orbit_km=checked['full_orbit']['closest_pair']['distance_km'],
        physical_ok=not checked['physical_violations'],physical_violations=checked['physical_violations'],
        altitude_orbit_km=checked['full_orbit']['altitude_range_km'],
        elements_roundtrip_max_error=checked['elements_roundtrip_max_error'],
        step_1_vs_half_position_km=checked['step_1_vs_half_max_position_km'],
        orbit_limits=checked['orbit_limits'])
    period=checked['full_orbit']['period_s']
    full_t=np.r_[np.arange(0.,period,5.),period]
    full=integrate(candidate['states'],full_t,cfg,2.)
    row['max_pair_normal_on_orbit_deg']=pair_normal_angle(full)
    # Same force call applied to each Earth-centered state: batching is inert.
    batch=acceleration(candidate['states'],cfg)
    single=np.stack([acceleration(s,cfg) for s in candidate['states']])
    row['batch_single_acceleration_error_km_s2']=float(np.max(abs(batch-single)))
    print(name,n,cfg['angular_side_deg'],row['shape_visibility_s'],row['square_grid_s'],row['physical_ok'],flush=True)
    return row


def historical_states(result,t,cfg):
    r0,b,_,motion=zenith_orbit(t,0.,cfg['orbit'],cfg['observer'])
    rho,vrel=hcw_free(t,result.metadata['initial_rho_oso_km'],
        result.metadata['initial_vel_oso_m_s']/1000.,motion)
    states=np.empty((len(t),4,2,3))
    states[:,0,0]=r0;states[:,0,1]=b[:,:,1]*np.sqrt(cfg['orbit']['mu_km3_s2']/np.linalg.norm(r0,axis=1))[:,None]
    dr=np.einsum('tij,tsj->tsi',b,rho)
    omega=motion*b[:,:,2]
    states[:,1:,0]=r0[:,None]+dr
    states[:,1:,1]=states[:,0:1,1]+np.cross(omega[:,None],dr)+np.einsum('tij,tsj->tsi',b,vrel)
    return states,b


def historical_rows(provider,cfg):
    scenario=json.loads(git_read(OLD,'scenarios/free_zenith.json'))
    scenario.update(angular_side_deg=cfg['angular_side_deg'],orbit=cfg['orbit'],observer=cfg['observer'])
    rows=[]
    for variant in ('trial','fitted'):
        result=provider.free_hcw_two_body(dict(scenario,variant=variant))
        t=np.arange(-450.,451.)
        states,b=historical_states(result,t,cfg)
        obs=observer_eci(t,0.,cfg['orbit']['earth_radius_km'],cfg['observer'])
        row=dict(stage='9a1675c HCW '+variant,N=2,side_deg=cfg['angular_side_deg'],altitude_km=cfg['orbit']['altitude_km'],
            **criteria(states[:,:,0],obs,b,t,cfg,2))
        row['historical_5s_shape_visibility_s']=duration(result.t_s,result.metadata['observability']['eligible'])
        row['historical_5s_with_size_s']=criteria(result.r_eci_km,result.observer_eci_km,
            zenith_orbit(result.t_s,0.,cfg['orbit'],cfg['observer'])[1],result.t_s,cfg,2)['with_outer_size_s']
        period=2*np.pi/np.sqrt(cfg['orbit']['mu_km3_s2']/(cfg['orbit']['earth_radius_km']+cfg['orbit']['altitude_km'])**3)
        full_t=np.r_[np.arange(0,period,.5),period]
        full,_=historical_states(result,full_t,cfg)
        minpair=min(float(np.linalg.norm(full[:,i,0]-full[:,j,0],axis=1).min()) for i in range(4) for j in range(i+1,4))
        # The HCW curve is kinematic in ECI, not a solution of adopted forces.
        dt=.05
        plus,_=historical_states(result,t+dt,cfg);minus,_=historical_states(result,t-dt,cfg)
        residual=(plus[:,:,1]-minus[:,:,1])/(2*dt)-acceleration(states,cfg)
        row.update(physical_ok=False,physical_violations=['HCW: не независимое решение принятой модели J2+сопротивление'],
            max_normal_to_chief_deg=audit_orbit_normals(full,full_t)['max_angle_deg'],
            max_pair_normal_on_orbit_deg=pair_normal_angle(full),
            min_pair_orbit_km=minpair,pair_minimum_method='HCW analytic samples 0.5 s, not a physical safety audit',
            max_dynamics_residual_m_s2=float(np.linalg.norm(residual,axis=2).max()*1000))
        rows.append(row)
        initial=states[450]
        candidate=_candidate(initial,result.target_image_gnomonic,2,cfg,'исторический '+variant)
        rows.append(independent_row('HCW '+variant+' → independent',candidate,cfg))
        if variant=='fitted':
            constrained=neighboring_candidate(candidate,neighboring_seed(cfg,2),cfg)
            rows.append(independent_row('HCW fitted → constrained',constrained,cfg))
    return rows


def circular_control():
    cfg=configuration(2,20.);cfg['orbit']['j2']=0.
    s=neighboring_seed(cfg,2)['states'][:1];mu=cfg['orbit']['mu_km3_s2'];r=np.linalg.norm(s[0,0])
    period=2*np.pi*np.sqrt(r**3/mu);t=np.linspace(0,period,1001)
    x=integrate(s,t,cfg,2.,with_drag=False)[:,0]
    radius=np.linalg.norm(x[:,0],axis=1);energy=np.sum(x[:,1]**2,axis=1)/2-mu/radius
    normal=np.cross(x[:,0],x[:,1]);normal/=np.linalg.norm(normal,axis=1)[:,None]
    return dict(radius_max_error_km=float(max(abs(radius-r))),
        energy_max_relative_drift=float(max(abs(energy/energy[0]-1))),
        normal_max_vector_drift=float(np.linalg.norm(normal-normal[0],axis=1).max()),
        closure_position_km=float(np.linalg.norm(x[-1,0]-x[0,0])),period_s=float(period))


def before_candidates():
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',BEFORE,'orbit_study/optimizer_results'],cwd=ROOT,text=True).splitlines()
    for path in paths:
        if path.endswith('/best.json'):
            data=json.loads(git_read(BEFORE,path))
            if data.get('schema_version')==5:yield path,data


def display_checks(candidate,cfg):
    result=calculate_drag_case(cfg,candidate)
    basis=result.metadata['local_basis_eci'];center=result.metadata['center_eci_km']
    reconstructed=center[:,None]+np.einsum('tij,tsj->tsi',basis,result.rho_oso_km)
    image,valid,_=project(result.r_eci_km,result.observer_eci_km,basis,cfg['orbit']['earth_radius_km'])
    local_velocity=result.metadata['local_velocity_oso_km_s']
    t=result.t_s;dt=.05
    shifted=[]
    from square_demo.orbit_j2 import centroid_relative_frame
    from orbit_study.drag_model import drag_acceleration_km_s2
    for offset in (-dt,dt):
        states=integrate(candidate['states'],t+offset,cfg,1.)
        extra=drag_acceleration_km_s2(states[:,0,0],states[:,0,1],cfg['orbit']['earth_radius_km'],cfg['observer']['earth_rotation_rad_s'],cfg['spacecraft'],cfg['atmosphere'])
        shifted.append(centroid_relative_frame(states,cfg['orbit']['mu_km3_s2'],cfg['orbit']['earth_radius_km'],cfg['orbit']['j2'],extra)[3])
    samples=[]
    for k in (60,90,120):
        samples.append(dict(t_s=float(t[k]),spacecraft_1_eci_km=result.r_eci_km[k,1].tolist(),
            spacecraft_1_local_km=result.rho_oso_km[k,1].tolist(),
            spacecraft_1_yoz_km=result.rho_oso_km[k,1,1:].tolist(),
            spacecraft_1_image_deg=np.rad2deg(np.arctan(image[k,1])).tolist()))
    return dict(eci_reconstruction_error_km=float(np.max(abs(reconstructed-result.r_eci_km))),
        projection_recalculation_error=float(np.nanmax(abs(image-result.image_gnomonic))),
        local_velocity_derivative_error_km_s=float(np.max(abs((shifted[1]-shifted[0])/(2*dt)-local_velocity))),
        samples=samples)


def main():
    OUT.mkdir(exist_ok=True)
    provider=historical_provider();rows=[];checks=[]
    old_optimizer=types.ModuleType('_baseline_optimizer')
    old_optimizer.__file__=str(ROOT/'orbit_study'/'passive_optimizer.py')
    exec(compile(git_read(BEFORE,'orbit_study/passive_optimizer.py'),
                 BEFORE+':passive_optimizer.py','exec'),old_optimizer.__dict__)
    before=list(before_candidates())
    for n,side,alt in CASES:
        cfg=configuration(n,side,altitude_km=alt)
        if n==2:rows+=historical_rows(provider,cfg)
        path,old=next((p,d) for p,d in before if d['N']==n and d['config']==cfg)
        old_candidate=_candidate(reconstruct(old['candidate']['elements_at_epoch'],cfg['orbit']['mu_km3_s2']),
            np.asarray(old['candidate']['target_gnomonic']),n,cfg,'baseline')
        rows.append(independent_row('before ba62a774',old_candidate,cfg))
        base=neighboring_seed(cfg,n)
        rows.append(independent_row('same seed → inverse v5, 150 s',
            old_optimizer.inverse_refine(old_candidate,cfg,base),cfg))
        rows.append(independent_row('same seed → inverse v6, 150 s',
            inverse_refine(old_candidate,cfg,base),cfg))
        current=load_cached(cfg,n)
        if current is None:raise ValueError(f'Run optimizer first: N={n},side={side},alt={alt}')
        _,candidate,_,path=current
        rows.append(independent_row('after v6',candidate,cfg))
        checks.append(dict(N=n,side_deg=side,altitude_km=alt,**display_checks(candidate,cfg)))
    output=dict(historical_commit=OLD,baseline_commit=BEFORE,circular_control=circular_control(),
                cases=rows,display_chain_checks=checks,
                normal_convention='max directed angle to spacecraft 0, full orbit; not inclination difference',
                interval_convention='longest contiguous accepted sampled interval, 1 s; historical 5 s also listed')
    (OUT/'physics_display.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# Проверка физики, рисунка и поиска','',
        'Повторяется командой `.venv/bin/python -m orbit_study.physics_display_audit`. Исторические модули читаются непосредственно из `9a1675c`, состояния «до» — из `ba62a774`, без checkout и изменения рабочей копии.',
        '', 'Окно −450…450 с, узлы 1 с; одинаковые критерии формы 10%/10%/10°, угол места ≥15°, внешний размер ±10%. Для N>2 отдельно проверяется вся сетка. 2° — дополнительный сценарий прибора. Угол в таблице — максимум между всеми парами направленных нормалей на витке; ограничение модели 5° проверяется относительно КА 0 и отдельно записано в JSON. Все расстояния всех пар проверены на витке; для HCW приведена только аналитическая выборка 0,5 с. Строки inverse показывают полный пробный шаг из одного исходного seed, а не принятый итог поиска.', '',
        '| Этап | КА | Сторона, ° | Высота, км | Форма+видимость, с | +размер, с | +сетка, с | +2°, с | Нормали, ° | min пары, км | Физически допустим |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for row in rows:
        lines.append(f"| {row['stage']} | {row['N']**2} | {row['side_deg']:.9f} | {row['altitude_km']:g} | {row['shape_visibility_s']:g} | {row['with_outer_size_s']:g} | {row['square_grid_s']:g} | {row['with_resolution_2deg_s']:g} | {row['max_pair_normal_on_orbit_deg']:.6f} | {row['min_pair_orbit_km']:.6f} | {'да, в рамках модели' if row['physical_ok'] else 'нет'} |")
    lines+=['','Нарушения и численные проверки (включая высоты, элементы, уменьшение шага и координаты трёх графиков в одинаковые моменты) записаны в [JSON](physics_display.json).','']
    for row in rows:
        if row['physical_violations']:
            lines.append(f"- {row['stage']}, {row['side_deg']:.9f}°, {row['altitude_km']:g} км: "+'; '.join(row['physical_violations']))
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    rejected_leader()
    print(OUT/'README.md')


def rejected_leader():
    """Keep the stronger-looking but physically rejected N=10 candidate."""
    cfg=configuration(10,20.)
    _,_,data,path=load_cached(cfg,10)
    history=[json.loads(line) for line in path.with_name('candidates.jsonl').read_text().splitlines()]
    leader=max(history,key=lambda row:row['score'])
    candidate=_candidate(np.asarray(leader['initial_states_eci_km_km_s']),
        np.asarray(data['candidate']['target_gnomonic']),10,cfg,leader['source'])
    checked=verify(candidate,cfg)
    evidence=dict(N=10,side_deg=20.,altitude_km=700.,iteration=leader['iteration'],
        initial_states_eci_km_km_s=leader['initial_states_eci_km_km_s'],
        elements_at_epoch=leader['elements_at_epoch'],
        square_duration_s=checked['one_pass']['best_square_duration_s'],
        closest_pair=checked['full_orbit']['closest_pair'],
        physical_violations=checked['physical_violations'])
    (OUT/'rejected_leader.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
