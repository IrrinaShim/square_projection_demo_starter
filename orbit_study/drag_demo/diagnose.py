"""Сравнить сохранённые орбиты: .venv/bin/python -m orbit_study.drag_demo.diagnose."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from orbit_study.drag_model import integrate
from orbit_study.model import reconstruct
from square_demo.grid import expected_neighbor_angle_deg

HERE=Path(__file__).resolve().parent/'results'

def diagnose(path):
    case=json.loads(Path(path).read_text(encoding='utf-8'))
    cfg=case['config']; elements=case['elements_at_epoch']
    mu=cfg['orbit']['mu_km3_s2']; radius=cfg['orbit']['earth_radius_km']
    a=np.array([x['a_km'] for x in elements]); e=np.array([x['e'] for x in elements]);inc=np.array([x['inclination_deg'] for x in elements])
    states=reconstruct(elements,mu)
    t=np.arange(cfg['display']['start_s'],cfg['display']['end_s']+2.5,5.)
    propagated=integrate(states,t,cfg,cfg['display']['integration_step_s'])
    center=propagated[:,:,0].mean(axis=1,keepdims=True)
    radii=np.linalg.norm(propagated[:,:,0]-center,axis=2)
    relative_speed=np.linalg.norm(states[:,1]-states[:,1].mean(axis=0),axis=1)
    return {'case':str(path),'N':case['N'],'side_deg':case['parameters']['side_deg'],
        'a_km':[float(a.min()),float(a.max())],'e':[float(e.min()),float(e.max())],
        'i_deg':[float(inc.min()),float(inc.max())],
        'perigee_altitude_km':[float(np.min(a*(1-e)-radius)),float(np.max(a*(1-e)-radius))],
        'apogee_altitude_km':[float(np.min(a*(1+e)-radius)),float(np.max(a*(1+e)-radius))],
        'period_spread_s':float(np.ptp(2*np.pi*np.sqrt(a**3/mu))),
        'epoch_centroid_relative_speed_km_s':[float(relative_speed.min()),float(relative_speed.max())],
        'centroid_radius_over_display_km':[float(radii.min()),float(radii.max())],
        'expected_min_true_neighbor_deg':expected_neighbor_angle_deg(case['N'],case['parameters']['side_deg']),
        'one_pass_duration_2deg_s':case['one_pass']['longest_contiguous_duration_2deg_s'],
        'one_pass_minimum_pair_km':case['one_pass']['closest_pair']['distance_km'],
        'full_orbit_minimum_pair_km':case.get('full_orbit',{}).get('closest_pair',{}).get('distance_km'),
        'verified_acceptable':case.get('verified_acceptable',False)}

if __name__=='__main__':
    for path in (HERE/'case_unconstrained_legacy.json',HERE/'n2/case.json',
                 HERE/'n3/case.json',HERE/'case.json'):
        print(json.dumps(diagnose(path),ensure_ascii=False))
