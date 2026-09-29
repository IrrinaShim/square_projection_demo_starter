"""Reproduce the first four-spacecraft HCW comparison and current audits.

Run with ``.venv/bin/python -m orbit_study.first_iteration_comparison``.
The old 425 s criterion omitted a bound on the *target external size*.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from orbit_study.passive_optimizer import configuration, load_cached
from square_demo.core import square_geometry
from square_demo.providers import SCENARIOS, free_hcw_two_body, load_scenario


def longest(t,mask):
    delta=np.diff(np.r_[False,mask,False].astype(int))
    starts=np.flatnonzero(delta==1)
    ends=np.flatnonzero(delta==-1)-1
    if not len(starts):return None
    a,b=max(zip(starts,ends),key=lambda indices:t[indices[1]]-t[indices[0]])
    return {"start_s":float(t[a]),"end_s":float(t[b]),
            "duration_s":float(t[b]-t[a])}


def historical():
    scenario=load_scenario(SCENARIOS["free_hcw_two_body"])
    out={}
    for name in ("trial","fitted"):
        result=free_hcw_two_body(dict(scenario,variant=name))
        sides,_,_,_=square_geometry(result.image_gnomonic,result.image_valid)
        angular=np.rad2deg(np.arctan(np.nanmean(sides,axis=1)))
        relative_size=np.tan(np.deg2rad(angular))/np.tan(
            np.deg2rad(scenario["angular_side_deg"]))-1
        historical_mask=result.metadata["observability"]["eligible"]
        current_mask=historical_mask&(abs(relative_size)<=.1)
        out[name]={"historical_shape_visibility_only_s":longest(result.t_s,historical_mask),
                   "same_trajectory_with_target_size_10pct_s":longest(result.t_s,current_mask),
                   "outer_side_at_epoch_deg":float(angular[np.argmin(abs(result.t_s))]),
                   "initial_relative_velocities_m_s":result.metadata["initial_vel_oso_m_s"].tolist(),
                   "initial_ranges_km":result.metadata["initial_lambda_km"].tolist()}
    return out


def current():
    out={}
    for size,side,alt in ((2,1.818302964,630.),(2,20.,700.),(3,20.,700.),(10,20.,700.)):
        cfg=configuration(size,side,altitude_km=alt)
        cached=load_cached(cfg,size)
        if cached is None:continue
        _,_,data,path=cached
        check=data["verification"]
        out[f"N{size}_side{side:g}_alt{alt:g}"]={
            "file":str(path.relative_to(Path(__file__).resolve().parent.parent)),
            "one_pass_square_s":check["one_pass"]["best_square_duration_s"],
            "one_pass_resolution_2deg_s":check["one_pass"]["best_resolution_2deg_s"],
            "orbital_limits":check["orbit_limits"],
            "full_orbit_pair_minimum_km":check["full_orbit"]["closest_pair"]["distance_km"],
            "full_orbit_max_normal_angle_deg":check["orbital_direction"]["full_orbit"]["max_angle_deg"],
            "physical_violations":check["physical_violations"]}
    return out


def main():
    data={"historical_HCW":historical(),"current_independent_J2_drag":current(),
          "note":"HCW 425 s is shape+visibility only; its same trajectory lasts 70 s with ±10% external-size tolerance. HCW does not represent four independently propagated physical orbits."}
    path=Path(__file__).with_name("first_iteration_comparison.json")
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(path)
    for name,values in data["historical_HCW"].items():
        print(name,values["historical_shape_visibility_only_s"],
              values["same_trajectory_with_target_size_10pct_s"])
    for name,values in data["current_independent_J2_drag"].items():
        print(name,values["one_pass_square_s"],"s, min pair",
              values["full_orbit_pair_minimum_km"],"km")


if __name__=="__main__":main()
