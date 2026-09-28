"""Контроль крупных сеток на полярной орбите 97,8° за одни сутки."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from square_demo.providers import SCENARIOS, load_scenario
from .evaluate import detailed
from .model import make_candidate

ROOT=Path(__file__).resolve().parent


def run():
    base=load_scenario(SCENARIOS["independent_j2"])
    output=ROOT/"results"
    results={"description":"polar comparison, not selected solution",
             "epoch_utc":base["epoch_utc"],"cases":{}}
    for n in (9,10):
        candidate=make_candidate(base,n,700,10,velocity_factor=1.,inclination_deg=97.8)
        audit=detailed(candidate,base,1.,sample_s=60.,integration_step_s=10.,
            pass_step_s=5.,collision_km=.02,margin_km=1.)
        minima=audit.pop("pair_minima")
        filename=f"polar_n{n}_pair_minima.csv"
        with (output/filename).open("w",encoding="utf-8",newline="") as f:
            writer=csv.writer(f,lineterminator="\n")
            writer.writerow(("pair","t_s","utc","distance_km","latitude_deg",
                             "longitude_deg","method"))
            for pair,event in minima.items():
                writer.writerow((pair,event["t_s"],event["utc"],event["distance_km"],
                                 event["latitude_deg"],event["longitude_deg"],event["method"]))
        results["cases"][str(n)]={"parameters":candidate["parameters"],
            "satellite_count":n*n,"elements_at_epoch":candidate["elements"],
            "grid_indices":candidate["grid_indices"],
            "all_pair_minima_csv":filename,"verification":audit}
        print(f"N={n}, 97.8°: {audit['pair_count']} пар, "
              f"|широта|≤{audit['maximum_chief_latitude_deg']:.2f}°, "
              f"min={audit['closest_pair']['distance_km']:.6f} км",flush=True)
    (output/"polar_controls.json").write_text(json.dumps(results,ensure_ascii=False,
        indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    run()
