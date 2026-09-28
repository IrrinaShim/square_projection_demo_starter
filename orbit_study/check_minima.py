"""Повторная J₂-проверка двух критичных минимумов: N=8 и N=9."""
from __future__ import annotations

import json
from pathlib import Path

from square_demo.providers import SCENARIOS, load_scenario
from .evaluate import detailed
from .model import make_candidate

ROOT=Path(__file__).resolve().parent


def run():
    base=load_scenario(SCENARIOS["independent_j2"])
    results=ROOT/"results"
    data=json.loads((results/"solutions.json").read_text(encoding="utf-8"))
    checks={}
    for n in (8,9):
        row=data["solutions"][str(n)]
        candidate=make_candidate(base,n,**row["parameters"])
        fine=detailed(candidate,base,1.,sample_s=30.,integration_step_s=5.,pass_step_s=5.)
        old=row["verification"]["closest_pair"]
        new=fine["closest_pair"]
        checks[str(n)]={"coarse":old,"fine":new,
            "difference_km":new["distance_km"]-old["distance_km"],
            "difference_s":new["t_s"]-old["t_s"],
            "fine_pair_count":fine["pair_count"],
            "sample_s":30,"integration_step_s":5}
        print(f"N={n}: {old['distance_km']:.9f} -> {new['distance_km']:.9f} км")
    (results/"minimum_convergence.json").write_text(
        json.dumps(checks,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    run()
