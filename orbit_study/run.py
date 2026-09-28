"""Запуск из корня: .venv/bin/python -m orbit_study.run [--quick|--full]."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from square_demo.providers import load_scenario
from square_demo.orbit_j2 import integrate
from .model import make_candidate, reconstruct
from .evaluate import detailed, survey, geometry
from .provenance import MODEL_VERSION, FRAME, physics_fingerprint, solutions_fingerprint

ROOT=Path(__file__).resolve().parent.parent
HERE=Path(__file__).resolve().parent


def designs(config):
    sequence=[]
    for height in config["heights_km"]:
        for side in config["outer_side_deg"]:
            sequence.append(dict(height_km=height,side_deg=side))
    for side in (5,10,20):
        for v in (0.75,1.25):
            sequence.append(dict(height_km=700,side_deg=side,velocity_factor=v))
    for phase in (-60,60):
        sequence.append(dict(height_km=700,side_deg=10,phase_s=phase))
    sequence.append(dict(height_km=700,side_deg=10,inclination_deg=60))
    for depth in (0.8,1.2):
        sequence.append(dict(height_km=700,side_deg=10,depth_factor=depth))
    return sequence[:config["survey_candidate_budget_per_size"]]


def _clean_survey(item):
    return {"parameters":item["candidate"]["parameters"],
            "best":item["best"],
            "maximum_point_error_steps":item["maximum_point_error_steps"],
            "minimum_neighbor_angle_deg":item["minimum_neighbor_angle_deg"],
            "minimum_elevation_deg":item["minimum_elevation_deg"]}


def _select(rows):
    for key in ("2","1"):
        feasible=[r for r in rows if r["best"][key] is not None]
        if feasible:
            return max(feasible,key=lambda r:(r["best"][key]["duration_s"]*
                    r["best"][key]["min_outer_side_deg"],
                    r["best"][key]["duration_s"])),key
    return max(rows,key=lambda r:r["minimum_neighbor_angle_deg"]),"none"


def _pareto(rows,key):
    feasible=[r for r in rows if r["best"][key] is not None]
    out=[]
    for row in feasible:
        duration=row["best"][key]["duration_s"]
        side=row["best"][key]["min_outer_side_deg"]
        dominated=any(other is not row and
            other["best"][key]["duration_s"]>=duration and
            other["best"][key]["min_outer_side_deg"]>=side and
            (other["best"][key]["duration_s"]>duration or
             other["best"][key]["min_outer_side_deg"]>side)
            for other in feasible)
        if not dominated:
            out.append({"parameters":row["candidate"]["parameters"],
                        "duration_s":duration,"min_outer_side_deg":side})
    return sorted(out,key=lambda x:(-x["duration_s"],-x["min_outer_side_deg"]))


def _plot(size,record,path):
    audit=record["verification"]
    fig,axes=plt.subplots(2,1,figsize=(8,5),layout="constrained")
    for threshold,color in (("1","tab:blue"),("2","tab:orange")):
        accepted=[p["best"][threshold] for p in audit["passes"] if p["best"][threshold]]
        if accepted:
            times=np.array([(e["start_s"]+e["end_s"])/2/86400 for e in accepted])
            axes[0].scatter(times,[e["duration_s"] for e in accepted],
                            label=f"шаг ≥{threshold}°",c=color,s=18)
            axes[1].scatter(times,[e["min_outer_side_deg"] for e in accepted],
                            label=f"шаг ≥{threshold}°",c=color,s=18)
    axes[0].set_ylabel("Длительность, с")
    axes[1].set_ylabel("Мин. внешняя сторона, °")
    axes[1].set_xlabel("Сутки от эпохи")
    for ax in axes:
        ax.grid(True,alpha=.3)
        ax.set_xlim(-.03,audit["horizon_days"]+.03)
        ax.legend(loc="best") if ax.collections else None
        if not ax.collections:
            ax.text(.5,.5,"Принятых пролётов нет",transform=ax.transAxes,ha="center")
    fig.suptitle(f"N={size}; J₂ без тяги; все КА и все ячейки")
    fig.savefig(path,dpi=140)
    plt.close(fig)


def _plot_first_pass(candidate,scenario,path):
    size=candidate["size"]
    orbit=scenario["orbit"]
    t=np.arange(-450.,451.,5.)
    states=integrate(candidate["states"],t,orbit["mu_km3_s2"],
                     orbit["earth_radius_km"],orbit["j2"],2.)
    g=geometry(states,t,scenario,size)
    fig,axes=plt.subplots(3,1,figsize=(8,7),sharex=True,layout="constrained")
    axes[0].plot(t,g["outer_side_deg"],label="фактическая внешняя сторона")
    axes[0].axhline(candidate["parameters"]["side_deg"],color="gray",ls="--",
                    label="начальная заданная")
    axes[0].set_ylabel("Внешняя сторона, °")
    axes[0].legend(loc="best")
    axes[1].plot(t,g["min_true_neighbor_deg"],color="tab:purple",
                 label="минимальный истинный шаг")
    for angle,color in ((1,"tab:blue"),(2,"tab:orange")):
        axes[1].axhline(angle,ls="--",color=color,label=f"порог {angle}°")
    axes[1].set_ylabel("Шаг соседей, °")
    axes[1].legend(loc="best")
    axes[2].plot(t,g["min_elevation_deg"],color="tab:brown")
    axes[2].axhline(scenario["visibility"]["min_elevation_deg"],color="gray",ls="--")
    for key,color in (("1","tab:blue"),("2","tab:orange")):
        axes[2].fill_between(t,0,1,where=g["accepted"][key],color=color,alpha=.18,
                             transform=axes[2].get_xaxis_transform(),
                             label=f"вся сетка, шаг ≥{key}°")
    axes[2].set_ylabel("Мин. угол места, °")
    axes[2].set_xlabel("Время от эпохи, с")
    axes[2].legend(loc="best")
    for ax in axes: ax.grid(True,alpha=.3)
    fig.suptitle(f"N={size}; первый пролёт, J₂ без тяги")
    fig.savefig(path,dpi=140)
    plt.close(fig)


def run(config,quick=False,full=False,sizes=None,resume=False):
    base=load_scenario(ROOT/config["source_scenario"])
    output=HERE/"results"
    output.mkdir(exist_ok=True)
    all_survey={}
    solutions={"schema_version":1,"epoch_utc":base["epoch_utc"],
        "model_version":MODEL_VERSION,"physics_fingerprint_sha256":physics_fingerprint(),
        "frame":FRAME,
        "elements":"osculating classical Kepler a,e,i,RAAN,argument of perigee,true anomaly; degrees/km",
        "dynamics":"independent RK4 nonlinear central gravity + J2; no thrust",
        "orbit_constants":base["orbit"],"observer":base["observer"],
        "visibility":base["visibility"],"grid_tolerances":base["grid_tolerances"],
        "search_config":config,"solutions":{}}
    summary=[]
    if resume and (output/"solutions.json").exists():
        prior=json.loads((output/"solutions.json").read_text(encoding="utf-8"))
        solutions["solutions"]=prior["solutions"]
        all_survey=json.loads((output/"survey.json").read_text(encoding="utf-8"))
        with (output/"summary.csv").open(encoding="utf-8",newline="") as f:
            summary=list(csv.DictReader(f))
        replacements=set(sizes or config["sizes"])
        summary=[row for row in summary if int(row["N"]) not in replacements]
    for size in (sizes or config["sizes"]):
        rows=[]
        errors=[]
        for params in designs(config):
            try:
                candidate=make_candidate(base,size,**params)
                rows.append(survey(candidate,base,config["survey_sample_s"]))
            except ValueError as exc:
                errors.append({"parameters":params,"error":str(exc)})
        if not rows:
            raise RuntimeError(f"N={size}: все пробные элементы отвергнуты")
        selected,key=_select(rows)
        candidate=selected["candidate"]
        horizon=(0.25 if quick else config["detailed_horizon_base_days"]
                 if full else min(config["detailed_horizon_base_days"],
                        config["detailed_horizon_budget_days_by_size"][str(size)]))
        print(f"N={size}: {len(rows)} допустимых начальных наборов; выбран "
              f"{candidate['parameters']}, горизонт {horizon:g} сут",flush=True)
        audit=detailed(candidate,base,horizon,config["detailed_sample_s"],
            config["detailed_integration_step_s"],config["detailed_pass_step_s"],
            collision_km=config["collision_radius_sum_km"],
            margin_km=config["additional_safety_margin_km"])
        assert len(audit["pair_minima"])==size*size*(size*size-1)//2
        reconstructed=reconstruct(candidate["elements"],base["orbit"]["mu_km3_s2"])
        residual=float(np.max(np.abs(reconstructed-candidate["states"])))
        assert residual<1e-7,residual
        minima=audit.pop("pair_minima")
        pair_csv=f"grid_n{size}_pair_minima.csv"
        with (output/pair_csv).open("w",encoding="utf-8",newline="") as f:
            writer=csv.writer(f,lineterminator="\n")
            writer.writerow(("pair","time_s","utc","distance_km","latitude_deg",
                             "longitude_deg","method"))
            for pair,event in minima.items():
                writer.writerow((pair,event["t_s"],event["utc"],event["distance_km"],
                                 event["latitude_deg"],event["longitude_deg"],event["method"]))
        audit["all_pair_minima_csv"]=pair_csv
        required=config["collision_radius_sum_km"]+config["additional_safety_margin_km"]
        violations=[]
        if audit["closest_pair"]["distance_km"]<config["collision_radius_sum_km"]:
            violations.append("геометрический порог столкновения")
        elif audit["closest_pair"]["distance_km"]<required:
            violations.append("дополнительный запас расстояния")
        if audit["minimum_altitude_km"]<config["minimum_model_altitude_km"]:
            violations.append("высота ниже условного предела применимости J2 без атмосферы")
        if horizon<config["detailed_horizon_base_days"]:
            violations.append("базовый 30-суточный горизонт не проверен")
        elif audit["repeat"]["1"]["qualifying_passes"]<2:
            violations.append("повторяемость квадрата за 30 суток не найдена")
        if audit["repeat"]["2"]["qualifying_passes"]==0:
            violations.append("порог различимости 2° не выполнен")
        record={"N":size,"satellite_count":size*size,"grid_indices":candidate["grid_indices"],
            "parameters":candidate["parameters"],"selected_survey_threshold_deg":key,
            "selected_survey":_clean_survey(selected),
            "pareto_survey_1deg":_pareto(rows,"1"),
            "pareto_survey_2deg":_pareto(rows,"2"),
            "elements_at_epoch":candidate["elements"],
            "initial_state_roundtrip_max_error":residual,
            "verification":audit,
            "unmet_constraints":violations,
            "mission_recommendation":False,
            "verified_30_days":bool(horizon>=config["detailed_horizon_base_days"])}
        solutions["solutions"][str(size)]=record
        all_survey[str(size)]={"candidates_evaluated":len(rows),
                               "invalid_initial_states":errors,
                               "candidates":[_clean_survey(row) for row in rows]}
        _plot(size,record,output/f"grid_n{size}_passes.png")
        _plot_first_pass(candidate,base,output/f"grid_n{size}_first_pass.png")
        closest=audit["closest_pair"]
        for threshold in ("1","2"):
            repeat=audit["repeat"][threshold]
            best=repeat["best_pass"]
            summary.append({"N":size,"KA":size*size,"threshold_deg":threshold,
                "height_km":candidate["parameters"]["height_km"],
                "target_outer_side_deg":candidate["parameters"]["side_deg"],
                "horizon_days":horizon,"visible_passes":audit["visible_pass_count"],
                "qualifying_passes":repeat["qualifying_passes"],
                "repeat_span_days":repeat["repeat_span_days"],
                "best_duration_s":best["duration_s"] if best else 0,
                "best_min_outer_side_deg":best["min_outer_side_deg"] if best else "",
                "best_max_outer_side_deg":best["max_outer_side_deg"] if best else "",
                "closest_pair":closest["pair"],
                "closest_distance_km":closest["distance_km"],
                "closest_time_s":closest["t_s"],
                "minimum_altitude_km":audit["minimum_altitude_km"],
                "threshold_separation_ok":closest["distance_km"]>=
                    config["collision_radius_sum_km"]+config["additional_safety_margin_km"],
                "verified_30_days":record["verified_30_days"],
                "unmet_constraints":"; ".join(violations)})
        solutions["solutions_fingerprint_sha256"]=solutions_fingerprint(solutions)
        (output/"solutions.json").write_text(json.dumps(solutions,ensure_ascii=False,
            indent=2,allow_nan=False)+"\n",encoding="utf-8")
        (output/"survey.json").write_text(json.dumps(all_survey,ensure_ascii=False,
            indent=2,allow_nan=False)+"\n",encoding="utf-8")
        with (output/"summary.csv").open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=summary[0].keys(),lineterminator="\n")
            writer.writeheader();writer.writerows(sorted(summary,key=lambda row:
                (int(row["N"]),int(row["threshold_deg"]))))
        print(f"  {audit['pair_count']} пар, min {closest['distance_km']:.6f} км; "
              f"пролётов ≥1°/{'≥2°'}: {audit['repeat']['1']['qualifying_passes']}/"
              f"{audit['repeat']['2']['qualifying_passes']}",flush=True)
    return solutions,summary


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--quick",action="store_true",help="по 0.25 суток: отладка")
    parser.add_argument("--full",action="store_true",help="30 суток для каждого N")
    parser.add_argument("--sizes",type=int,nargs="+",help="подмножество N=2..10")
    parser.add_argument("--resume",action="store_true",help="сохранить остальные N при --sizes")
    args=parser.parse_args()
    if args.quick and args.full:
        parser.error("--quick и --full несовместимы")
    config=json.loads((HERE/"search.json").read_text(encoding="utf-8"))
    if args.sizes and any(n<2 or n>10 for n in args.sizes):
        parser.error("N должно быть 2..10")
    run(config,args.quick,args.full,args.sizes,args.resume)
