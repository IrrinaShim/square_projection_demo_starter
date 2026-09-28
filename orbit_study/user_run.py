"""Явно запрошенный повторный анализ одного выбранного набора орбит."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from square_demo.study import STUDY_ROOT, load_study_case, study_status
from .evaluate import detailed
from .model import reconstruct
from .provenance import MODEL_VERSION, physics_fingerprint


def _figure(path, draw):
    figure=Figure(figsize=(8,5),layout="constrained")
    FigureCanvasAgg(figure)
    draw(figure)
    figure.savefig(path,dpi=140)


def _passes_plot(path,audit):
    def draw(fig):
        a,b=fig.subplots(2,1,sharex=True)
        for key,color in (("1","tab:blue"),("2","tab:orange")):
            hits=[p["best"][key] for p in audit["passes"] if p["best"][key]]
            if hits:
                day=[(x["start_s"]+x["end_s"])/172800 for x in hits]
                a.scatter(day,[x["duration_s"] for x in hits],s=23,color=color,label=f"≥{key}°")
                b.scatter(day,[x["min_outer_side_deg"] for x in hits],s=23,color=color,label=f"≥{key}°")
        a.set_ylabel("Длительность, с")
        b.set_ylabel("Мин. внешняя сторона, °")
        b.set_xlabel("Сутки от эпохи")
        for ax in (a,b):
            ax.grid(alpha=.3)
            ax.set_xlim(-.05,audit["horizon_days"]+.05)
            if ax.collections: ax.legend()
            else: ax.text(.5,.5,"Подходящих пролётов нет",transform=ax.transAxes,ha="center")
        fig.suptitle("Повторение принятой наблюдаемой сетки")
    _figure(path,draw)


def _shape_plot(path,audit):
    def draw(fig):
        a,b=fig.subplots(2,1,sharex=True)
        passes=audit["passes"]
        x=np.array([(p["chief_visible_start_s"]+p["chief_visible_end_s"])/172800
                    for p in passes])
        if len(x):
            errors=np.asarray([p["max_shape_errors"] for p in passes],dtype=float)
            a.plot(x,errors[:,0],"o",label="max εb")
            a.plot(x,errors[:,1],"s",label="max εd")
            a.axhline(.1,color="gray",ls="--",label="допуск 0,1")
            b.plot(x,np.rad2deg(errors[:,2]),"o",color="tab:purple",label="max εγ")
            b.axhline(10,color="gray",ls="--",label="допуск 10°")
        a.set_ylabel("Ошибка формы")
        b.set_ylabel("Ошибка угла, °")
        b.set_xlabel("Сутки от эпохи")
        for ax in (a,b):
            ax.grid(alpha=.3)
            if ax.lines: ax.legend()
        fig.suptitle("Максимальные ошибки внешнего квадрата на видимых пролётах")
    _figure(path,draw)


def _distance_plot(path,audit,pair_minima):
    def draw(fig):
        ax=fig.subplots()
        events=list(pair_minima.values())
        ax.scatter([e["t_s"]/86400 for e in events],
                   [e["distance_km"] for e in events],s=7,alpha=.55)
        ax.axhline(audit["collision_threshold_km"],color="tab:red",ls="--",
                   label="сумма радиусов")
        ax.axhline(audit["absolute_minimum_requirement_km"],color="tab:orange",ls="--",
                   label="радиусы + запас")
        closest=audit["closest_pair"]
        ax.scatter([closest["t_s"]/86400],[closest["distance_km"]],
                   color="black",s=45,zorder=3,label=f"min {closest['pair']}")
        ax.set_yscale("log")
        ax.set(xlabel="Сутки от эпохи",ylabel="Минимум каждой пары, км",
               title="Все пары: уточнённые минимумы за горизонт")
        ax.grid(alpha=.3,which="both")
        ax.legend()
    _figure(path,draw)


def run_selected_case(size,threshold_deg,horizon_days,output_parent=None):
    """Запускается только явным вызовом; каждый вызов создаёт отдельную папку."""
    scenario,record,data=load_study_case(size,float(threshold_deg))
    days=float(horizon_days)
    if not np.isfinite(days) or not 0.05<=days<=30:
        raise ValueError("Горизонт должен быть от 0,05 до 30 суток")
    config=data["search_config"]
    candidate={"size":size,"states":reconstruct(record["elements_at_epoch"],
                scenario["orbit"]["mu_km3_s2"])}
    audit=detailed(candidate,scenario,days,
        sample_s=float(config["detailed_sample_s"]),
        integration_step_s=float(config["detailed_integration_step_s"]),
        pass_step_s=float(config["detailed_pass_step_s"]),
        collision_km=float(config["collision_radius_sum_km"]),
        margin_km=float(config["additional_safety_margin_km"]))
    pair_minima=audit.pop("pair_minima")
    parent=Path(output_parent) if output_parent is not None else STUDY_ROOT/"user_runs"
    parent.mkdir(parents=True,exist_ok=True)
    tag=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder=parent/f"N{size}_{days:g}d_{tag}_{uuid4().hex[:6]}"
    folder.mkdir()
    with (folder/"pair_minima.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.writer(f,lineterminator="\n")
        writer.writerow(("pair","t_s","utc","distance_km","latitude_deg","longitude_deg","method"))
        for pair,event in pair_minima.items():
            writer.writerow((pair,event["t_s"],event["utc"],event["distance_km"],
                             event["latitude_deg"],event["longitude_deg"],event["method"]))
    with (folder/"passes.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.writer(f,lineterminator="\n")
        writer.writerow(("threshold_deg","chief_visible_start_s","chief_visible_end_s",
                         "accepted_start_s","accepted_end_s","duration_s",
                         "min_outer_side_deg","max_outer_side_deg","min_true_neighbor_deg",
                         "max_side_error","max_diagonal_error","max_angle_error_deg",
                         "max_point_error_steps","max_step_error"))
        for p in audit["passes"]:
            for key in ("1","2"):
                for interval in p["intervals"][key]:
                    writer.writerow((key,p["chief_visible_start_s"],p["chief_visible_end_s"],
                        interval["start_s"],interval["end_s"],interval["duration_s"],
                        interval["min_outer_side_deg"],interval["max_outer_side_deg"],
                        interval["min_true_neighbor_deg"],*interval["max_shape_errors"],
                        interval["max_point_error_steps"],interval["max_step_error"]))
    _passes_plot(folder/"passes.png",audit)
    _shape_plot(folder/"shape_quality.png",audit)
    _distance_plot(folder/"pair_minima.png",audit,pair_minima)
    status=study_status({"verification":audit},float(threshold_deg),config)
    payload={"schema_version":1,"source_case_N":size,"selected_threshold_deg":threshold_deg,
        "requested_horizon_days":days,"epoch_utc":scenario["epoch_utc"],
        "model_version":MODEL_VERSION,"physics_fingerprint_sha256":physics_fingerprint(),
        "orbit":scenario["orbit"],"observer":scenario["observer"],
        "visibility":scenario["visibility"],"grid_tolerances":scenario["grid_tolerances"],
        "safety":scenario["safety"],"elements_at_epoch":record["elements_at_epoch"],
        "grid_indices":record["grid_indices"],"status":status,"verification":audit,
        "pair_minima_file":"pair_minima.csv","passes_file":"passes.csv",
        "graphs":["passes.png","shape_quality.png","pair_minima.png"]}
    (folder/"case.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2,
        allow_nan=False)+"\n",encoding="utf-8")
    return {"folder":str(folder),"status":status,"closest_pair":audit["closest_pair"],
            "qualifying_passes":audit["repeat"][str(int(threshold_deg))]["qualifying_passes"],
            "horizon_days":days}
