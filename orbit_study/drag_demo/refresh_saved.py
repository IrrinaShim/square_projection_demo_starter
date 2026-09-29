"""Повторная проверка прежних элементов с раздельными критериями квадрата и разрешения.

    .venv/bin/python -m orbit_study.drag_demo.refresh_saved

Старые элементы и обзор поиска сохраняются рядом с новыми результатами.
Динамика и состояния не меняются; каждый КА вновь распространяется независимо.
"""
from __future__ import annotations

import json
from pathlib import Path
from shutil import copyfile

import numpy as np

from orbit_study.drag_demo.run import HERE, run
from orbit_study.drag_demo.search import initial_candidate, prepare_config
from orbit_study.model import reconstruct


def main():
    for size in (2,3,10):
        folder=HERE/"results"/(f"n{size}" if size<10 else "")
        source=folder/"case.json"
        archive=folder/"case_pre_observability_split.json"
        if not archive.exists():
            copyfile(source,archive)
        old=json.loads(archive.read_text(encoding="utf-8"))
        survey=folder/"survey.json"
        rows=json.loads(survey.read_text(encoding="utf-8")) if survey.exists() else []
        cfg=prepare_config(old["config"])
        p=old["parameters"]
        candidate=initial_candidate(cfg,size,p["side_deg"],p["depth_factor"],
                                    p["range_rate_factor"],p["warp_factor"])
        reconstructed=reconstruct(old["elements_at_epoch"],cfg["orbit"]["mu_km3_s2"])
        error=float(np.max(np.abs(candidate["states"]-reconstructed)))
        if error>1e-7:
            raise AssertionError(f"N={size}: сохранённые элементы не восстанавливают состояния: {error}")
        candidate["inverse_states"]=candidate["states"].copy()
        candidate["states"]=reconstructed
        candidate["elements_at_epoch"]=old["elements_at_epoch"]
        result=run(output=folder,size=size,side_deg=p["side_deg"],
                   candidate_override=candidate,rows_override=rows)
        print(f"N={size}: квадрат {result['one_pass']['longest_contiguous_duration_square_s']:g} с; "
              f"при 2° на сетке окна {result['one_pass']['longest_contiguous_duration_2deg_display_s']:g} с "
              f"(аудит 1 с: {result['one_pass']['longest_contiguous_duration_2deg_s']:g} с); "
              f"полный виток {result['full_orbit_ok']}; элементы Δ={error:.2e}")


if __name__=="__main__":
    main()
