"""Версия численных определений, по которым сохранены решения исследования."""
from __future__ import annotations

import hashlib
import inspect
import json

from square_demo import core, grid, orbit_j2, providers
from . import evaluate, model

MODEL_VERSION = "passive-j2-grid-2026-09-v1"
FRAME = "conditional ECI: Z Earth spin axis; Greenwich angle=0 at epoch"


def solutions_fingerprint(data):
    """Контроль содержимого, включая элементы и сохранённые выводы."""
    content={key:value for key,value in data.items()
             if key!="solutions_fingerprint_sha256"}
    encoded=json.dumps(content,sort_keys=True,ensure_ascii=False,
                       separators=(",",":"),allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def physics_fingerprint():
    functions = (
        orbit_j2.elements_to_state, orbit_j2.state_to_elements,
        orbit_j2.acceleration_j2, orbit_j2.rk4, orbit_j2.integrate,
        orbit_j2.orbital_basis, orbit_j2.centroid_relative_frame,
        core.observer_eci, core.image_frame, core.project,
        core.square_geometry, core.observability, core.visibility_summary,
        core.assemble_absolute, core.spatial_size_metrics, providers.independent_j2,
        grid.grid_indices, grid.corner_indices, grid.grid_metrics,
        grid.grid_acceptance, model.make_candidate,
        evaluate.geometry, evaluate.intervals, evaluate.detailed,
    )
    digest = hashlib.sha256(MODEL_VERSION.encode())
    for function in functions:
        digest.update(f"\n{function.__module__}.{function.__name__}\n".encode())
        digest.update(inspect.getsource(function).encode())
    return digest.hexdigest()
