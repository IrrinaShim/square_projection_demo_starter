"""Инварианты новой обратной задачи и сохраняемого ограниченного поиска."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import patch

import numpy as np

from orbit_study.passive_optimizer import (configuration, corner_seed_hcw,
    geometry_at_epoch, load_cached, optimize, rotation_seed, survey, verify, _candidate)
from orbit_study.orbit_direction import audit_orbit_normals
from square_demo.core import project
from square_demo.grid import corner_indices, grid_indices
from square_demo.orbit_j2 import orbital_basis, elements_to_state


class PassiveOptimizerTests(unittest.TestCase):
    def test_same_inclination_different_raan_rejected(self):
        cfg=configuration(2,20.)
        mu=cfg["orbit"]["mu_km3_s2"]
        states=[]
        for raan in (0.,30.,180.):
            states.append(elements_to_state({"a_km":7071.,"e":0.,
                "inclination_deg":97.8,"raan_deg":raan,
                "arg_perigee_deg":0.,"true_anomaly_deg":0.},mu))
        result=audit_orbit_normals(np.asarray(states)[None],np.array([0.]))
        self.assertFalse(result["ok"])
        self.assertLess(result["min_dot"],0.)
        self.assertGreater(result["max_angle_deg"],90.)
        near=audit_orbit_normals(np.asarray(states[:2])[None],np.array([0.]))
        self.assertFalse(near["ok"])
        self.assertGreater(near["min_dot"],0.)
        self.assertGreater(near["max_angle_deg"],5.)
        candidate=rotation_seed(cfg,2)
        candidate["states"][1]=states[2]
        self.assertIn("нормали", " ".join(survey(candidate,cfg)["violations"]))
        candidate=_candidate(candidate["states"],candidate["target"],2,cfg,"контроль RAAN")
        final=verify(candidate,cfg)
        self.assertFalse(final["orbital_direction"]["epoch"]["ok"])
        self.assertFalse(final["orbital_direction"]["one_pass"]["ok"])
        self.assertFalse(final["orbital_direction"]["full_orbit"]["ok"])
        self.assertIn("нормали", " ".join(final["physical_violations"]))

    def test_regular_gnomonic_rays_and_corner_order(self):
        for size in (2,3,10):
            cfg=configuration(size,20.)
            chief,basis,observer,rays,target,_=geometry_at_epoch(cfg,size)
            ids=grid_indices(size)
            self.assertEqual(corner_indices(size).tolist(),
                             [int(ids[0,0]),int(ids[0,-1]),int(ids[-1,-1]),int(ids[-1,0])])
            np.testing.assert_allclose(target[ids[0]],
                np.column_stack((np.linspace(0,np.tan(np.deg2rad(20)),size),np.zeros(size))))
            candidate=rotation_seed(cfg,size)
            image,valid,_=project(candidate["states"][None,:,0],observer[None],
                                   orbital_basis(chief[None]),cfg["orbit"]["earth_radius_km"])
            self.assertTrue(valid.all())
            self.assertLess(np.max(abs(image[0]-target)),1e-12)

    def test_hcw_seed_and_resume(self):
        cfg=configuration(2,1.818302964,altitude_km=630.)
        seed=corner_seed_hcw(cfg,2,1.,150.)
        self.assertEqual(len(seed["elements_at_epoch"]),4)
        _,_,observer,_,target,_=geometry_at_epoch(cfg,2)
        image,valid,_=project(seed["states"][None,:,0],observer[None],
            orbital_basis(seed["states"][0][None]),cfg["orbit"]["earth_radius_km"])
        self.assertTrue(valid.all())
        self.assertLess(np.max(abs(image[0]-target)),1e-12)
        with TemporaryDirectory() as tmp:
            folder=Path(tmp)
            result,path=optimize(cfg,2,budget=4,output_root=folder,resume=False)
            self.assertEqual(result["N"],2)
            self.assertEqual(result["verification"]["full_orbit"]["pair_count"],6)
            for scope in ("epoch","one_pass","full_orbit"):
                self.assertTrue(result["verification"]["orbital_direction"][scope]["ok"])
            self.assertEqual(len(result["candidate"]["elements_at_epoch"]),4)
            self.assertLess(result["verification"]["one_pass"]["initial_target_max_error_gnomonic"],1e-10)
            rows=[json.loads(line) for line in (path.parent/"candidates.jsonl").read_text().splitlines()]
            self.assertGreaterEqual(len(rows),4)
            self.assertTrue(all("rejection_reasons" in row for row in rows))
            cached=load_cached(cfg,2,folder)
            self.assertIsNotNone(cached)
            stale=json.loads(path.read_text())
            stale["schema_version"]=2
            path.write_text(json.dumps(stale))
            with self.assertRaises(ValueError):load_cached(cfg,2,folder)
            stale["schema_version"]=result["schema_version"]
            stale["verification"]["orbital_direction"]["full_orbit"]["max_angle_deg"]=6.
            path.write_text(json.dumps(stale))
            with self.assertRaises(ValueError):load_cached(cfg,2,folder)
            path.write_text(json.dumps(result))
            self.assertIsNone(load_cached(configuration(2,1.818302964,
                latitude_deg=26.,altitude_km=630.),2,folder))
            result2,_=optimize(cfg,2,budget=4,output_root=folder,resume=True)
            self.assertGreater(result2["survey_count_total"],result["survey_count_total"])

    def test_cancel_before_first_propagation(self):
        cfg=configuration(2,20.)
        event=Event();event.set()
        with TemporaryDirectory() as tmp:
            with self.assertRaises(InterruptedError):
                optimize(cfg,2,budget=4,output_root=Path(tmp),cancel_event=event)

    def test_reverse_orbit_is_never_saved_as_best(self):
        cfg=configuration(2,20.)
        bad=rotation_seed(cfg,2)
        bad["states"][1]=elements_to_state({"a_km":7071.,"e":0.,
            "inclination_deg":97.8,"raan_deg":180.,
            "arg_perigee_deg":0.,"true_anomaly_deg":0.},cfg["orbit"]["mu_km3_s2"])
        bad=_candidate(bad["states"],bad["target"],2,cfg,"разнонаправленный контроль")
        with TemporaryDirectory() as tmp, \
             patch("orbit_study.passive_optimizer.archived_seed",return_value=None), \
             patch("orbit_study.passive_optimizer.rotation_seed",return_value=bad), \
             patch("orbit_study.passive_optimizer.corner_seed_hcw",return_value=bad):
            with self.assertRaisesRegex(ValueError,"Подходящий пассивный вариант не найден"):
                optimize(cfg,2,budget=4,output_root=Path(tmp),resume=False)
            folders=list(Path(tmp).iterdir())
            self.assertEqual(len(folders),1)
            self.assertFalse((folders[0]/"best.json").exists())
            rejected=json.loads((folders[0]/"rejected.json").read_text())
            self.assertIn("нормали", " ".join(rejected["physical_violations"]))


if __name__=="__main__":unittest.main()
