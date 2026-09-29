"""Инварианты новой обратной задачи и сохраняемого ограниченного поиска."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest

import numpy as np

from orbit_study.passive_optimizer import (configuration, corner_seed_hcw,
    geometry_at_epoch, load_cached, optimize, rotation_seed)
from square_demo.core import project
from square_demo.grid import corner_indices, grid_indices
from square_demo.orbit_j2 import orbital_basis


class PassiveOptimizerTests(unittest.TestCase):
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
            self.assertEqual(len(result["candidate"]["elements_at_epoch"]),4)
            self.assertLess(result["verification"]["one_pass"]["initial_target_max_error_gnomonic"],1e-10)
            rows=[json.loads(line) for line in (path.parent/"candidates.jsonl").read_text().splitlines()]
            self.assertGreaterEqual(len(rows),4)
            self.assertTrue(all("rejection_reasons" in row for row in rows))
            cached=load_cached(cfg,2,folder)
            self.assertIsNotNone(cached)
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


if __name__=="__main__":unittest.main()
