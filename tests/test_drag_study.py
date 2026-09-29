"""Проверки новой физики, сохранённых элементов и межузлового минимума."""
import json
import unittest

import numpy as np

from orbit_study.drag_demo.audit import audit_one_pass
from orbit_study.drag_demo.run import HERE, model_fingerprint
from orbit_study.drag_model import density_kg_m3, drag_acceleration_km_s2
from orbit_study.model import reconstruct
from square_demo.core import project
from square_demo.drag_study import load_or_search
from square_demo.drag_provider import calculate_drag_case
from square_demo.grid import expected_neighbor_angle_deg


class DragStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved=json.loads((HERE/"results"/"case.json").read_text(encoding="utf-8"))

    def test_drag_relative_to_rotating_atmosphere(self):
        cfg=self.saved["config"]
        r=np.array([7071.,0.,0.])
        v=np.cross([0,0,cfg["observer"]["earth_rotation_rad_s"]],r)
        a=drag_acceleration_km_s2(r,v,6371.,cfg["observer"]["earth_rotation_rad_s"],
                                  cfg["spacecraft"],cfg["atmosphere"])
        self.assertLess(np.linalg.norm(a),1e-25)
        self.assertAlmostEqual(density_kg_m3(700.,cfg["atmosphere"]),4.6e-13)
        faster=v+np.array([0.,7.,0.])
        self.assertLess(drag_acceleration_km_s2(r,faster,6371.,
            cfg["observer"]["earth_rotation_rad_s"],cfg["spacecraft"],
            cfg["atmosphere"])[1],0)

    def test_saved_elements_and_pass(self):
        p=self.saved
        self.assertEqual(p["model_fingerprint_sha256"],model_fingerprint())
        c,candidate,audit,source=load_or_search(10,20,2.)
        states=reconstruct(p["elements_at_epoch"],c["orbit"]["mu_km3_s2"])
        np.testing.assert_allclose(states,candidate["states"],atol=1e-7)
        result=calculate_drag_case(c,candidate)
        self.assertEqual(result.r_eci_km.shape[1],100)
        self.assertEqual(audit["pair_count"],4950)
        self.assertEqual(result.metadata["quality"]["longest_contiguous_duration_s"],0.)
        self.assertFalse(audit["verified_acceptable"])
        self.assertGreater(audit["closest_pair"]["distance_km"],1.02)
        self.assertEqual(audit["horizon_s"][0],0.)
        self.assertGreater(audit["horizon_s"][1],5000.)
        self.assertLess(p["orbit_closeness"]["a_range_km"][1]-p["orbit_closeness"]["a_range_km"][0],100.)
        self.assertLess(p["orbit_closeness"]["e_range"][1],.03)
        epoch=int(np.argmin(np.abs(result.t_s)))
        self.assertLess(np.max(np.linalg.norm(result.image_gnomonic[epoch]
                              -result.target_image_gnomonic,axis=1)),2e-12)
        self.assertLess(expected_neighbor_angle_deg(10,20),2.)
        self.assertTrue(np.allclose(result.rho_oso_km.mean(axis=1),0,atol=1e-9))
        basis=result.metadata["local_basis_eci"]
        absolute=result.metadata["center_eci_km"][:,None,:]+np.einsum(
            "tij,tsj->tsi",basis,result.rho_oso_km)
        np.testing.assert_allclose(absolute,result.r_eci_km,atol=1e-9)
        image,valid,_=project(absolute,result.observer_eci_km,basis,
                              c["orbit"]["earth_radius_km"])
        np.testing.assert_allclose(image[valid],result.image_gnomonic[valid],atol=1e-12)
        center=90
        finite_difference=(result.rho_oso_km[center+1]-result.rho_oso_km[center-1])/10
        self.assertLess(np.max(np.abs(finite_difference-
            result.metadata["local_velocity_oso_km_s"][center])),1e-5)
        self.assertTrue(np.all(result.accel_oso_m_s2==0))

    def test_old_candidate_rejected_and_smaller_grids_retained(self):
        old=json.loads((HERE/"results"/"case_unconstrained_legacy.json").read_text(encoding="utf-8"))
        self.assertGreater(max(entry["a_km"] for entry in old["elements_at_epoch"]),9000.)
        for size in (2,3):
            c,candidate,audit,_=load_or_search(size,20,2.)
            self.assertTrue(audit["verified_acceptable"])
            self.assertEqual(len(candidate["elements_at_epoch"]),size*size)
            self.assertGreater(audit["horizon_s"][1],5000.)
            self.assertGreater(audit["closest_pair"]["distance_km"],audit["threshold_km"])

    def test_inter_node_pair_minimum(self):
        cfg=self.saved["config"]
        states=np.zeros((2,4,2,3))
        states[:,0,0]=[7000,0,0]
        states[0,1,0]=[7001,-1,0]
        states[1,1,0]=[7001,1,0]
        states[:,1,1]=[0,1,0]
        states[:,2,0]=[7100,0,0]
        states[:,3,0]=[7200,0,0]
        checked=audit_one_pass(states,np.array([-1.,1.]),cfg,threshold_km=1.5)
        self.assertEqual(checked["pair_count"],6)
        self.assertEqual(checked["closest_pair"]["pair"],"0-1")
        self.assertLess(checked["closest_pair"]["distance_km"],1.01)
        self.assertLess(abs(checked["closest_pair"]["t_s"]),.01)
        self.assertEqual(len(checked["dangerous_events"]),1)
        self.assertIn("utc",checked["dangerous_events"][0])
        self.assertIn("latitude_deg",checked["dangerous_events"][0])

    def test_changed_observer_recalculates_initial_states(self):
        _,first,_,_=load_or_search(2,20,2.,budget=1)
        cfg,second,_,source=load_or_search(2,20,2.,latitude_deg=26.,budget=1)
        self.assertEqual(cfg["observer"]["latitude_deg"],26.)
        self.assertIn("новый",source)
        self.assertGreater(np.max(np.abs(first["states"]-second["states"])),1.)


if __name__=="__main__": unittest.main()
