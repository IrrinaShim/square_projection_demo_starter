import unittest
import csv
import json
from pathlib import Path

import numpy as np

from orbit_study.evaluate import detailed, geometry, intervals
from orbit_study.model import make_candidate, reconstruct
from square_demo.orbit_j2 import elements_to_state, integrate
from square_demo.providers import SCENARIOS, load_scenario


class PassiveStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenario=load_scenario(SCENARIOS["independent_j2"])

    def test_initial_elements_and_original_four_point_geometry(self):
        s=self.scenario
        candidate=make_candidate(s,2,630,s["angular_side_deg"])
        original=np.stack([elements_to_state(e,s["orbit"]["mu_km3_s2"])
                           for e in s["elements_at_epoch"]])
        np.testing.assert_allclose(candidate["states"],original,atol=1e-8)
        np.testing.assert_allclose(reconstruct(candidate["elements"],s["orbit"]["mu_km3_s2"]),
                                   candidate["states"],atol=1e-8)
        t=np.arange(-450.,451.,5.)
        o=s["orbit"]
        states=integrate(candidate["states"],t,o["mu_km3_s2"],
                         o["earth_radius_km"],o["j2"],2.)
        m=geometry(states,t,s,2,thresholds=(0.1,1.,2.))
        self.assertEqual(max(i["duration_s"] for i in intervals(m,0.1)),425.)
        self.assertEqual(max(i["duration_s"] for i in intervals(m,1.)),205.)
        self.assertEqual(intervals(m,2.),[])

    def test_all_pair_minimum_is_refined_between_nodes(self):
        s=self.scenario
        candidate=make_candidate(s,3,630,s["angular_side_deg"])
        coarse=detailed(candidate,s,0.1,sample_s=60,collision_km=0.02,margin_km=5.)
        fine=detailed(candidate,s,0.1,sample_s=30,collision_km=0.02,margin_km=5.)
        self.assertEqual(coarse["pair_count"],36)
        self.assertEqual(len(coarse["pair_minima"]),36)
        self.assertEqual(coarse["closest_pair"]["pair"],"5-6")
        self.assertEqual(coarse["closest_pair"]["method"],"J2 refined global minimum")
        self.assertAlmostEqual(coarse["closest_pair"]["distance_km"],
                               fine["closest_pair"]["distance_km"],delta=1e-4)
        self.assertAlmostEqual(coarse["closest_pair"]["t_s"],
                               fine["closest_pair"]["t_s"],delta=0.1)
        self.assertGreater(coarse["maximum_chief_latitude_deg"],80)

    def test_hundred_independent_initial_states(self):
        s=self.scenario
        candidate=make_candidate(s,10,700,10)
        self.assertEqual(candidate["states"].shape,(100,2,3))
        self.assertEqual(len(candidate["elements"]),100)
        self.assertEqual(len({tuple(np.round(x[0],8)) for x in candidate["states"]}),100)
        np.testing.assert_allclose(reconstruct(candidate["elements"],s["orbit"]["mu_km3_s2"]),
                                   candidate["states"],atol=1e-7)

    def test_saved_solutions_roundtrip_and_all_pair_files(self):
        root=Path(__file__).resolve().parent.parent/"orbit_study"/"results"
        data=json.loads((root/"solutions.json").read_text(encoding="utf-8"))
        self.assertEqual(set(data["solutions"]),set(map(str,range(2,11))))
        for key,solution in data["solutions"].items():
            n=int(key)
            candidate=make_candidate(self.scenario,n,**solution["parameters"])
            states=reconstruct(solution["elements_at_epoch"],
                               self.scenario["orbit"]["mu_km3_s2"])
            np.testing.assert_allclose(states,candidate["states"],atol=1e-7)
            audit=solution["verification"]
            self.assertEqual(audit["pair_count"],n*n*(n*n-1)//2)
            with (root/audit["all_pair_minima_csv"]).open(encoding="utf-8",newline="") as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),audit["pair_count"])
            closest=min(rows,key=lambda row:float(row["distance_km"]))
            self.assertEqual(closest["pair"],audit["closest_pair"]["pair"])
            self.assertAlmostEqual(float(closest["distance_km"]),
                                   audit["closest_pair"]["distance_km"],places=8)
        polar=json.loads((root/"polar_controls.json").read_text(encoding="utf-8"))
        for key,row in polar["cases"].items():
            n=int(key)
            self.assertGreater(row["verification"]["maximum_chief_latitude_deg"],80)
            self.assertEqual(row["verification"]["pair_count"],n*n*(n*n-1)//2)
            with (root/row["all_pair_minima_csv"]).open(encoding="utf-8",newline="") as f:
                self.assertEqual(len(list(csv.DictReader(f))),row["verification"]["pair_count"])
        convergence=json.loads((root/"minimum_convergence.json").read_text(encoding="utf-8"))
        self.assertLess(abs(convergence["8"]["difference_km"]),1e-5)
        self.assertLess(abs(convergence["9"]["difference_km"]),1e-5)


if __name__=="__main__":
    unittest.main()
