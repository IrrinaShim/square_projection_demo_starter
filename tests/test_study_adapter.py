"""Исследовательские элементы должны воспроизводиться текущей моделью."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from orbit_study.model import reconstruct
from orbit_study.provenance import physics_fingerprint, solutions_fingerprint
from square_demo.providers import calculate
from square_demo.study import (CandidateUnavailable, SOLUTIONS,
                               load_study_case, study_status)


class StudyAdapterTests(unittest.TestCase):
    def test_all_sizes_and_recomputed_selected_pass(self):
        for n in range(2,11):
            with self.subTest(n=n):
                scenario,record,data=load_study_case(n,2.)
                self.assertEqual(len(scenario["elements_at_epoch"]),n*n)
                self.assertEqual(record["satellite_count"],n*n)
                result=calculate(scenario,include_audit=False)
                self.assertEqual(result.r_eci_km.shape,(181,n*n,3))
                self.assertEqual(result.metadata["grid_indices"],record["grid_indices"])
                self.assertEqual(result.metadata["orbital_elements"],record["elements_at_epoch"])
                initial=reconstruct(record["elements_at_epoch"],
                                    scenario["orbit"]["mu_km3_s2"])
                np.testing.assert_allclose(result.r_eci_km[90],initial[:,0],atol=1e-8)
                saved=record["verification"]["repeat"]["2"]["best_pass"]
                duration=result.metadata["quality"]["longest_contiguous_duration_s"]
                self.assertAlmostEqual(duration,saved["duration_s"] if saved else 0,delta=5)
                self.assertEqual(result.metadata["observability"]["eligible"].shape,(181,))
                self.assertEqual(result.min_pair_distance_km.shape,(181,))
                self.assertEqual(result.metadata["observability"]["outer_side_deg"].shape,(181,))
        self.assertEqual(study_status(load_study_case(9)[1]),"нет проверенного решения")

    def test_reject_changed_model_and_results(self):
        data=json.loads(Path(SOLUTIONS).read_text(encoding="utf-8"))
        self.assertEqual(data["physics_fingerprint_sha256"],physics_fingerprint())
        self.assertEqual(data["solutions_fingerprint_sha256"],solutions_fingerprint(data))
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/"solutions.json"
            for field in ("model_version","physics_fingerprint_sha256"):
                tampered=json.loads(json.dumps(data))
                tampered[field]="old"
                tampered["solutions_fingerprint_sha256"]=solutions_fingerprint(tampered)
                p.write_text(json.dumps(tampered),encoding="utf-8")
                with self.assertRaisesRegex(ValueError,"несовместимо"):
                    load_study_case(2,solutions_path=p)
            tampered=json.loads(json.dumps(data))
            tampered["solutions"]["2"]["elements_at_epoch"][1]["a_km"]+=0.1
            p.write_text(json.dumps(tampered),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"содержимое"):
                load_study_case(2,solutions_path=p)

    def test_exact_external_side_required(self):
        self.assertEqual(load_study_case(10,outer_side_deg=10)[0]["angular_side_deg"],10)
        for side in (20,30):
            with self.assertRaises(CandidateUnavailable):
                load_study_case(10,outer_side_deg=side)


if __name__=="__main__":
    unittest.main()
