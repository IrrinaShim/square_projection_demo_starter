import unittest

import numpy as np

from square_demo.core import hcw_free, hcw_transition
from square_demo.providers import SCENARIOS, calculate, calculate_free_variants, load_scenario


class CalculationTests(unittest.TestCase):
    def test_hcw_initial_state_and_equations(self):
        n = 0.0010777765959730339
        p0 = np.array([[1.4, 20., -4.], [-2., 3., 10.]])
        v0 = np.array([[0.004, -0.002, 0.003], [-0.001, 0.002, -0.004]])
        t = np.array([-0.01, 0., 0.01])
        p, v = hcw_free(t, p0, v0, n)
        np.testing.assert_allclose(p[1], p0, atol=1e-12)
        np.testing.assert_allclose(v[1], v0, atol=1e-12)
        acceleration = (v[2]-v[0]) / 0.02
        expected = np.column_stack((2*n*v0[:,1]+3*n*n*p0[:,0],
                                    -2*n*v0[:,0], -n*n*p0[:,2]))
        np.testing.assert_allclose(acceleration, expected, atol=1e-10)
        p_matrix,v_matrix = hcw_transition(t,n)
        np.testing.assert_allclose(p,np.einsum("nij,sj->nsi",p_matrix,p0)
                                   +np.einsum("nij,sj->nsi",v_matrix,v0),atol=1e-12)

    def test_inverse_square_and_unforced_drift(self):
        s = load_scenario(SCENARIOS["free_hcw_two_body"])
        s["angular_side_deg"] = np.rad2deg(np.arctan(20/630))
        r = calculate(s)
        zero = np.where(np.isclose(r.t_s, 0))[0][0]
        np.testing.assert_allclose(r.rho_oso_km[zero,1,1],20,atol=1e-12)
        np.testing.assert_allclose(r.image_gnomonic[zero],r.target_image_gnomonic,atol=1e-14)
        np.testing.assert_allclose(r.shape_errors[zero],0,atol=1e-13)
        self.assertTrue(np.array_equal(r.accel_oso_m_s2, np.zeros_like(r.accel_oso_m_s2)))
        self.assertTrue(np.array_equal(r.dv_cumulative_m_s, np.zeros_like(r.dv_cumulative_m_s)))
        self.assertGreater(np.max(abs(r.image_gnomonic[-1]-r.image_gnomonic[zero])), 1e-5)
        self.assertGreater(np.min(r.min_pair_distance_km),0)
        self.assertGreater(r.metadata["min_altitude_km"],0)
        self.assertTrue(r.metadata["separation_meets_illustrative_threshold"])
        s["angular_side_deg"] = 20
        wide = calculate(s)
        self.assertGreater(wide.rho_oso_km[zero,1,1],200)
        self.assertGreater(wide.metadata["max_rho_over_a"],r.metadata["max_rho_over_a"])

    def test_three_variants_and_nodewise_geometry(self):
        s = load_scenario(SCENARIOS["free_hcw_two_body"])
        variants = calculate_free_variants(s)
        self.assertEqual(set(variants),{"ideal","trial","fitted"})
        trial = variants["trial"]
        fitted = variants["fitted"]
        ideal = variants["ideal"]
        self.assertGreater(fitted.metadata["quality"]["zenith_contiguous_duration_s"],
                           trial.metadata["quality"]["zenith_contiguous_duration_s"])
        zero = int(np.argmin(abs(fitted.t_s)))
        np.testing.assert_allclose(fitted.image_gnomonic[zero],fitted.target_image_gnomonic,atol=1e-12)
        self.assertTrue(np.all(fitted.metadata["initial_lambda_km"] > 0))
        self.assertLess(abs(fitted.metadata["observability"]["size_deviation_percent"][zero]),1e-8)
        self.assertTrue(fitted.metadata["observability"]["eligible"][zero])
        self.assertFalse(fitted.metadata["observability"]["eligible"][0])
        self.assertIn("угол места",fitted.metadata["observability"]["reasons"][0])
        self.assertGreater(fitted.metadata["min_altitude_km"],0)
        self.assertTrue(fitted.metadata["separation_meets_illustrative_threshold"])
        strict = load_scenario(SCENARIOS["free_hcw_two_body"])
        strict["safety"]["illustrative_min_separation_km"] = 1000
        strict_result = calculate(strict)
        self.assertFalse(strict_result.metadata["observability"]["safety_illustrative_ok"][zero])
        self.assertTrue(strict_result.metadata["observability"]["eligible"][zero])
        self.assertEqual(fitted.square_sides_gnomonic.shape,(len(fitted.t_s),4))
        self.assertEqual(fitted.square_diagonals_gnomonic.shape,(len(fitted.t_s),2))
        self.assertEqual(fitted.square_angles_rad.shape,(len(fitted.t_s),4))
        self.assertTrue(np.array_equal(fitted.accel_oso_m_s2,np.zeros_like(fitted.accel_oso_m_s2)))
        self.assertGreater(np.max(ideal.accel_oso_m_s2),0)
        self.assertGreater(ideal.dv_cumulative_m_s[-1,2],0)
        np.testing.assert_allclose(ideal.rho_oso_km[0],ideal.rho_oso_km[-1],atol=1e-12)

    def test_legacy_csv_units_order_and_dv(self):
        r = calculate(load_scenario(SCENARIOS["legacy_replay"]))
        self.assertEqual(len(r.t_s),241)
        self.assertEqual(len(r.accel_oso_m_s2),240)
        np.testing.assert_allclose(r.rho_oso_km[0,1],[0,20,0],atol=1e-12)
        np.testing.assert_allclose(r.rho_oso_km[0,2],[0,20,18.96559195525973],atol=1e-12)
        np.testing.assert_allclose(r.rho_oso_km[0,3],[0,0,18.96559195525973],atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(r.accel_oso_m_s2[0,2]),8.037860236637312e-9,places=15)
        np.testing.assert_allclose(r.dv_cumulative_m_s[1,2],
                                   np.linalg.norm(r.accel_oso_m_s2[0,2])*5,atol=1e-14)
        self.assertEqual(r.metadata["zenith_s"],600)
        self.assertIsNone(r.metadata["initial_lambda_km"])
        self.assertTrue(np.isfinite(r.metadata["reference_period_s"]))


if __name__ == "__main__":
    unittest.main()
