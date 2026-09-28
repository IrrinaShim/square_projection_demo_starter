import unittest

import numpy as np

from square_demo.core import hcw_free, hcw_transition
from square_demo.orbit_j2 import elements_to_state, state_to_elements, acceleration_j2
from square_demo.grid import corner_indices, grid_indices
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

    def test_independent_j2_elements_projection_and_full_orbit_audit(self):
        scenario = load_scenario(SCENARIOS["independent_j2"])
        mu = scenario["orbit"]["mu_km3_s2"]
        for element in scenario["elements_at_epoch"]:
            state = elements_to_state(element,mu)
            recovered = elements_to_state(state_to_elements(*state,mu),mu)
            np.testing.assert_allclose(recovered,state,atol=1e-8)
        result = calculate(scenario)
        zero = int(np.argmin(abs(result.t_s)))
        np.testing.assert_allclose(result.image_gnomonic[zero],
                                   result.target_image_gnomonic,atol=1e-12)
        self.assertTrue(result.metadata["observability"]["eligible"][zero])
        self.assertEqual(result.metadata["quality"]["zenith_contiguous_duration_s"],425)
        self.assertTrue(np.array_equal(result.accel_oso_m_s2,np.zeros_like(result.accel_oso_m_s2)))
        self.assertTrue(np.array_equal(result.dv_cumulative_m_s,np.zeros_like(result.dv_cumulative_m_s)))
        audit = result.metadata["full_orbit_audit"]
        self.assertEqual(set(audit["pair_minima"]),{"0-1","0-2","0-3","1-2","1-3","2-3"})
        self.assertEqual(set(audit["high_latitude_sample_minima"]),set(audit["pair_minima"]))
        self.assertGreater(audit["max_abs_chief_latitude_deg"],80)
        self.assertLess(audit["pair_minima"]["0-3"]["distance_km"],10)
        self.assertGreater(min(item["distance_km"] for item in audit["pair_minima"].values()),1)
        self.assertEqual(audit["dangerous_events"],[])
        self.assertTrue(result.metadata["full_orbit_meets_illustrative_threshold"])
        position = result.r_eci_km[zero,0]
        j2 = scenario["orbit"]["j2"]
        re = scenario["orbit"]["earth_radius_km"]
        self.assertGreater(np.linalg.norm(acceleration_j2(position,mu,re,j2)
                                          -acceleration_j2(position,mu,re,0)),1e-6)

    def test_grid_3_independent_states_and_danger_detection(self):
        self.assertEqual(grid_indices(2).tolist(),[[0,1],[3,2]])
        self.assertEqual(corner_indices(2).tolist(),[0,1,2,3])
        scenario = load_scenario(SCENARIOS["independent_j2"])
        scenario["grid_size"] = 3
        scenario["safety"]["illustrative_min_separation_km"] = 5
        result = calculate(scenario)
        self.assertEqual(result.r_eci_km.shape,(181,9,3))
        self.assertEqual(result.metadata["grid_indices"],[[0,1,2],[5,4,3],[6,7,8]])
        self.assertEqual(result.metadata["corner_indices"],[0,2,8,6])
        self.assertEqual(len(result.metadata["orbital_elements"]),9)
        np.testing.assert_allclose(result.image_gnomonic[90],result.target_image_gnomonic,atol=1e-12)
        self.assertEqual(result.metadata["full_orbit_audit"]["pair_count"],36)
        self.assertGreater(result.metadata["full_orbit_audit"]["max_abs_chief_latitude_deg"],80)
        self.assertTrue(any(event["pair"]=="5-6" and event["method"]=="J2 refined"
                            for event in result.metadata["full_orbit_audit"]["dangerous_events"]))
        self.assertFalse(result.metadata["full_orbit_meets_illustrative_threshold"])
        self.assertTrue(np.all(result.accel_oso_m_s2==0))

    def test_grid_10_all_pairs_and_no_false_acceptance_at_two_degrees(self):
        scenario = load_scenario(SCENARIOS["independent_j2"])
        scenario["grid_size"] = 10
        scenario["grid_tolerances"]["min_neighbor_angle_deg"] = 2
        result = calculate(scenario)
        self.assertEqual(result.r_eci_km.shape,(181,100,3))
        self.assertEqual(result.image_gnomonic.shape,(181,100,2))
        self.assertEqual(result.target_yoz_km.shape,(100,2))
        self.assertEqual(len(result.metadata["orbital_elements"]),100)
        self.assertEqual(result.metadata["full_orbit_audit"]["pair_count"],4950)
        self.assertEqual(result.metadata["quality"]["zenith_contiguous_duration_s"],0)
        self.assertEqual(result.metadata["quality"]["accepted_nodes"],0)
        self.assertFalse(result.metadata["quality"]["has_positive_duration_interval"])
        zero = int(np.argmin(abs(result.t_s)))
        self.assertLess(abs(result.metadata["observability"]["size_deviation_percent"][zero]),1e-8)
        self.assertIn("истинный угловой шаг",result.metadata["observability"]["reasons"][zero])
        self.assertTrue(np.all(result.accel_oso_m_s2==0))


if __name__ == "__main__":
    unittest.main()
