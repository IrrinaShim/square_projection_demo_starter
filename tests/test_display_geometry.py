"""Центр локальной системы не должен менять абсолютную наблюдаемую картину."""
from copy import deepcopy
import unittest

import numpy as np

from square_demo.core import assemble_absolute, spatial_size_metrics
from square_demo.orbit_j2 import (centroid_relative_frame, elements_to_state,
                                   integrate)
from square_demo.providers import calculate
from square_demo.study import load_study_case


class DisplayGeometryTests(unittest.TestCase):
    def test_centroid_frame_and_projection_invariance(self):
        scenario,_,_=load_study_case(2,2.,outer_side_deg=20.)
        result=calculate(scenario,include_audit=False)
        orbit=scenario["orbit"]
        initial=np.stack([elements_to_state(e,orbit["mu_km3_s2"])
                          for e in scenario["elements_at_epoch"]])
        states=integrate(initial,result.t_s,orbit["mu_km3_s2"],
                         orbit["earth_radius_km"],orbit["j2"],
                         scenario["integration"]["max_step_s"])
        center,center_velocity,basis,rho,velocity=centroid_relative_frame(
            states,orbit["mu_km3_s2"],orbit["earth_radius_km"],orbit["j2"])
        np.testing.assert_allclose(center,states[:,:,0].mean(axis=1),atol=1e-12)
        np.testing.assert_allclose(center_velocity,states[:,:,1].mean(axis=1),atol=1e-12)
        np.testing.assert_allclose(rho,result.rho_oso_km,atol=1e-9)
        np.testing.assert_allclose(rho.mean(axis=1),0,atol=1e-10)
        reconstructed=center[:,None,:]+np.einsum("nij,nsj->nsi",basis,rho)
        np.testing.assert_allclose(reconstructed,result.r_eci_km,atol=1e-9)
        numerical_velocity=np.gradient(rho,result.t_s,axis=0,edge_order=2)
        self.assertLess(np.max(abs(numerical_velocity[1:-1]-velocity[1:-1])),2e-6)
        metadata=deepcopy(result.metadata)
        metadata.pop("local_frame_origin")
        other=assemble_absolute(result.t_s,result.r_eci_km,basis,
            result.observer_eci_km,result.accel_oso_m_s2,
            result.target_image_gnomonic,orbit,metadata,
            result.reference_orbit_eci_km,result.metadata["reference_period_s"])
        np.testing.assert_allclose(other.image_gnomonic,result.image_gnomonic,
                                   atol=1e-12,equal_nan=True)
        np.testing.assert_array_equal(other.image_valid,result.image_valid)
        np.testing.assert_allclose(other.elevation_deg,result.elevation_deg,atol=1e-12)

    def test_all_pairs_and_neighbor_sizes(self):
        scenario,_,_=load_study_case(3,2.,outer_side_deg=10.)
        result=calculate(scenario,include_audit=False)
        size=spatial_size_metrics(result.r_eci_km,3)
        k=90
        r=result.r_eci_km[k]
        distances=np.linalg.norm(r[:,None,:]-r[None,:,:],axis=2)
        self.assertAlmostEqual(size["max_pair_km"][k],np.max(distances))
        self.assertAlmostEqual(result.min_pair_distance_km[k],
                               np.min(distances[np.triu_indices(9,1)]))
        self.assertLessEqual(size["min_neighbor_km"][k],size["max_neighbor_km"][k])
        self.assertEqual(size["corner_sides_km"].shape,(181,4))
        self.assertEqual(size["corner_diagonals_km"].shape,(181,2))


if __name__=="__main__":
    unittest.main()
