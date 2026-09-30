import unittest
import numpy as np
from square_demo.providers import calculate,load_scenario,SCENARIOS
from square_demo.display import prepare_display

class DisplayTests(unittest.TestCase):
    def test_centering_and_outer_predicate_do_not_mutate_calculation(self):
        reference=None
        for n in (2,3,10):
            scenario=load_scenario(SCENARIOS['independent_j2']);scenario['grid_size']=n
            r=calculate(scenario,include_audit=False)
            original_r=r.r_eci_km.copy();original_rho=r.rho_oso_km.copy()
            original_mask=r.metadata['observability']['eligible'].copy()
            d=prepare_display(r)
            np.testing.assert_array_equal(r.r_eci_km,original_r)
            np.testing.assert_array_equal(r.rho_oso_km,original_rho)
            np.testing.assert_array_equal(r.metadata['observability']['eligible'],original_mask)
            np.testing.assert_allclose(d.local_km.mean(axis=1),0,atol=1e-12)
            self.assertEqual(d.outer_quality['longest_contiguous_duration_s'],425.)
            self.assertEqual(r.metadata['quality']['longest_contiguous_duration_s'],{2:425.,3:185.,10:55.}[n])
            if reference is None:reference=r.r_eci_km[:,d.corners].copy()
            np.testing.assert_array_equal(r.r_eci_km[:,d.corners],reference)
            for k in (60,90,120):
                finite=(d.local_km[k+1]-d.local_km[k-1])/(r.t_s[k+1]-r.t_s[k-1])
                np.testing.assert_allclose(d.local_velocity_km_s[k],finite,atol=1e-14)
                p=r.r_eci_km[k,d.indices]
                distances=np.r_[np.linalg.norm(np.diff(p,axis=0),axis=2).ravel(),np.linalg.norm(np.diff(p,axis=1),axis=2).ravel()]
                self.assertAlmostEqual(d.spatial['min_neighbor_km'][k],distances.min())
                self.assertAlmostEqual(d.spatial['max_neighbor_km'][k],distances.max())

if __name__=='__main__':unittest.main()
