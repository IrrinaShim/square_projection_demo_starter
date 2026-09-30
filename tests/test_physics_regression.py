"""Physical invariants and the independent-deputy inverse derivative."""
import unittest
from unittest.mock import patch

import numpy as np

from orbit_study.drag_model import integrate
from orbit_study.passive_optimizer import (configuration, neighboring_seed,
    inverse_refine, geometry_at_epoch)
from square_demo.core import observer_eci, project
from square_demo.orbit_j2 import orbital_basis, state_to_elements, elements_to_state


class PhysicsRegressionTests(unittest.TestCase):
    def test_circular_two_body_one_revolution(self):
        cfg=configuration(2,20.);cfg['orbit']['j2']=0.
        initial=neighboring_seed(cfg,2)['states'][:1]
        mu=cfg['orbit']['mu_km3_s2'];radius=np.linalg.norm(initial[0,0])
        period=2*np.pi*np.sqrt(radius**3/mu)
        states=integrate(initial,np.linspace(0,period,1001),cfg,2.,with_drag=False)[:,0]
        r=np.linalg.norm(states[:,0],axis=1)
        energy=np.sum(states[:,1]**2,axis=1)/2-mu/r
        h=np.cross(states[:,0],states[:,1]);h/=np.linalg.norm(h,axis=1)[:,None]
        self.assertLess(np.max(abs(r-radius)),1e-7)
        self.assertLess(np.max(abs(energy/energy[0]-1)),1e-11)
        self.assertLess(np.max(np.linalg.norm(h-h[0],axis=1)),1e-12)
        self.assertLess(np.linalg.norm(states[-1,0]-states[0,0]),1e-6)
        for index in (0,250,500,750,1000):
            rebuilt=elements_to_state(state_to_elements(*states[index],mu),mu)
            np.testing.assert_allclose(rebuilt,states[index],atol=2e-6,rtol=0)

    def test_inverse_jacobian_matches_single_deputy_perturbations(self):
        cfg=configuration(2,20.);seed=neighboring_seed(cfg,2)
        chief,b,robs,rays,_,_=geometry_at_epoch(cfg,2)
        times=np.linspace(-60.,60.,11)
        states=seed['states'];nominal=integrate(states,times,cfg,2.)
        basis=orbital_basis(nominal[:,0])
        observer=observer_eci(times,0.,cfg['orbit']['earth_radius_km'],cfg['observer'])
        image=project(nominal[:,:,0],observer,basis,cfg['orbit']['earth_radius_km'])[0]
        designs=[];solve=np.linalg.lstsq
        def capture(a,rhs,**kwargs):
            designs.append(a.copy())
            return solve(a,rhs,**kwargs)
        with patch('orbit_study.passive_optimizer.np.linalg.lstsq',side_effect=capture):
            inverse_refine(seed,cfg,seed,half_window_s=60.)
        # Independently vary only spacecraft 1, using a smaller central
        # difference; its chief, image axis and other deputies remain fixed.
        scales=np.array([15.,.015,.015,.015])
        expected=[]
        for axis,delta in enumerate((.01,1e-5,1e-5,1e-5)):
            images=[]
            for sign in (-1,1):
                trial=states.copy()
                if axis==0:trial[1,0]+=sign*delta*rays[1]
                else:trial[1,1]+=sign*delta*b[:,axis-1]
                propagated=integrate(trial,times,cfg,2.)
                images.append(project(propagated[:,:,0],observer,basis,cfg['orbit']['earth_radius_km'])[0][:,1])
            expected.append(((images[1]-images[0])/(2*delta)).reshape(-1))
        expected=np.stack(expected,axis=1)*scales
        np.testing.assert_allclose(designs[0][:-4],expected,atol=2e-6,rtol=.003)


class FinalSelectionTests(unittest.TestCase):
    def test_failed_leader_tries_runner_up_before_older_archive(self):
        from copy import deepcopy
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from orbit_study.passive_optimizer import optimize
        cfg=configuration(2,20.)
        base=neighboring_seed(cfg,2)
        candidates=[]
        for source in ('archive','runner-up','leader'):
            c=deepcopy(base);c['parameters']['source']=source;candidates.append(c)
        archive,runner,leader=candidates
        archive['archive_audit']={'one_pass':{'accepted_intervals_square':[{'duration_s':10.}]}}
        durations={'archive':10.,'runner-up':12.,'leader':17.}
        def surveyed(c,config,step_s=None):
            self.assertEqual(step_s,1.)
            d=durations[c['parameters']['source']]
            return dict(score=(True,d,0.),duration_square_s=d,duration_resolution_2deg_s=0.,
                near_miss=0.,orbit_limits={},minimum_pair_sample_km=5.,violations=[])
        def verified(c,*args):
            name=c['parameters']['source']
            return dict(physical_violations=['full-orbit collision'] if name=='leader' else [],
                        one_pass={'best_square_duration_s':durations[name]})
        with TemporaryDirectory() as tmp, \
            patch('orbit_study.passive_optimizer.previous_version_seed',return_value=None), \
            patch('orbit_study.passive_optimizer.archived_seed',return_value=archive), \
            patch('orbit_study.passive_optimizer.neighboring_seed',return_value=runner), \
            patch('orbit_study.passive_optimizer.rotation_seed',return_value=leader), \
            patch('orbit_study.passive_optimizer.corner_seed_hcw',return_value=runner), \
            patch('orbit_study.passive_optimizer.neighboring_candidate',side_effect=lambda c,*args:c), \
            patch('orbit_study.passive_optimizer.survey',side_effect=surveyed), \
            patch('orbit_study.passive_optimizer.verify',side_effect=verified):
            result,path=optimize(cfg,2,budget=3,output_root=Path(tmp),resume=False)
        self.assertEqual(result['candidate']['source'],'runner-up')
        self.assertEqual(result['verification']['one_pass']['best_square_duration_s'],12.)


if __name__=='__main__':unittest.main()
