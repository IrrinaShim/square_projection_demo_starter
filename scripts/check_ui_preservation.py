"""Compare current calculations/display with frozen c910c70, without checkout."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from square_demo import providers
from square_demo.display import prepare_display
from square_demo.orbit_j2 import orbital_basis

BASE='c910c70c70caeabfa8b169c6413abcb9013dc38a'


def frozen_provider():
    name='_frozen_c910c70'
    package=types.ModuleType(name);package.__path__=[];sys.modules[name]=package
    for short in ('orbit_j2','grid','core','providers'):
        full=f'{name}.{short}';module=types.ModuleType(full)
        module.__file__=str(ROOT/'square_demo'/f'{short}.py');sys.modules[full]=module
        source=subprocess.check_output(['git','show',f'{BASE}:square_demo/{short}.py'],cwd=ROOT,text=True)
        exec(compile(source,f'{BASE}:{short}.py','exec'),module.__dict__)
    return module


def captured_calculation(provider,scenario):
    calls=[];integrate=provider.integrate
    def capture(initial,*args,**kwargs):
        if not calls:calls.append(initial.copy())
        return integrate(initial,*args,**kwargs)
    with patch.object(provider,'integrate',side_effect=capture):
        result=provider.calculate(scenario,include_audit=False)
    return result,calls[0] if calls else None


def verify_preservation():
    frozen=frozen_provider();cases=[];corners_ref=None;initial_ref=None
    protected=['square_demo/'+name+'.py' for name in ('core','grid','orbit_j2','providers')]
    protected+=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE,'scenarios','legacy/example_controlled'],cwd=ROOT,text=True).splitlines()
    hashes={}
    for path in protected:
        old=subprocess.check_output(['git','show',f'{BASE}:{path}'],cwd=ROOT)
        current=(ROOT/path).read_bytes()
        assert old==current,path+' changed'
        hashes[path]=hashlib.sha256(current).hexdigest()
    for n in (2,3,10):
        scenario=providers.load_scenario(providers.SCENARIOS['independent_j2']);scenario['grid_size']=n
        before,old_initial=captured_calculation(frozen,scenario)
        after,initial=captured_calculation(providers,scenario)
        np.testing.assert_array_equal(initial,old_initial)
        arrays={k:v.copy() for k,v in vars(after).items() if isinstance(v,np.ndarray)}
        for name,array in arrays.items():np.testing.assert_array_equal(array,getattr(before,name),err_msg=name)
        assert after.metadata['orbital_elements']==before.metadata['orbital_elements']
        assert after.metadata['quality']==before.metadata['quality']
        np.testing.assert_array_equal(after.metadata['observability']['eligible'],before.metadata['observability']['eligible'])
        shown=prepare_display(after)
        for name,array in arrays.items():np.testing.assert_array_equal(array,getattr(after,name),err_msg='display mutated '+name)
        indices=shown.corners
        if corners_ref is None:
            corners_ref=after.r_eci_km[:,indices].copy();initial_ref=initial[indices].copy()
        np.testing.assert_array_equal(initial[indices],initial_ref)
        np.testing.assert_array_equal(after.r_eci_km[:,indices],corners_ref)
        # Reconstruct the displayed centroid frame with the exact OLD model
        # states; this is validation only, never an extra GUI propagation.
        o=scenario['orbit']
        states=providers.integrate(initial,after.t_s,o['mu_km3_s2'],o['earth_radius_km'],o['j2'],scenario['integration']['max_step_s'])
        basis=orbital_basis(states[:,0])
        reconstructed=shown.center_eci_km[:,None]+np.einsum('tij,tsj->tsi',basis,shown.local_km)
        np.testing.assert_allclose(reconstructed,after.r_eci_km,rtol=0,atol=2e-10)
        np.testing.assert_allclose(shown.local_km.mean(axis=1),0,atol=1e-12)
        audit=providers.audit_independent_grid(scenario,after.metadata['orbital_elements'])
        closest=min(audit['full_orbit_audit']['pair_minima'].values(),key=lambda x:x['distance_km'])
        samples=[]
        for k in (60,90,120):
            samples.append(dict(t_s=float(after.t_s[k]),eci_corner_km=after.r_eci_km[k,indices].tolist(),
                local_corner_km=shown.local_km[k,indices].tolist(),
                projection_corner_deg=shown.image_deg[k,indices].tolist(),
                yoz_corner_km=shown.local_km[k,indices,1:].tolist()))
        cases.append(dict(N=n,spacecraft_count=n*n,side_deg=scenario['angular_side_deg'],altitude_km=o['altitude_km'],
            initial_states_max_change=0.,elements_identical=True,trajectory_max_change_km=0.,
            corner_initial_states_max_change_vs_N2=0.,corner_trajectory_max_change_vs_N2_km=0.,
            outer_duration_s=shown.outer_quality['longest_contiguous_duration_s'],
            grid_duration_before_s=before.metadata['quality']['longest_contiguous_duration_s'],
            grid_duration_after_s=after.metadata['quality']['longest_contiguous_duration_s'],
            closest_pair=closest,full_orbit_passes=audit['full_orbit_meets_illustrative_threshold'],
            display_reconstruction_max_error_km=float(np.max(abs(reconstructed-after.r_eci_km))),
            array_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()},samples=samples))
        print(f"N={n}: outer {cases[-1]['outer_duration_s']} s; grid {cases[-1]['grid_duration_after_s']} s; exact orbit preservation",flush=True)
    # CSV remains its original independent historical mode.
    scenario=providers.load_scenario(providers.SCENARIOS['legacy_replay'])
    before=frozen.calculate(scenario);after=providers.calculate(scenario)
    for name,value in vars(before).items():
        if isinstance(value,np.ndarray):np.testing.assert_array_equal(value,getattr(after,name))
    prepare_display(after)
    return dict(baseline_commit=BASE,protected_file_sha256=hashes,cases=cases,
                controlled_csv_arrays_identical=True,
                controlled_csv_duration_s=after.metadata['quality']['longest_contiguous_duration_s'])


if __name__=='__main__':
    report=verify_preservation()
    folder=ROOT/'docs'/'ui_preview_validation';folder.mkdir(exist_ok=True)
    (folder/'preservation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
