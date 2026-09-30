"""Native/offscreen window check, with optional SQUARE_SCREENSHOTS directory."""
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np
from PySide6.QtWidgets import QApplication

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from square_demo.ui import DemoWindow

app=QApplication([]);w=DemoWindow();w.show()

def ready(n,mode='independent_j2',audit=True):
    deadline=time.monotonic()+60
    while time.monotonic()<deadline:
        app.processEvents()
        if (w.result is not None and w.result.metadata['mode']==mode
            and w.result.r_eci_km.shape[1]==n*n
            and (not audit or mode!='independent_j2' or w.result.metadata['audit_status']=='complete')):return
        time.sleep(.02)
    raise AssertionError(w.status.text()+'; '+w.safety_badge.text())

def screenshot(name):
    if os.environ.get('SQUARE_SCREENSHOTS'):
        folder=Path(os.environ['SQUARE_SCREENSHOTS']);folder.mkdir(parents=True,exist_ok=True)
        app.processEvents();w.grab().save(str(folder/name))

ready(2)
assert [w.mode.itemText(k) for k in range(2)]==['Без управления','С управлением']
assert w.calculate_button.text()=='Рассчитать'
assert not any(hasattr(w,key) for key in ('variant','comparison','search_budget','search_button'))
for key in ('angular_side_deg','altitude_km','inclination_deg'):assert not w.fields[key].isEnabled()
assert 'без атмосферного сопротивления' in w.model_summary.text()
base=w.result.r_eci_km.copy()
for n in (2,3,10):
    if n!=2:
        w.grid_size.setValue(n);assert w.result is None
        w.calculate_button.click();ready(n)
    np.testing.assert_array_equal(w.result.r_eci_km[:,w.display.corners],base)
    assert w.result.metadata['quality']['longest_contiguous_duration_s']=={2:425.,3:185.,10:55.}[n]
    assert w.display.outer_quality['longest_contiguous_duration_s']==425.
    assert '425 с' in w.pass_badge.text()
    assert len(w.result.metadata['full_orbit_audit']['pair_minima'])==n*n*(n*n-1)//2
    for mode in range(4):
        w.view.setCurrentIndex(mode)
        for frame in (60,90,120):
            w.time_slider.setValue(frame);app.processEvents()
            local=w.view.currentData()=='local'
            xyz=w.display.local_km if local else w.result.r_eci_km
            velocities=w.display.local_velocity_km_s if local else w.display.eci_velocity_km_s
            chosen=w._space_artists['arrow_spacecraft_indices']
            np.testing.assert_allclose(w._space_artists['arrow_vectors_km_s'],velocities[frame,chosen])
            np.testing.assert_allclose(np.array(w._space_artists['scatter']._offsets3d).T,xyz[frame])
            np.testing.assert_allclose(w._grid_artists[w.proj][2].get_offsets(),np.rad2deg(np.arctan(w.result.image_gnomonic[frame])))
            np.testing.assert_allclose(w._grid_artists[w.yoz][2].get_offsets(),w.display.local_km[frame,:,1:])
            bounds=np.array([w.space.axes.get_xlim(),w.space.axes.get_ylim(),w.space.axes.get_zlim()])
            assert np.allclose(np.diff(bounds,axis=1),np.diff(bounds,axis=1)[0])
            for trace in w._space_artists['traces'].values():
                points=np.array(trace.get_data_3d()).T
                assert np.all(points>=bounds[:,0]) and np.all(points<=bounds[:,1])
        w.time_slider.setValue(90);app.processEvents()
        if mode in (0,1,2):screenshot(f'n{n}-view{mode}.png')
    w.view.setCurrentIndex(0)
    for canvas in (w.proj,w.yoz):
        canvas.draw();rgb=np.asarray(canvas.buffer_rgba())[:,:,:3].astype(int)
        assert np.count_nonzero(rgb.max(axis=2)-rgb.min(axis=2)>50)>40
    w.elements_button.setChecked(True);assert w.elements_table.isVisible()
    assert w.elements_table.rowCount()==n*n;w.elements_button.setChecked(False)
with TemporaryDirectory() as folder:
    path=Path(folder)/'elements.json';w.export_orbit_elements(path)
    assert len(json.loads(path.read_text())['elements_at_epoch'])==100
# Unsupported controlled grids must never reuse the last passive trajectory.
w.mode.setCurrentIndex(1);w.calculate_button.click()
assert w.result is None and 'не рассчитан' in w.status.text()
w.grid_size.setValue(2);w.calculate_button.click();ready(2,'legacy_replay')
assert w.result.t_s.shape==(241,)
assert w.result.accel_oso_m_s2.shape==(240,4,3)
assert not w.elements_button.isEnabled()
assert 'реконструкция' in w.model_summary.text()
w.time_slider.setValue(120);screenshot('controlled-n2.png')
# Return to J2, then prove changed observer clears old geometry until calculate.
w.mode.setCurrentIndex(0);w.calculate_button.click();ready(2)
w.fields['longitude_deg'].setValue(51.6);assert w.result is None
w.calculate_button.click();ready(2)
assert w.result.metadata['observer_longitude_deg']==51.6
w.time_slider.setValue(90);w.toggle_play()
start=time.monotonic()
while time.monotonic()-start<.5:app.processEvents();time.sleep(.01)
assert w.time_slider.value()>90;w.toggle_play()
w.close()
print('OK: native/offscreen J2 N=2/3/10, identical corners, CSV N=2, unsupported N>2, three panels, camera bounds, arrows, elements, animation')
