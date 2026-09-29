"""QT_QPA_PLATFORM=offscreen .venv/bin/python tests/smoke_ui.py"""
import json
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from square_demo.ui import DemoWindow

app=QApplication([])
w=DemoWindow();w.show()
def ready(n,side,timeout=30):
    start=time.perf_counter()
    while time.perf_counter()-start<timeout:
        app.processEvents()
        if w.result is not None and w.scenario['angular_side_deg']==side and w.result.r_eci_km.shape[1]==n*n:
            return
        time.sleep(.02)
    raise AssertionError((n,side,w.status.text()))

ready(10,20)
assert w.windowTitle().endswith('подбор v5')
assert w.calculate_button.text()=='Рассчитать'
assert not hasattr(w,'search_budget') and not hasattr(w,'search_button')
assert 'ffcf6dd1184f55b05c72' in w.status.text()
assert 'Проверка проекции через 1 с: 12 с' in w.status.text()
assert [w.mode.itemText(i) for i in range(w.mode.count())]==['Без управления','С управлением']
assert w.result.r_eci_km.shape==(181,100,3)
assert w.study_record['pair_count']==4950
assert w.study_record['horizon_s'][1]>5000
assert w.study_record['orbital_direction']['full_orbit']['ok']
assert w.study_record['orbital_direction']['full_orbit']['max_angle_deg']<5.
for scope in ('epoch','one_pass','full_orbit'):
    assert w.study_record['orbital_direction'][scope]['min_dot']>0
    assert w.study_record['orbital_direction'][scope]['max_angle_deg']<=5.
assert 'максимальный угол нормалей' in w.status.text().lower()
assert 'часть условий' in w.status.text()
assert 'ожидаемый' not in w.metrics.text().lower()
assert 'порог модели' not in w.metrics.text().lower()
assert '10 с' in w.pass_badge.text() and '12 с' in w.pass_badge.text() and '2°: 0 с' in w.pass_badge.text()
assert 'атмосферное сопротивление' in w.model_summary.text()
assert 'Минимум между КА сейчас' in w.safety_badge.text()
assert 'Внешняя сторона: 20.000°' in w.metrics.text()
assert abs(w.result.rho_oso_km.mean(axis=1)).max()<1e-9
frame=w.time_slider.value()
chosen=w._space_artists['arrow_spacecraft_indices']
local_vectors=w._space_artists['arrow_vectors_km_s']
expected=w.result.metadata['velocity_eci_km_s'][frame,chosen] @ w.result.metadata['local_basis_eci'][frame]
np.testing.assert_allclose(local_vectors,expected,atol=1e-12)
assert np.all(local_vectors @ local_vectors[0]>0)
with TemporaryDirectory() as tmp:
    with patch.object(QFileDialog,'getExistingDirectory',return_value=tmp): w.long_button.click()
    folders=list(Path(tmp).glob('N10_*'))
    assert len(folders)==1
    report=json.loads((folders[0]/'long_term_assessment.json').read_text(encoding='utf-8'))
    assert report['status']=='not_computed' and len(report['elements_at_epoch'])==100
w.mode.setCurrentIndex(1)
assert w.result is None and 'для N>2 рассчитанных данных управления нет' in w.status.text()
w.mode.setCurrentIndex(0);w.calculate_button.click();ready(10,20)
w.elements_button.click()
assert w.elements_table.isVisible() and w.elements_table.rowCount()==100
w.elements_button.click()
local=np.array(w.space.axes.get_xlim())
w.view.setCurrentIndex(1)
close=np.array(w.space.axes.get_xlim())
chief=w.result.r_eci_km[w.time_slider.value(),0,0]
assert abs(close.mean()-chief)<1e-8
assert w._space_artists['arrow_frame_index']==w.time_slider.value()
assert w._space_artists['trace_end_time_s']==w.result.t_s[w.time_slider.value()]
w.view.setCurrentIndex(2)
assert np.diff(w.space.axes.get_xlim())[0]>10000
w.view.setCurrentIndex(3)
fixed=np.array(w.space.axes.get_xlim())
w.time_slider.setValue(110)
assert np.allclose(w.space.axes.get_xlim(),fixed)
assert w._space_artists['arrow_frame_index']==110
assert w._space_artists['trace_end_time_s']==w.result.t_s[110]
w.view.setCurrentIndex(1)
assert not np.allclose(w.space.axes.get_xlim(),close)
w.view.setCurrentIndex(0)
assert np.allclose(w.space.axes.get_xlim(),local)
for n in (2,3):
    w.grid_size.setValue(n)
    assert w.result is None and 'Параметры изменены' in w.status.text()
    w.calculate_button.click()
    ready(n,20)
    assert w.study_record['pair_count']==n*n*(n*n-1)//2
    for scope in ('epoch','one_pass','full_orbit'):
        assert w.study_record['orbital_direction'][scope]['min_dot']>0
        assert w.study_record['orbital_direction'][scope]['max_angle_deg']<=5.
    assert w.elements_table.rowCount()==n*n
    w.time_slider.setValue(0)
    assert 'Наблюдаемый квадрат сейчас' in w.observable_badge.text()
    w.time_slider.setValue(90)
    assert w.timeline.eligible.shape==(181,)
    if n==2:
        assert '−55…25 с' not in w.pass_badge.text()  # ASCII minus is used by the numeric formatter.
        assert '-55…25 с (80 с)' in w.pass_badge.text()
        assert 'наблюдаемый квадрат сейчас: да' in w.observable_badge.text().lower()
        assert 'Полный виток: орбиты' in w.orbit_badge.text()
        original=w.study_record['full_orbit_ok']
        w.study_record['full_orbit_ok']=False
        w.study_record['full_orbit_violations']=['контрольная недопустимая орбита']
        w.update_frame()
        assert 'контрольная недопустимая орбита' in w.orbit_badge.text()
        assert '-55…25 с (80 с)' in w.pass_badge.text()
        assert w.timeline.eligible[90]
        w.study_record['full_orbit_ok']=original
        w.study_record['full_orbit_violations']=[]
        w.update_frame()
        w.time_slider.setValue(96)
        assert 'наблюдаемый квадрат сейчас: нет' in w.observable_badge.text().lower()
        assert '-55…25 с (80 с)' in w.pass_badge.text()
w.grid_size.setValue(10);w.calculate_button.click();ready(10,20)
w.time_slider.setValue(90);w.toggle_play()
start=time.perf_counter()
while time.perf_counter()-start<.5: app.processEvents();time.sleep(.01)
assert w.time_slider.value()>90
w.toggle_play()
with TemporaryDirectory() as tmp:
    path=Path(tmp)/'elements.json';w.export_orbit_elements(path)
    data=json.loads(path.read_text(encoding='utf-8'))
    assert len(data['elements_at_epoch'])==100
    assert data['epoch_utc']=='2026-09-28T00:00:00Z'
    with patch.object(QFileDialog,'getExistingDirectory',return_value=tmp):w.long_button.click()
    assert len(list(Path(tmp).glob('N*')))==1
w.angular_side.setValue(30)
assert w.result is None and 'Параметры изменены' in w.status.text()
w.angular_side.setValue(20);w.calculate_button.click();ready(10,20)
print('OK: N=2/3/10, 100 КА, два режима, ползунок, анимация, камера, отбрасывание старых траекторий, отдельный отчёт')
w.close()
