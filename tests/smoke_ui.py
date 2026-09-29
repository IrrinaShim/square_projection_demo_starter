"""QT_QPA_PLATFORM=offscreen .venv/bin/python tests/smoke_ui.py"""
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from square_demo.ui import DemoWindow

app=QApplication([])
w=DemoWindow();w.show()
assert [w.mode.itemText(i) for i in range(w.mode.count())]==["Без управления","С управлением"]
assert w.grid_size.value()==10 and w.angular_side.value()==20
assert w.result is not None and w.result.r_eci_km.shape==(181,100,3)
assert w.result.metadata['quality']['longest_contiguous_duration_s']==30
assert w.study_record['pair_count']==4950
assert 'выполнены частично' in w.status.text()
assert 'атмосферное сопротивление' in w.model_summary.text()
assert 'Минимум между КА сейчас' in w.safety_badge.text()
assert abs(w.result.rho_oso_km.mean(axis=1)).max()<1e-9
w.elements_button.click()
assert w.elements_table.isVisible() and w.elements_table.rowCount()==100
w.elements_button.click()
local=np.array(w.space.axes.get_xlim())
w.view.setCurrentIndex(1)
close=np.array(w.space.axes.get_xlim())
chief=w.result.r_eci_km[w.time_slider.value(),0,0]
assert abs(close.mean()-chief)<1e-8
w.view.setCurrentIndex(2)
assert np.diff(w.space.axes.get_xlim())[0]>10000
w.view.setCurrentIndex(0)
assert np.allclose(w.space.axes.get_xlim(),local)

for n,side in ((2,20),(3,20),(10,10),(10,30)):
    w.grid_size.blockSignals(True);w.angular_side.blockSignals(True)
    w.grid_size.setValue(n);w.angular_side.setValue(side)
    w.grid_size.blockSignals(False);w.angular_side.blockSignals(False)
    w.calculate_current()
    assert w.result is not None,(n,side,w.status.text())
    assert w.result.r_eci_km.shape==(181,n*n,3)
    assert w.scenario['angular_side_deg']==side
    assert w.study_record['pair_count']==n*n*(n*n-1)//2
    assert w.elements_table.rowCount()==n*n
    assert len(w.yoz.axes.collections)>0
    w.time_slider.setValue(0)
    assert 'Сетка сейчас' in w.observable_badge.text()
    w.time_slider.setValue(90)
    assert w.timeline.eligible.shape==(181,)

w.grid_size.blockSignals(True);w.angular_side.blockSignals(True)
w.grid_size.setValue(2);w.angular_side.setValue(20)
w.grid_size.blockSignals(False);w.angular_side.blockSignals(False);w.calculate_current()
w.time_slider.setValue(90);w.toggle_play()
loop=QEventLoop();QTimer.singleShot(450,loop.quit);loop.exec()
assert w.time_slider.value()>90
w.toggle_play()
with TemporaryDirectory() as tmp:
    p=Path(tmp)/'elements.json';w.export_orbit_elements(p)
    data=json.loads(p.read_text(encoding='utf-8'))
    assert len(data['elements_at_epoch'])==4
    assert data['model']==w.scenario['model_id']
    assert data['epoch_utc']=='2026-09-28T00:00:00Z'
    w.start_long_analysis(tmp)
    assert 'отклонён' in w.long_status.text()
w.mode.setCurrentIndex(1)
assert w.result is None and 'проверенного решения нет' in w.status.text()
w.mode.setCurrentIndex(0)
assert w.result is not None
print('OK: N=2/3/10, 10°/20°/30°, два режима, проекции, ползунок, анимация, камера, экспорт, отказ от старого долгосрочного J₂')
w.close()
