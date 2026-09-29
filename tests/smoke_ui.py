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
w=DemoWindow()
w.show()
assert w.mode.count()==2
assert [w.mode.itemText(i) for i in range(2)]==["Без управления","С управлением"]
assert not hasattr(w,"variant") and not hasattr(w,"comparison")
assert w.grid_size.value()==10 and w.angular_side.value()==20
assert w.angular_side.maximum()==30
assert w.result is None
assert "Сохранённый набор N=10 имеет сторону 10°" in w.status.text()
assert not w.play_button.isEnabled()
w.angular_side.setValue(10)
assert w.result is not None and w.result.r_eci_km.shape==(181,100,3)
assert w.scenario["angular_side_deg"]==10
assert w.study_record["parameters"]["side_deg"]==10
assert "нет проверенного решения" in w.status.text()
assert "Подходящий интервал пролёта: отсутствует" in w.observable_badge.text()
assert "Минимум между КА сейчас" in w.safety_badge.text()
assert "Порог: размеры КА + принятый запас" in w.safety_badge.text()
assert "соседние стороны" in w.geometry.text()
assert abs(w.result.rho_oso_km.mean(axis=1)).max()<1e-9
assert len(w.result.metadata["orbital_elements"])==100
assert w.elements_button.isEnabled()
w.elements_button.click()
assert w.elements_table.isVisible() and w.elements_table.rowCount()==100
w.elements_button.click()
local=np.array(w.space.axes.get_xlim())
w.view.setCurrentIndex(1)
close=np.array(w.space.axes.get_xlim())
chief=w.result.r_eci_km[w.time_slider.value(),0,0]
assert abs(close.mean()-chief)<1e-8
assert close[1]-close[0]<1000
w.view.setCurrentIndex(2)
wide=np.array(w.space.axes.get_xlim())
assert wide[1]-wide[0]>10000
w.view.setCurrentIndex(0)
assert np.allclose(w.space.axes.get_xlim(),local)

for n,side in ((2,20),(3,10),(10,10)):
    w.grid_size.setValue(n)
    w.angular_side.setValue(side)
    assert w.result is not None,(n,side,w.status.text())
    assert w.result.r_eci_km.shape==(181,n*n,3)
    assert w.scenario["angular_side_deg"]==side
    assert w.study_record["parameters"]["side_deg"]==side
    assert w.elements_table.rowCount()==n*n
    assert w.result.metadata["spatial_size_metrics"]["min_neighbor_km"].shape==(181,)
    before=np.asarray(w.proj.axes.lines[0].get_xydata()).copy()
    w.time_slider.setValue(0)
    assert "угол места" in w.observable_badge.text()
    after=np.asarray(w.proj.axes.lines[1].get_xydata()).copy()
    assert not np.allclose(before,after)
    assert w.timeline.eligible.shape==(181,)
    assert len(w.yoz.axes.collections)>0

w.grid_size.setValue(2);w.angular_side.setValue(20)
w.time_slider.setValue(90)
w.toggle_play()
loop=QEventLoop()
QTimer.singleShot(450,loop.quit)
loop.exec()
assert w.time_slider.value()>90
w.toggle_play()
with TemporaryDirectory() as tmp:
    p=Path(tmp)/"elements.json"
    w.export_orbit_elements(p)
    data=json.loads(p.read_text(encoding="utf-8"))
    assert len(data["elements_at_epoch"])==4
    assert data["epoch_utc"]=="2026-09-28T00:00:00Z"
    w.long_horizon.setValue(.2)
    w.start_long_analysis(tmp)
    loop=QEventLoop()
    poll=QTimer()
    poll.timeout.connect(lambda:loop.quit() if w.analysis_future is None else None)
    poll.start(100)
    QTimer.singleShot(30000,loop.quit)
    loop.exec()
    poll.stop()
    assert w.analysis_future is None
    assert "файлы:" in w.long_status.text(),w.long_status.text()
    folders=list(Path(tmp).glob("N2_*"))
    assert len(folders)==1
    assert {"case.json","passes.csv","pair_minima.csv","passes.png",
            "shape_quality.png","pair_minima.png"}<={p.name for p in folders[0].iterdir()}

w.angular_side.setValue(30)
assert w.result is None and "требуется новый подбор" in w.status.text()
w.angular_side.setValue(20)
assert w.result is not None
w.mode.setCurrentIndex(1)
assert w.result is None and "Историческая траектория не является" in w.status.text()
w.mode.setCurrentIndex(0)
assert w.result is not None
w.grid_size.setValue(10)
assert w.result is None and "сторону 10°" in w.status.text()
print("OK: два режима, N=2/3/10, стороны 10°/20°/30°, неподмена орбит, центр группы, камера, экспорт и отдельный расчёт")
w.close()
