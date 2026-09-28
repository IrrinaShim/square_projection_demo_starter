"""Короткая проверка окна: QT_QPA_PLATFORM=offscreen .venv/bin/python tests/smoke_ui.py"""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from square_demo.ui import DemoWindow

app = QApplication([])
window = DemoWindow()
window.show()
assert window.result.metadata["mode"] == "free_hcw_two_body"
assert window.result.metadata["variant"] == "fitted"
assert window.view.currentIndex() == 0
assert not hasattr(window,"control")
assert "период" in window.parameters.text()
assert "подобраны по проекции" in window.parameters.text()
window.initial_button.click()
assert window.initial_table.isVisible()
assert window.initial_table.item(0,1).text() != "—"
window.initial_button.click()
zenith = int(abs(window.result.t_s).argmin())
assert abs(window.result.image_gnomonic[zenith]-window.result.target_image_gnomonic).max() < 1e-12
window.time_slider.setValue(zenith)
assert "приемлем" in window.observable_badge.text()
assert "#d9f4e3" in window.observable_badge.styleSheet()
assert abs(window.result.metadata["observability"]["size_deviation_percent"][zenith]) < 1e-8
projected_at_zenith = np.asarray(window.proj.axes.lines[1].get_xydata()).copy()
yoz_at_zenith = np.asarray(window.yoz.axes.lines[1].get_xydata()).copy()
window.time_slider.setValue(0)
assert "неприемлем" in window.observable_badge.text()
assert "угол места" in window.observable_badge.text()
assert not np.allclose(window.proj.axes.lines[1].get_xydata(),projected_at_zenith)
assert not np.allclose(window.yoz.axes.lines[1].get_xydata(),yoz_at_zenith)
assert window.timeline.eligible[zenith] and not window.timeline.eligible[0]
assert window.result.metadata["quality"]["zenith_contiguous_duration_s"] > window.variants["trial"].metadata["quality"]["zenith_contiguous_duration_s"]
window.variant.setCurrentIndex(1)
assert window.result.metadata["variant"] == "trial"
assert "не подтверждена как безопасная" in window.safety_badge.text()
window.variant.setCurrentIndex(2)
assert window.result.metadata["variant"] == "ideal"
window.variant.setCurrentIndex(0)
window.fields["angular_side_deg"].setValue(20)
window.calculate_current()
assert window.result.metadata["angular_side_deg"] == 20
assert "20.000000°" in window.parameters.text()
window.time_slider.setValue(15)
assert window.time_slider.value() == 15
window.toggle_play()
QTimer.singleShot(600,app.quit)
app.exec()
assert window.timer.isActive() and window.time_slider.value() > 15
window.toggle_play()
window.mode.setCurrentIndex(1)
assert window.result.metadata["mode"] == "legacy_replay"
assert not window.fields["angular_side_deg"].isEnabled()
assert "исторические состояния CSV" in window.parameters.text()
window.initial_button.click()
assert window.initial_table.item(0,1).text() == "—"
window.initial_button.click()
window.view.setCurrentIndex(1)
window.time_slider.setValue(120)
window.toggle_play()
QTimer.singleShot(600,app.quit)
app.exec()
assert window.time_slider.value() > 120
window.toggle_play()
app.processEvents()
assert "Наблюдаемый квадрат" in window.observable_badge.text()
window.show()
window.mode.setCurrentIndex(2)
assert window.result.metadata["mode"] == "independent_j2"
assert "J₂" in window.parameters.text()
assert "полном витке" in window.safety_badge.text()
window.elements_button.click()
assert window.elements_table.isVisible(), (window.elements_button.isChecked(),
                                          window.elements_button.isVisible(),
                                          window.elements_table.isHidden())
assert window.elements_table.item(3,1).text() != ""
window.elements_button.click()
zenith = int(abs(window.result.t_s).argmin())
window.time_slider.setValue(zenith)
assert "приемлем" in window.observable_badge.text()
projected = np.asarray(window.proj.axes.lines[1].get_xydata()).copy()
window.time_slider.setValue(0)
assert not np.allclose(window.proj.axes.lines[1].get_xydata(),projected)
window.view.setCurrentIndex(0)
window.toggle_play()
QTimer.singleShot(330,app.quit)
app.exec()
assert window.time_slider.value() > 0
window.toggle_play()
window.close()
window = DemoWindow()
window.show()
window.mode.setCurrentIndex(2)
window.grid_size.setValue(3)
window.calculate_current()
assert window.result.r_eci_km.shape == (181,9,3)
assert "выполняется" in window.safety_badge.text()
assert len(window.proj.axes.collections) >= 1
with TemporaryDirectory() as temporary:
    path = Path(temporary)/"elements.json"
    window.export_orbit_elements(path)
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["elements_at_epoch"]) == 9
window.grid_size.setValue(10)
window.calculate_current()
assert window.result.r_eci_km.shape == (181,100,3)
assert "выполняется" in window.safety_badge.text()
assert window.elements_table.rowCount() == 100
window.time_slider.setValue(90)
assert "Вся сетка" in window.metrics.text()
poll = QTimer()
poll.timeout.connect(lambda: app.quit() if window.result.metadata["audit_status"] == "complete" else None)
poll.start(100)
QTimer.singleShot(12000,app.quit)
app.exec()
poll.stop()
assert window.result.metadata["audit_status"] == "complete"
assert window.result.metadata["full_orbit_audit"]["pair_count"] == 4950
print("OK: HCW, J₂ и CSV; обе 2D-картины, 3D, элементы, индикаторы, ползунок, анимация")
window.close()
