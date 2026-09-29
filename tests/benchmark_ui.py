"""Реплицируемый Qt/Matplotlib бенчмарк; запускать на целевой машине."""
import statistics,time,sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from square_demo.ui import DemoWindow
app=QApplication([])
t0=time.perf_counter(); window=DemoWindow();window.show()
while window.result is None and time.perf_counter()-t0<20:
    app.processEvents();time.sleep(.01)
if window.result is None: raise RuntimeError(window.status.text())
print(f'saved_startup_s={time.perf_counter()-t0:.3f}')
for mode in range(window.view.count()):
    window.view.setCurrentIndex(mode);app.processEvents()
    elapsed=[]
    for index in (20,40,60,80,100,120,140,160):
        start=time.perf_counter();window.time_slider.setValue(index);app.processEvents()
        elapsed.append(time.perf_counter()-start)
    print(f'frame_mode_{mode}_median_s={statistics.median(elapsed):.3f} max_s={max(elapsed):.3f}')
start=time.perf_counter();window.angular_side.setValue(19);app.processEvents()
print(f'field_event_s={time.perf_counter()-start:.3f} old_result_cleared={window.result is None}')
window.close()
