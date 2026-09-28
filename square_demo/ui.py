"""Окно демонстрации. Здесь нет уравнений орбит или проекции."""
from __future__ import annotations

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QFormLayout,
                               QFileDialog, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                               QPushButton, QSlider, QSplitter, QTableWidget, QToolButton,
                               QTableWidgetItem, QVBoxLayout, QWidget, QSpinBox)

from .providers import (SCENARIOS, audit_independent_grid, calculate,
                        calculate_free_variants, load_scenario)


COLORS = ["#223b85", "#d63d35", "#15925b", "#b87607", "#9245a7", "#00a1ac"]


class Plot(FigureCanvasQTAgg):
    def __init__(self, projection=None):
        self.figure = Figure(figsize=(5, 4), constrained_layout=projection != "3d")
        self.axes = self.figure.add_subplot(111, projection=projection)
        if projection == "3d":
            self.figure.subplots_adjust(left=0.02,right=0.98,bottom=0.03,top=0.92)
        super().__init__(self.figure)
        self.setMinimumWidth(320)


class ObservableTimeline(QWidget):
    """Подсветка рассчитанных допустимых узлов под ползунком времени."""
    def __init__(self):
        super().__init__()
        self.setFixedHeight(14)
        self.times = None
        self.eligible = None
        self.index = 0

    def set_result(self, times, eligible):
        self.times = times
        self.eligible = eligible
        self.update()

    def set_index(self, index):
        self.index = index
        self.update()

    def paintEvent(self, event):
        if self.times is None:
            return
        painter = QPainter(self)
        painter.setPen(Qt.PenStyle.NoPen)
        left, width = 5, max(1,self.width()-10)
        painter.fillRect(left,4,width,6,QColor("#d4d8de"))
        span = self.times[-1]-self.times[0]
        for k,good in enumerate(self.eligible):
            if good:
                start = self.times[0] if k == 0 else (self.times[k-1]+self.times[k])/2
                end = self.times[-1] if k == len(self.times)-1 else (self.times[k]+self.times[k+1])/2
                x0 = left+round(width*(start-self.times[0])/span)
                x1 = left+round(width*(end-self.times[0])/span)
                painter.fillRect(x0,4,max(1,x1-x0),6,QColor("#259b59"))
        cursor = left+round(width*(self.times[self.index]-self.times[0])/span)
        painter.fillRect(cursor-1,1,2,12,QColor("#263238"))


class DemoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Наблюдаемая сетка КА")
        self.resize(1530, 900)
        self.result = None
        self.scenario = None
        self.variants = {}
        self.audit_executor = ThreadPoolExecutor(max_workers=1)
        self.audit_future = None
        self.audit_timer = QTimer(self)
        self.audit_timer.setInterval(120)
        self.audit_timer.timeout.connect(self._poll_audit)
        self.timer = QTimer(self)
        self.timer.setInterval(110)
        self.timer.timeout.connect(self.advance)
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        top = QHBoxLayout()
        root.addLayout(top)
        self.mode = QComboBox()
        self.mode.addItem("Свободный пролёт HCW", "free_hcw_two_body")
        self.mode.addItem("Историческая управляемая траектория", "legacy_replay")
        self.mode.addItem("Независимые орбиты J₂", "independent_j2")
        self.mode.currentIndexChanged.connect(self.reload_scenario)
        top.addWidget(QLabel("Режим:"))
        top.addWidget(self.mode, 2)
        self.variant = QComboBox()
        self.variant.addItem("Подобранное свободное", "fitted")
        self.variant.addItem("Пробное свободное", "trial")
        self.variant.addItem("Удерживаемый квадрат ОСО", "ideal")
        self.variant.currentIndexChanged.connect(self.select_variant)
        top.addWidget(self.variant, 2)
        self.view = QComboBox()
        self.view.addItems(["3D: локальная ОСО", "3D: общая инерциальная система"])
        self.view.currentIndexChanged.connect(self.update_frame)
        top.addWidget(self.view, 2)
        self.reload_button = QPushButton("Перезагрузить JSON")
        self.reload_button.clicked.connect(self.reload_scenario)
        top.addWidget(self.reload_button)

        settings = QGroupBox("Параметры сценария")
        settings_layout = QHBoxLayout(settings)
        root.addWidget(settings)
        self.fields = {}
        specs = [
            ("angular_side_deg", "Угловая сторона, °", 1, 20, 12),
            ("altitude_km", "Высота, км", 1, 100000, 2),
            ("inclination_deg", "Наклонение, °", 0.01, 179.99, 3),
            ("latitude_deg", "Широта, °", -90, 90, 4),
            ("longitude_deg", "Долгота, °", -180, 180, 4),
        ]
        for key, label, lo, hi, decimals in specs:
            column = QFormLayout()
            box = QDoubleSpinBox()
            box.setRange(lo, hi)
            box.setDecimals(decimals)
            box.setSingleStep(0.1 if key.endswith("deg") else 10)
            column.addRow(label, box)
            settings_layout.addLayout(column)
            self.fields[key] = box
        grid_column = QFormLayout()
        self.grid_size = QSpinBox()
        self.grid_size.setRange(2,10)
        grid_column.addRow("КА на стороне N",self.grid_size)
        settings_layout.addLayout(grid_column)
        self.recalculate = QPushButton("Пересчитать")
        self.recalculate.clicked.connect(self.calculate_current)
        settings_layout.addWidget(self.recalculate)

        parameter_group = QGroupBox("Параметры расчёта и выбор начального состояния")
        parameter_layout = QVBoxLayout(parameter_group)
        self.parameters = QLabel()
        self.parameters.setWordWrap(True)
        parameter_layout.addWidget(self.parameters)
        self.initial_button = QToolButton()
        self.initial_button.setText("Начальные дальности, положения и скорости ▸")
        self.initial_button.setCheckable(True)
        parameter_layout.addWidget(self.initial_button)
        self.initial_table = QTableWidget(3,8)
        self.initial_table.setHorizontalHeaderLabels([
            "КА", "λ, км", "x₀, км", "y₀, км", "z₀, км",
            "vx₀, м/с", "vy₀, м/с", "vz₀, м/с"])
        self.initial_table.verticalHeader().setVisible(False)
        self.initial_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.initial_table.setFixedHeight(122)
        self.initial_table.setVisible(False)
        self.initial_button.toggled.connect(self._toggle_initial_table)
        parameter_layout.addWidget(self.initial_table)
        self.elements_button = QToolButton()
        self.elements_button.setText("Оскулирующие элементы всех КА на общей эпохе ▸")
        self.elements_button.setCheckable(True)
        self.elements_button.setVisible(False)
        parameter_layout.addWidget(self.elements_button)
        self.elements_table = QTableWidget(4,7)
        self.elements_table.setHorizontalHeaderLabels([
            "КА", "a, км", "e", "i, °", "Ω, °", "ω, °", "ν, °"])
        self.elements_table.verticalHeader().setVisible(False)
        self.elements_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.elements_table.setFixedHeight(150)
        self.elements_table.setVisible(False)
        self.elements_button.toggled.connect(self._toggle_elements_table)
        parameter_layout.addWidget(self.elements_table)
        self.export_elements_button = QPushButton("Сохранить элементы JSON")
        self.export_elements_button.clicked.connect(self._choose_elements_path)
        self.export_elements_button.setVisible(False)
        parameter_layout.addWidget(self.export_elements_button)
        root.addWidget(parameter_group)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.proj = Plot()
        self.yoz = Plot()
        self.space = Plot("3d")
        splitter.addWidget(self.proj)
        splitter.addWidget(self.yoz)
        splitter.addWidget(self.space)
        splitter.setSizes([500, 500, 500])
        root.addWidget(splitter, 1)

        transport = QHBoxLayout()
        root.addLayout(transport)
        self.start_button = QPushButton("В начало")
        self.start_button.clicked.connect(lambda: self.time_slider.setValue(0))
        self.play_button = QPushButton("Пуск")
        self.play_button.clicked.connect(self.toggle_play)
        self.speed = QComboBox()
        for label, value in (("0,5×", 0.5), ("1×", 1), ("2×", 2), ("5×", 5)):
            self.speed.addItem(label, value)
        self.speed.setCurrentIndex(1)
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.valueChanged.connect(self.update_frame)
        self.timeline = ObservableTimeline()
        self.timeline.setToolTip("Зелёные участки: выполнены критерии формы и видимости")
        self.time_label = QLabel("t = — с")
        transport.addWidget(self.start_button)
        transport.addWidget(self.play_button)
        time_column = QVBoxLayout()
        time_column.setSpacing(0)
        time_column.addWidget(self.time_slider)
        time_column.addWidget(self.timeline)
        transport.addLayout(time_column,1)
        transport.addWidget(self.time_label)
        transport.addWidget(QLabel("Скорость:"))
        transport.addWidget(self.speed)

        indicators = QHBoxLayout()
        self.observable_badge = QLabel()
        self.observable_badge.setWordWrap(True)
        self.safety_badge = QLabel()
        self.safety_badge.setWordWrap(True)
        indicators.addWidget(self.observable_badge,3)
        indicators.addWidget(self.safety_badge,2)
        root.addLayout(indicators)
        self.metrics = QLabel()
        self.metrics.setWordWrap(True)
        self.metrics.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.metrics)
        self.geometry = QLabel()
        self.geometry.setWordWrap(True)
        root.addWidget(self.geometry)
        self.comparison = QTableWidget(3,7)
        self.comparison.setHorizontalHeaderLabels([
            "Вариант", "Допустимо, с", "max εb", "max εd", "max εγ, °", "min h, °", "min d, км"])
        self.comparison.setFixedHeight(113)
        self.comparison.verticalHeader().setVisible(False)
        self.comparison.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.comparison.setColumnWidth(0,180)
        for col in range(1,7):
            self.comparison.setColumnWidth(col,100)
        root.addWidget(self.comparison)
        self.verdict = QLabel()
        root.addWidget(self.verdict)
        self.info = QLabel()
        self.info.setWordWrap(True)
        root.addWidget(self.info)
        self.reload_scenario()

    def reload_scenario(self):
        self.timer.stop()
        self.play_button.setText("Пуск")
        try:
            self.scenario = load_scenario(SCENARIOS[self.mode.currentData()])
            s = self.scenario
            self.fields["angular_side_deg"].setValue(float(s.get("angular_side_deg", 1.8183029644518331)))
            orbit = s.get("orbit")
            if orbit is None:
                # Поля исторического режима показывают исходные параметры CSV.
                import json
                from pathlib import Path
                path = Path(s["_path"]).parent / s["data_directory"] / "config.json"
                with path.open(encoding="utf-8") as f:
                    old = json.load(f)["orbit"]
                orbit = {"altitude_km": old["h_km"], "inclination_deg": old["i_deg"]}
            self.fields["altitude_km"].setValue(float(orbit["altitude_km"]))
            self.fields["inclination_deg"].setValue(float(orbit["inclination_deg"]))
            for key in ("latitude_deg", "longitude_deg"):
                self.fields[key].setValue(float(s["observer"][key]))
            free = s["mode"] == "free_hcw_two_body"
            j2 = s["mode"] == "independent_j2"
            self.grid_size.setValue(int(s.get("grid_size",2)))
            self.grid_size.setEnabled(j2)
            self.variant.setEnabled(free)
            self.comparison.setVisible(free)
            self.verdict.setVisible(free)
            self.elements_button.setVisible(j2)
            self.elements_table.setVisible(j2 and self.elements_button.isChecked())
            self.export_elements_button.setVisible(j2)
            self.fields["angular_side_deg"].setEnabled(free)
            self.fields["altitude_km"].setEnabled(free)
            self.fields["inclination_deg"].setEnabled(free)
            self.calculate_current()
        except Exception as exc:
            self._error(exc)

    def calculate_current(self):
        if self.scenario is None:
            return
        try:
            s = deepcopy(self.scenario)
            if s["mode"] == "free_hcw_two_body":
                s["angular_side_deg"] = self.fields["angular_side_deg"].value()
                s["orbit"]["altitude_km"] = self.fields["altitude_km"].value()
                s["orbit"]["inclination_deg"] = self.fields["inclination_deg"].value()
            elif s["mode"] == "independent_j2":
                s["grid_size"] = self.grid_size.value()
            for key in ("latitude_deg", "longitude_deg"):
                s["observer"][key] = self.fields[key].value()
            if s["mode"] == "free_hcw_two_body":
                variants = calculate_free_variants(s)
                result = variants[self.variant.currentData()]
            else:
                variants = {}
                result = calculate(s,include_audit=(s["mode"] != "independent_j2"
                    or s["grid_size"] == 2))
        except Exception as exc:
            self._error(exc)
            return
        self.timer.stop()
        self.play_button.setText("Пуск")
        self.result = result
        self.variants = variants
        if self.audit_future is not None and not self.audit_future.done():
            self.audit_future.cancel()
        if result.metadata.get("audit_status") == "pending":
            self.audit_future = self.audit_executor.submit(
                audit_independent_grid,deepcopy(s),deepcopy(result.metadata["orbital_elements"]))
            self.audit_timer.start()
        else:
            self.audit_future = None
            self.audit_timer.stop()
        self.time_slider.blockSignals(True)
        self.time_slider.setRange(0, len(result.t_s)-1)
        zenith_idx = int(np.argmin(abs(result.t_s-result.metadata["zenith_s"])))
        self.time_slider.setValue(zenith_idx)
        self.time_slider.blockSignals(False)
        self.timeline.set_result(result.t_s,result.metadata["observability"]["eligible"])
        self._update_parameters()
        self._draw_comparison()
        self.update_frame()

    def _poll_audit(self):
        if self.audit_future is None or not self.audit_future.done():
            return
        future = self.audit_future
        self.audit_future = None
        self.audit_timer.stop()
        if future.cancelled():
            return
        try:
            self.result.metadata.update(future.result())
        except Exception as exc:
            self.result.metadata["audit_status"] = "failed"
            self.result.metadata["audit_error"] = str(exc)
        self.update_frame()

    def select_variant(self):
        if not self.variants or self.mode.currentData() != "free_hcw_two_body":
            return
        self.result = self.variants[self.variant.currentData()]
        self.timeline.set_result(self.result.t_s,self.result.metadata["observability"]["eligible"])
        self._update_parameters()
        self._draw_comparison()
        self.update_frame()

    def _toggle_initial_table(self, expanded):
        self.initial_table.setVisible(expanded)
        self.initial_button.setText("Начальные дальности, положения и скорости ▾" if expanded
                                    else "Начальные дальности, положения и скорости ▸")

    def _toggle_elements_table(self, expanded):
        self.elements_table.setVisible(expanded and self.mode.currentData() == "independent_j2")
        self.elements_button.setText("Оскулирующие элементы всех КА на общей эпохе ▾" if expanded
                                     else "Оскулирующие элементы всех КА на общей эпохе ▸")

    def _update_parameters(self):
        result = self.result
        meta = result.metadata
        model = meta["source"]
        epoch_text = (f" Общая эпоха {meta['epoch_utc']}; J₂={self.scenario['orbit']['j2']:.9f}; "
                      f"сетка {meta['grid_size']}×{meta['grid_size']} ({meta['grid_size']**2} КА)."
                      if meta["mode"] == "independent_j2" else "")
        self.parameters.setText(
            f"Модель: {model}. Высота {meta['reference_altitude_km']:.1f} км; "
            f"наклонение {meta['inclination_deg']:.2f}°; номинальный период {meta['reference_period_s']:.1f} с. "
            f"Наблюдатель: {meta['observer_latitude_deg']:.4f}° широты, "
            f"{meta['observer_longitude_deg']:.4f}° долготы; заданная угловая сторона "
            f"{meta['angular_side_deg']:.6f}°; рабочий интервал {result.t_s[0]:g}…{result.t_s[-1]:g} с. "
            f"Начальное состояние: {meta['initial_choice']} (t={meta['initial_epoch_s']:g} с).{epoch_text}")
        positions = meta["initial_rho_oso_km"]
        velocities = meta["initial_vel_oso_m_s"]
        lambdas = meta["initial_lambda_km"]
        self.initial_table.setRowCount(len(positions))
        for row in range(len(positions)):
            values = [str(row+1), "—" if lambdas is None else f"{lambdas[row]:.3f}"]
            values += [f"{x:.3f}" for x in positions[row]]
            values += [f"{x:.3f}" for x in velocities[row]]
            for col,value in enumerate(values):
                self.initial_table.setItem(row,col,QTableWidgetItem(value))
        if meta["mode"] == "independent_j2":
            self.elements_table.setRowCount(len(meta["orbital_elements"]))
            for row,entry in enumerate(meta["orbital_elements"]):
                values = [str(row),f"{entry['a_km']:.6f}",f"{entry['e']:.9f}",
                          f"{entry['inclination_deg']:.6f}",f"{entry['raan_deg']:.6f}",
                          f"{entry['arg_perigee_deg']:.6f}",f"{entry['true_anomaly_deg']:.6f}"]
                for col,value in enumerate(values):
                    self.elements_table.setItem(row,col,QTableWidgetItem(value))

    def _choose_elements_path(self):
        if self.result is None or self.result.metadata.get("mode") != "independent_j2":
            return
        size = self.result.metadata["grid_size"]
        path,_ = QFileDialog.getSaveFileName(self,"Сохранить оскулирующие элементы",
            f"grid_n{size}_elements.json","JSON (*.json)")
        if path:
            try:
                self.export_orbit_elements(path)
            except OSError as exc:
                self._error(exc)

    def export_orbit_elements(self,path):
        meta = self.result.metadata
        payload = {"epoch_utc":meta["epoch_utc"],"grid_size":meta["grid_size"],
                   "index_map":meta["grid_indices"],"model":"independent_j2",
                   "elements_at_epoch":meta["orbital_elements"],
                   "units":{"a":"km","angles":"deg","e":"dimensionless"}}
        Path(path).write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    def _draw_comparison(self):
        if not self.variants:
            return
        for row,(name,title) in enumerate((("ideal","Удерживаемый ОСО"),
                                           ("trial","Пробное свободное"),
                                           ("fitted","Подобранное свободное"))):
            quality = self.variants[name].metadata["quality"]
            values = (title, f"{quality['zenith_contiguous_duration_s']:.0f}",
                      f"{quality['max_side_error']:.3f}",f"{quality['max_diagonal_error']:.3f}",
                      f"{quality['max_angle_error_deg']:.2f}",f"{quality['min_elevation_deg']:.1f}",
                      f"{quality['min_pair_distance_km']:.2f}")
            for col,value in enumerate(values):
                self.comparison.setItem(row,col,QTableWidgetItem(value))
        fitted = self.variants["fitted"].metadata["quality"]["zenith_contiguous_duration_s"]
        trial = self.variants["trial"].metadata["quality"]["zenith_contiguous_duration_s"]
        if fitted > trial:
            self.verdict.setText(f"Подбор увеличил непрерывную допустимую длительность вокруг зенита: {trial:.0f} → {fitted:.0f} с. Максимальные ошибки в таблице относятся ко всему окну.")
        else:
            self.verdict.setText(f"Подбор не улучшил непрерывную допустимую длительность вокруг зенита: пробное {trial:.0f} с, подобранное {fitted:.0f} с.")

    def _error(self, exc):
        QMessageBox.warning(self, "Ошибка входных данных", str(exc))

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Пуск")
        else:
            if self.time_slider.value() == self.time_slider.maximum():
                self.time_slider.setValue(0)
            self.timer.start()
            self.play_button.setText("Пауза")

    def advance(self):
        if self.result is None:
            return
        current = self.time_slider.value()
        t = self.result.t_s
        target = t[current] + self.timer.interval()/1000*self.speed.currentData()*10
        next_index = int(np.searchsorted(t, target, side="left"))
        self.time_slider.setValue(min(max(current+1, next_index), len(t)-1))
        if self.time_slider.value() == len(t)-1:
            self.timer.stop()
            self.play_button.setText("Пуск")

    def update_frame(self, *_):
        result = self.result
        if result is None:
            return
        k = self.time_slider.value()
        self.time_label.setText(f"t = {result.t_s[k]:.1f} с")
        self.timeline.set_index(k)
        self._draw_projection(k)
        self._draw_yoz(k)
        self._draw_space(k)
        errors = result.shape_errors[k]
        err = " / ".join("—" if not np.isfinite(x) else f"{x:.4f}" for x in errors[:2])
        ang = "—" if not np.isfinite(errors[2]) else f"{np.rad2deg(errors[2]):.2f}°"
        observation = result.metadata["observability"]
        min_elev = observation["min_elevation_deg"][k]
        good = bool(observation["eligible"][k])
        duration = result.metadata["quality"]["zenith_contiguous_duration_s"]
        subject = "Наблюдаемая сетка" if result.metadata.get("grid_size",2)>2 else "Наблюдаемый квадрат"
        if result.metadata.get("grid_size",2)>2:
            quality = result.metadata["quality"]
            longest = quality["longest_interval_s"]
            interval_text = (f"На пролёте допустимый интервал "
                f"{longest[0]:g}…{longest[1]:g} с, "
                f"длительность {quality['longest_contiguous_duration_s']:.0f} с."
                if quality["has_positive_duration_interval"] else
                "На пролёте связного допустимого интервала положительной длительности нет.")
        else:
            interval_text = f"Связный допустимый интервал вокруг зенита: {duration:.0f} с."
        if good:
            self.observable_badge.setText(
                f"● {subject} {'приемлема' if result.metadata.get('grid_size',2)>2 else 'приемлем'}. "
                +interval_text)
            self.observable_badge.setStyleSheet("background:#d9f4e3;color:#145b32;padding:5px;")
        else:
            self.observable_badge.setText(
                f"● {subject} {'неприемлема' if result.metadata.get('grid_size',2)>2 else 'неприемлем'}: "
                f"{observation['reasons'][k]}. {interval_text}")
            self.observable_badge.setStyleSheet("background:#ffe3df;color:#8b241d;padding:5px;")
        threshold = result.metadata["illustrative_min_separation_km"]
        safe_now = bool(observation["safety_illustrative_ok"][k])
        safe_window = bool(result.metadata["separation_meets_illustrative_threshold"])
        trial_full_orbit_warning = result.metadata.get("variant") == "trial"
        if result.metadata.get("audit_status") == "pending":
            safe_window = False
            self.safety_badge.setText(
                f"● Проверка всех {len(result.r_eci_km[0])*(len(result.r_eci_km[0])-1)//2} пар "
                f"на полном витке ещё выполняется. Сейчас min "
                f"{result.min_pair_distance_km[k]:.3f} км; "
                f"min в показанном окне {np.min(result.min_pair_distance_km):.3f} км. "
                "Вывод о сближениях за виток пока не готов.")
        elif result.metadata.get("audit_status") == "failed":
            safe_window = False
            self.safety_badge.setText("● Проверка полного витка не завершена: "
                                      +result.metadata.get("audit_error","ошибка"))
        elif "full_orbit_audit" in result.metadata:
            audit = result.metadata["full_orbit_audit"]
            closest = min(audit["pair_minima"].values(),key=lambda event:event["distance_km"])
            safe_window = result.metadata["full_orbit_meets_illustrative_threshold"]
            self.safety_badge.setText(
                f"{'● На полном витке ниже порога не найдено' if safe_window else '● Опасное сближение на полном витке'}: "
                f"min {closest['distance_km']:.3f} км, пара {closest['pair']}, "
                f"t={closest['t_s']:.1f} с, {closest['latitude_deg']:.1f}° / "
                f"{closest['longitude_deg']:.1f}°; сейчас {result.min_pair_distance_km[k]:.3f} км. "
                f"Иллюстративный порог {threshold:g} км; "
                f"опасных локальных минимумов: {len(audit['dangerous_events'])}.")
        elif trial_full_orbit_warning:
            safe_window = False
            self.safety_badge.setText(
                f"● Пробная конфигурация не подтверждена как безопасная: "
                f"на отдельном двухтельном витке пары 0–3 и 1–2 сближаются "
                f"примерно до 0,096 км около T/4. "
                f"Min в показываемом окне {np.min(result.min_pair_distance_km):.3f} км; "
                f"иллюстративный порог {threshold:g} км.")
        else:
            self.safety_badge.setText(
                f"{ '● Расстояния выше ориентира' if safe_now and safe_window else '● Есть сближение ниже ориентира' }: "
                f"сейчас {result.min_pair_distance_km[k]:.3f} км; "
                f"min за окно {np.min(result.min_pair_distance_km):.3f} км; "
                f"иллюстративный порог {threshold:g} км.")
        self.safety_badge.setStyleSheet(
            "background:#e7edf8;color:#1c3764;padding:5px;" if safe_now and safe_window
            else "background:#fff0d9;color:#805015;padding:5px;")
        size_dev = observation["size_deviation_percent"][k]
        size_text = "—" if not np.isfinite(size_dev) else f"{size_dev:+.2f}%"
        self.metrics.setText(
            f"εb / εd = {err}; εγ = {ang}; min угол места = {min_elev:.2f}°; "
            f"отклонение среднего размера от tan(θ) = {size_text} "
            f"(отдельный показатель, не входит в допуски формы).")
        if "grid_metrics" in result.metadata:
            gm = result.metadata["grid_metrics"]
            self.metrics.setText(self.metrics.text()+
                f" Вся сетка: отклонение точки {gm['point_error_steps'][k]:.3f} шага; "
                f"неравномерность шага {gm['neighbor_step_error'][k]:.3f}; "
                f"max εb ячейки {gm['max_cell_side_error'][k]:.3f}, "
                f"max εd ячейки {gm['max_cell_diagonal_error'][k]:.3f}, "
                f"max εγ ячейки {gm['max_cell_angle_error_deg'][k]:.2f}°; "
                f"min истинный угол соседей {gm['min_true_neighbor_angle_deg'][k]:.3f}°.")
        sides = ", ".join(f"{v:.4f}" for v in result.square_sides_gnomonic[k])
        diagonals = ", ".join(f"{v:.4f}" for v in result.square_diagonals_gnomonic[k])
        angles = ", ".join(f"{np.rad2deg(v):.1f}" for v in result.square_angles_rad[k])
        self.geometry.setText(f"Фактические стороны b₀…b₃ = [{sides}] и диагонали d₁,d₂ = [{diagonals}] "
                              f"в безразмерной картинной плоскости; углы γ₀…γ₃ = [{angles}]°.")
        fit_text = ""
        if result.metadata.get("variant") == "fitted":
            speeds = [item["initial_speed_m_s"] for item in result.metadata["fit_details"]]
            fit_text = (f" Подбор на ±{result.metadata['fit_half_window_s']:g} с; "
                        f"начальные |v_rel| = {', '.join(f'{v:.1f}' for v in speeds)} м/с.")
        self.info.setText(f"Источник: {result.metadata['source']}. max ||ρ||/a₀ = "
                          f"{result.metadata['max_rho_over_a']:.4f}; min высота КА = "
                          f"{result.metadata['min_altitude_km']:.2f} км. "
                          f"{result.metadata['warning']}{fit_text}")

    def _draw_projection(self,k):
        r = self.result
        ax = self.proj.axes
        ax.clear()
        to_deg = lambda x: np.rad2deg(np.arctan(x))
        if r.metadata.get("grid_size",2) > 2:
            self._draw_grid_2d(ax,to_deg(r.image_gnomonic[k]),
                to_deg(r.target_image_gnomonic),
                to_deg(r.image_gnomonic.reshape(-1,2)),
                "Гномоническая проекция", "atan(ξ), ° — координатный угол",
                "atan(η), ° — координатный угол",self.proj)
            return
        target = to_deg(r.target_image_gnomonic[:4])
        ax.plot(*np.r_[target,target[:1]].T, "--", color="#888888", label="цель в зените")
        p = to_deg(r.image_gnomonic[k,:4])
        if np.isfinite(p).all():
            ax.plot(*np.r_[p,p[:1]].T, color="#303030",linewidth=1.5,label="фактически")
        for j, point in enumerate(p):
            if np.isfinite(point).all():
                ax.scatter(*point,color=COLORS[j],s=65,zorder=3)
                ax.annotate(str(j),point,xytext=(5,5),textcoords="offset points",fontsize=10)
        all_points = to_deg(r.image_gnomonic.reshape(-1,2))
        finite = all_points[np.isfinite(all_points).all(axis=1)]
        both = np.vstack((target,finite)) if len(finite) else target
        low, high = np.min(both,axis=0), np.max(both,axis=0)
        center=(low+high)/2
        radius=max(float(np.max(high-low))/2*1.12, 0.5)
        ax.set_xlim(center[0]-radius,center[0]+radius)
        ax.set_ylim(center[1]-radius,center[1]+radius)
        ax.set_aspect("equal",adjustable="box")
        ax.set_title("Гномоническая проекция")
        ax.set_xlabel("atan(ξ), ° — координатный угол")
        ax.set_ylabel("atan(η), ° — координатный угол")
        ax.grid(alpha=.3)
        ax.legend(loc="best",fontsize=8)
        self.proj.draw_idle()

    def _draw_yoz(self,k):
        result = self.result
        ax = self.yoz.axes
        ax.clear()
        if result.metadata.get("grid_size",2) > 2:
            self._draw_grid_2d(ax,result.rho_oso_km[k,:,1:3],
                result.target_yoz_km,result.rho_oso_km[:,:,1:3].reshape(-1,2),
                "Пространственная конфигурация yOz", "y ОСО, км", "z ОСО, км",self.yoz)
            return
        target = result.target_yoz_km[:4]
        actual = result.rho_oso_km[k,:4,1:3]
        ax.plot(*np.r_[target,target[:1]].T,"--",color="#888888",label="целевой контур ОСО")
        ax.plot(*np.r_[actual,actual[:1]].T,color="#303030",linewidth=1.5,label="фактически")
        for j,point in enumerate(actual):
            ax.scatter(*point,color=COLORS[j],s=65,zorder=3)
            ax.annotate(str(j),point,xytext=(5,5),textcoords="offset points",fontsize=10)
        all_points = np.concatenate((result.rho_oso_km[:,:4,1:3].reshape(-1,2),target))
        low,high = np.min(all_points,axis=0),np.max(all_points,axis=0)
        center = (low+high)/2
        radius = max(float(np.max(high-low))*0.56,1.0)
        ax.set_xlim(center[0]-radius,center[0]+radius)
        ax.set_ylim(center[1]-radius,center[1]+radius)
        ax.set_aspect("equal",adjustable="box")
        ax.set_title("Пространственная конфигурация yOz")
        ax.set_xlabel("y ОСО, км")
        ax.set_ylabel("z ОСО, км")
        ax.grid(alpha=.3)
        ax.legend(loc="best",fontsize=8)
        self.yoz.draw_idle()

    def _draw_grid_2d(self,ax,points,target,all_points,title,xlabel,ylabel,canvas):
        indices = np.asarray(self.result.metadata["grid_indices"])
        size = len(indices)
        colors = plt.get_cmap("viridis",size)
        for row in range(size):
            ax.plot(*target[indices[row]].T,"--",color="#a7a7a7",linewidth=.7,alpha=.7)
            ax.plot(*points[indices[row]].T,color="#165f84",linewidth=1.1,alpha=.85)
        for col in range(size):
            ax.plot(*target[indices[:,col]].T,"--",color="#a7a7a7",linewidth=.7,alpha=.7)
            ax.plot(*points[indices[:,col]].T,color="#bd6a24",linewidth=1.1,alpha=.85)
        color_by_index = np.empty((size*size,4))
        for row in range(size):
            color_by_index[indices[row]] = colors(row)
        ax.scatter(points[:,0],points[:,1],c=color_by_index,s=19,zorder=3)
        corners = self.result.metadata["corner_indices"]
        for j in corners:
            if np.isfinite(points[j]).all():
                ax.annotate(str(j),points[j],xytext=(4,4),textcoords="offset points",fontsize=8)
        finite = all_points[np.isfinite(all_points).all(axis=1)]
        combined = np.vstack((target,finite)) if len(finite) else target
        low,high = np.min(combined,axis=0),np.max(combined,axis=0)
        center=(low+high)/2
        radius=max(float(np.max(high-low))*0.56,0.5 if "atan" in xlabel else 1.0)
        ax.set_xlim(center[0]-radius,center[0]+radius)
        ax.set_ylim(center[1]-radius,center[1]+radius)
        ax.set_aspect("equal",adjustable="box")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.3)
        canvas.draw_idle()

    def _draw_space(self,k):
        r = self.result
        ax = self.space.axes
        ax.clear()
        local = self.view.currentIndex() == 0
        if local:
            xyz = r.rho_oso_km
            label = ("x ОСО, км", "y ОСО, км", "z ОСО, км")
            low = np.min(xyz,axis=(0,1))
            high = np.max(xyz,axis=(0,1))
            center = (low+high)/2
            radius = max(1.0,float(np.max(high-low))*0.575)
            bounds = [(c-radius,c+radius) for c in center]
            ax.set_title("Локальная ОСО: относительные траектории")
        else:
            xyz = r.r_eci_km
            label = ("X ECI, км", "Y ECI, км", "Z ECI, км")
            orbit = r.reference_orbit_eci_km
            ax.plot(*orbit.T,color="#888888",linewidth=.9,label="номинальная орбита: полный виток")
            re = r.metadata["orbit_radius_km"]
            u = np.linspace(0,2*np.pi,25)
            v = np.linspace(0,np.pi,13)
            earth = np.array([re*np.outer(np.cos(u),np.sin(v)),
                              re*np.outer(np.sin(u),np.sin(v)),
                              re*np.outer(np.ones_like(u),np.cos(v))])
            ax.plot_surface(*earth,color="#9fc4e9",alpha=.38,linewidth=0)
            limit = re+r.metadata["reference_altitude_km"]+150
            bounds = [(-limit,limit)]*3
            ax.set_title("Инерциальный вид: доступный участок КА")
        if r.metadata.get("grid_size",2)>2:
            ids = np.asarray(r.metadata["grid_indices"])
            size = len(ids)
            corner_ids = r.metadata["corner_indices"]
            selected = list(dict.fromkeys(corner_ids+[int(ids[size//2,size//2])]))
            for j in selected:
                if j == 0:
                    continue
                start = max(0,k-40) if local else 0
                ax.plot(*xyz[start:k+1,j].T,color=COLORS[j%len(COLORS)],linewidth=1.4)
            rows = np.empty(size*size,dtype=int)
            for row in range(size):
                rows[ids[row]] = row
            ax.scatter(*xyz[k].T,c=rows,cmap="viridis",s=28 if local else 12,
                       depthshade=False)
            for j in corner_ids:
                ax.text(*xyz[k,j],f" {j}",fontsize=8)
        else:
            for j in range(1,xyz.shape[1]):
                start = max(0,k-40) if local else 0
                ax.plot(*xyz[start:k+1,j].T,color=COLORS[j%len(COLORS)],linewidth=2)
            for j in range(xyz.shape[1]):
                ax.scatter(*xyz[k,j],color=COLORS[j%len(COLORS)],s=90 if local else 55,depthshade=False)
                ax.text(*xyz[k,j],f" {j}",color=COLORS[j%len(COLORS)])
        ax.set(xlim=bounds[0],ylim=bounds[1],zlim=bounds[2],xlabel=label[0],ylabel=label[1],zlabel=label[2])
        ax.set_box_aspect((1,1,1))
        self.space.draw_idle()

    def closeEvent(self,event):
        self.audit_timer.stop()
        self.audit_executor.shutdown(wait=False,cancel_futures=True)
        super().closeEvent(event)


def run():
    app = QApplication.instance() or QApplication([])
    plt.rcParams["font.family"] = "DejaVu Sans"
    window = DemoWindow()
    window.show()
    return app.exec()
