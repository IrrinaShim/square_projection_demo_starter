"""Окно одного пролёта: готовая геометрия и траектории из расчётного слоя."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from threading import Event
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QLineEdit, QSlider, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QToolButton,
    QVBoxLayout, QWidget)

from .drag_provider import calculate_drag_case
from .drag_study import load_or_search
from .long_report import create_long_report


class Plot(FigureCanvasQTAgg):
    def __init__(self, projection=None):
        self.figure=Figure(figsize=(5,4),constrained_layout=projection!="3d")
        self.axes=self.figure.add_subplot(111,projection=projection)
        if projection=="3d":
            self.figure.subplots_adjust(left=.04,right=.94,bottom=.12,top=.86)
        super().__init__(self.figure)
        self.setMinimumWidth(300)
        self._animated_artists=[]
        self._background=None
        if projection!="3d": self.mpl_connect("draw_event",self._save_background)

    def _save_background(self,event):
        self._background=self.copy_from_bbox(self.axes.bbox)
        # Draw into this render before Qt paints/grabs it. Deferring to a
        # timer left animated points absent after resize/expose or screenshots.
        for artist in self._animated_artists:
            self.axes.draw_artist(artist)

    def fast_draw(self):
        if self._background is None: return
        self.restore_region(self._background)
        for artist in self._animated_artists:
            self.axes.draw_artist(artist)
        self.blit(self.axes.bbox)


class ObservableTimeline(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(14)
        self.times=None
        self.eligible=None
        self.index=0

    def set_result(self,times,eligible):
        self.times=times
        self.eligible=eligible
        self.update()

    def set_index(self,index):
        self.index=index
        self.update()

    def paintEvent(self,event):
        if self.times is None:
            return
        painter=QPainter(self)
        painter.setPen(Qt.PenStyle.NoPen)
        left,width=5,max(1,self.width()-10)
        painter.fillRect(left,4,width,6,QColor("#d4d8de"))
        span=self.times[-1]-self.times[0]
        for k,good in enumerate(self.eligible):
            if good:
                a=self.times[0] if k==0 else (self.times[k-1]+self.times[k])/2
                b=self.times[-1] if k==len(self.times)-1 else (self.times[k]+self.times[k+1])/2
                x0=left+round(width*(a-self.times[0])/span)
                x1=left+round(width*(b-self.times[0])/span)
                painter.fillRect(x0,4,max(1,x1-x0),6,QColor("#259b59"))
        cursor=left+round(width*(self.times[self.index]-self.times[0])/span)
        painter.fillRect(cursor-1,1,2,12,QColor("#263238"))


class DemoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        from orbit_study.passive_optimizer import VERSION as optimizer_version
        self.setWindowTitle(f"Наблюдаемая сетка КА — один пролёт — подбор v{optimizer_version}")
        self.resize(1530,900)
        self.result=None
        self.scenario=None
        self.study_record=None
        self.study_data=None
        self._grid_artists={}
        self._space_artists=None
        self._executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix="orbit-search")
        self._future=None
        self._cancel_event=None
        self._generation=0
        self._progress=Queue()
        self._queued_calculation=False
        self._poll=QTimer(self)
        self._poll.setInterval(80)
        self._poll.timeout.connect(self._collect_calculation)
        self.timer=QTimer(self)
        self.timer.setInterval(110)
        self.timer.timeout.connect(self.advance)
        central=QWidget()
        self.setCentralWidget(central)
        root=QVBoxLayout(central)
        top=QHBoxLayout()
        root.addLayout(top)
        self.mode=QComboBox()
        self.mode.addItem("Без управления","passive")
        self.mode.addItem("С управлением","controlled")
        top.addWidget(QLabel("Режим:"))
        top.addWidget(self.mode,2)
        self.view=QComboBox()
        self.view.addItem("3D: локальная система центра группы","local")
        self.view.addItem("3D: инерциальная система — приблизить группу","eci_close")
        self.view.addItem("3D: инерциальная система — Земля целиком","eci_earth")
        self.view.addItem("3D: инерциальная система — неподвижная камера","eci_fixed")
        self.view.currentIndexChanged.connect(self.update_frame)
        top.addWidget(self.view,3)

        groups=QHBoxLayout()
        root.addLayout(groups)
        config=QGroupBox("Конфигурация")
        cf=QFormLayout(config)
        self.grid_size=QSpinBox()
        self.grid_size.setRange(2,10)
        self.grid_size.setValue(10)
        cf.addRow("КА на стороне N",self.grid_size)
        self.angular_side=QDoubleSpinBox()
        self.angular_side.setRange(1,30)
        self.angular_side.setDecimals(9)
        self.angular_side.setValue(20)
        self.angular_side.setSuffix(" °")
        cf.addRow("Внешняя сторона (цель)",self.angular_side)
        groups.addWidget(config,2)
        observer=QGroupBox("Наблюдатель")
        of=QFormLayout(observer)
        self.observer_lat=QDoubleSpinBox(); self.observer_lat.setRange(-90,90); self.observer_lat.setDecimals(4); self.observer_lat.setValue(25.3026)
        self.observer_lon=QDoubleSpinBox(); self.observer_lon.setRange(-180,180); self.observer_lon.setDecimals(4); self.observer_lon.setValue(51.5132)
        self.elevation_limit=QDoubleSpinBox(); self.elevation_limit.setRange(0,90); self.elevation_limit.setValue(15); self.elevation_limit.setSuffix(" °")
        of.addRow("Широта, °",self.observer_lat)
        of.addRow("Долгота, °",self.observer_lon)
        of.addRow("Мин. угол места",self.elevation_limit)
        groups.addWidget(observer,2)
        orbit=QGroupBox("Орбита")
        orb=QFormLayout(orbit)
        self.orbit_height=QDoubleSpinBox(); self.orbit_height.setRange(500,700); self.orbit_height.setValue(700); self.orbit_height.setSuffix(" км")
        self.orbit_inclination=QDoubleSpinBox(); self.orbit_inclination.setRange(0,180); self.orbit_inclination.setDecimals(3); self.orbit_inclination.setValue(97.8); self.orbit_inclination.setSuffix(" °")
        self.orbit_parameters=QLabel("—")
        self.epoch=QLabel("—")
        self.epoch_input=QLineEdit("2026-09-28T00:00:00Z")
        self.epoch_input.setToolTip("Общая UTC-эпоха оскулирующих элементов")
        orb.addRow("Высота КА 0",self.orbit_height)
        orb.addRow("Наклонение КА 0",self.orbit_inclination)
        orb.addRow("Период КА 0",self.orbit_parameters)
        orb.addRow("Эпоха UTC (ввод)",self.epoch_input)
        orb.addRow("Эпоха расчёта",self.epoch)
        groups.addWidget(orbit,2)
        model=QGroupBox("Модель")
        ml=QVBoxLayout(model)
        self.model_summary=QLabel("Нелинейная гравитация Земли + J₂; тяга отсутствует")
        self.model_summary.setWordWrap(True)
        ml.addWidget(self.model_summary)
        self.model_button=QToolButton()
        self.model_button.setText("Параметры модели ▸")
        self.model_button.setCheckable(True)
        ml.addWidget(self.model_button)
        self.model_details=QLabel()
        self.model_details.setWordWrap(True)
        self.model_details.setVisible(False)
        self.model_button.toggled.connect(self.model_details.setVisible)
        ml.addWidget(self.model_details)
        self.model_details.setText("Разрешение прибора: отдельный исследовательский сценарий с порогом истинного угла соседей 2°; в критерий квадратности не входит.")
        groups.addWidget(model,2)
        self.grid_size.valueChanged.connect(self.calculate_current)
        self.angular_side.valueChanged.connect(self.calculate_current)
        self.mode.currentIndexChanged.connect(self.calculate_current)
        for widget in (self.observer_lat,self.observer_lon,self.elevation_limit,
                       self.orbit_height,self.orbit_inclination):
            widget.valueChanged.connect(self.calculate_current)
        self.epoch_input.editingFinished.connect(self.calculate_current)

        self.status=QLabel()
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        search_controls=QHBoxLayout()
        root.addLayout(search_controls)
        self.calculate_button=QPushButton("Рассчитать")
        self.calculate_button.clicked.connect(self.run_calculation)
        search_controls.addWidget(self.calculate_button)
        self.cancel_button=QPushButton("Отменить расчёт")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_current)
        search_controls.addWidget(self.cancel_button)
        search_controls.addStretch(1)
        technical=QHBoxLayout()
        root.addLayout(technical)
        self.elements_button=QToolButton()
        self.elements_button.setText("Элементы орбит ▸")
        self.elements_button.setCheckable(True)
        self.elements_button.toggled.connect(self._toggle_elements)
        technical.addWidget(self.elements_button)
        self.export_button=QPushButton("Сохранить элементы JSON")
        self.export_button.clicked.connect(self._choose_elements_path)
        technical.addWidget(self.export_button)
        technical.addStretch(1)
        technical.addWidget(QLabel("Отдельный анализ, сут.:"))
        self.long_horizon=QDoubleSpinBox()
        self.long_horizon.setRange(.05,30.)
        self.long_horizon.setDecimals(2)
        self.long_horizon.setValue(1.)
        technical.addWidget(self.long_horizon)
        self.long_button=QPushButton("Сохранить отдельный отчёт")
        self.long_button.clicked.connect(self._choose_long_analysis_dir)
        technical.addWidget(self.long_button)
        self.elements_table=QTableWidget(0,7)
        self.elements_table.setHorizontalHeaderLabels(["КА","a, км","e","i, °","Ω, °","ω, °","ν, °"])
        self.elements_table.verticalHeader().setVisible(False)
        self.elements_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.elements_table.setFixedHeight(160)
        self.elements_table.setVisible(False)
        root.addWidget(self.elements_table)
        self.long_status=QLabel("Долгосрочный расчёт не выполнен. Отдельный отчёт по запросу сохранит исходные данные и причину ограничения.")
        self.long_status.setWordWrap(True)
        root.addWidget(self.long_status)

        splitter=QSplitter(Qt.Orientation.Horizontal)
        self.proj=Plot()
        self.yoz=Plot()
        self.space=Plot("3d")
        for plot in (self.proj,self.yoz,self.space): splitter.addWidget(plot)
        splitter.setSizes([500,500,500])
        root.addWidget(splitter,1)
        transport=QHBoxLayout()
        root.addLayout(transport)
        self.start_button=QPushButton("В начало")
        self.start_button.clicked.connect(lambda:self.time_slider.setValue(0))
        self.play_button=QPushButton("Пуск")
        self.play_button.clicked.connect(self.toggle_play)
        self.time_slider=QSlider(Qt.Orientation.Horizontal)
        self.time_slider.valueChanged.connect(self.update_frame)
        self.timeline=ObservableTimeline()
        self.timeline.setToolTip("Зелёные участки: вся сетка удовлетворяет форме, размеру и видимости на показанном пролёте")
        time_column=QVBoxLayout()
        time_column.setSpacing(0)
        time_column.addWidget(self.time_slider)
        time_column.addWidget(self.timeline)
        self.time_label=QLabel("t = — с")
        self.speed=QComboBox()
        for label,value in (("0,5×",.5),("1×",1),("2×",2),("5×",5)):
            self.speed.addItem(label,value)
        self.speed.setCurrentIndex(1)
        transport.addWidget(self.start_button)
        transport.addWidget(self.play_button)
        transport.addLayout(time_column,1)
        transport.addWidget(self.time_label)
        transport.addWidget(QLabel("Скорость:"))
        transport.addWidget(self.speed)
        self.observable_badge=QLabel()
        self.observable_badge.setWordWrap(True)
        root.addWidget(self.observable_badge)
        self.pass_badge=QLabel()
        self.pass_badge.setWordWrap(True)
        root.addWidget(self.pass_badge)
        self.orbit_badge=QLabel()
        self.orbit_badge.setWordWrap(True)
        root.addWidget(self.orbit_badge)
        self.safety_badge=QLabel()
        self.safety_badge.setWordWrap(True)
        root.addWidget(self.safety_badge)
        self.metrics=QLabel()
        self.metrics.setWordWrap(True)
        root.addWidget(self.metrics)
        self.geometry=QLabel()
        self.geometry.setWordWrap(True)
        root.addWidget(self.geometry)
        self.note=QLabel()
        self.note.setWordWrap(True)
        root.addWidget(self.note)
        self.calculate_current()
        self.run_calculation()

    def _clear_result(self,message):
        self.timer.stop()
        self.play_button.setText("Пуск")
        self.result=None
        self.scenario=None
        self.study_record=None
        self.study_data=None
        for badge in (self.observable_badge,self.pass_badge,self.orbit_badge,
                      self.safety_badge,self.metrics,self.geometry,self.note):
            badge.clear()
        self._grid_artists={}
        self._space_artists=None
        self.timeline.set_result(None,None)
        self.status.setText(message)
        self.status.setStyleSheet("background:#ffe3df;color:#8b241d;padding:5px;")
        self.observable_badge.clear()
        self.safety_badge.clear()
        self.metrics.clear()
        self.geometry.clear()
        self.note.clear()
        self.elements_table.setRowCount(0)
        self.elements_table.setVisible(False)
        self.elements_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.long_button.setEnabled(False)
        self.time_slider.setEnabled(False)
        self.play_button.setEnabled(False)
        self.orbit_parameters.setText("—")
        self.epoch.setText("—")
        self.model_details.clear()
        self.long_status.setText("Долгосрочный отчёт недоступен без выбранного расчёта.")
        for canvas in (self.proj,self.yoz,self.space):
            canvas._animated_artists=[]
            canvas._background=None
            canvas.axes.clear()
            canvas.axes.set_axis_off()
            if canvas is self.space:
                canvas.axes.text2D(.5,.5,"Нет рассчитанного случая",
                    transform=canvas.axes.transAxes,ha="center",va="center")
            else:
                canvas.axes.text(.5,.5,"Нет рассчитанного случая",
                    transform=canvas.axes.transAxes,ha="center",va="center")
            canvas.draw_idle()

    def calculate_current(self,*_):
        self._generation+=1
        self._queued_calculation=False
        if self._cancel_event is not None: self._cancel_event.set()
        self._clear_result("Параметры изменены. Нажмите «Рассчитать»: сохранённый случай загрузится, для новых параметров начнётся подбор.")
        if self.mode.currentData()=="controlled":
            self._start_calculation()

    def _request_parameters(self):
        return (self.grid_size.value(),self.angular_side.value(),2.,
                self.observer_lat.value(),self.observer_lon.value(),
                self.elevation_limit.value(),self.orbit_height.value(),
                self.orbit_inclination.value(),self.epoch_input.text().strip())

    @staticmethod
    def _calculate_task(parameters,cancel_event,do_optimization,budget,progress):
        size,side,resolution,lat,lon,elevation,altitude,inclination,epoch=parameters
        try:
            scenario,candidate,audit,source=load_or_search(size,side,resolution,lat,lon,
                elevation,altitude,inclination,cancel_event=cancel_event,
                epoch_utc=epoch,allow_search=False)
        except (FileNotFoundError,ValueError) as exc:
            if isinstance(exc,ValueError) and "Сохранённый подбор несовместим" not in str(exc):
                raise
            if not do_optimization:raise
            from orbit_study.passive_optimizer import configuration, optimize
            cfg=configuration(size,side,lat,lon,elevation,epoch,altitude,inclination)
            progress("Совместимого сохранённого расчёта нет; подбираю соседние сонаправленные орбиты и проверяю пролёт…")
            optimize(cfg,size,budget=budget,cancel_event=cancel_event,progress=progress)
            scenario,candidate,audit,source=load_or_search(size,side,resolution,lat,lon,
                elevation,altitude,inclination,cancel_event=cancel_event,
                epoch_utc=epoch,allow_search=False)
        if cancel_event.is_set(): raise ValueError("Расчёт отменён")
        result=calculate_drag_case(scenario,candidate)
        return scenario,candidate,audit,source,result

    def run_calculation(self):
        self._generation+=1
        if self._cancel_event is not None:self._cancel_event.set()
        self._queued_calculation=True
        self._clear_result("Рассчитываю: проверяю сохранённый случай; при его отсутствии начну подбор…")
        self._start_calculation()

    def cancel_current(self):
        if self._cancel_event is not None:self._cancel_event.set()
        self._generation+=1
        self._queued_calculation=False
        self.status.setText("Отмена запрошена. Уже рассчитанные кандидаты сохранены; проверяю доступный результат…")

    def _start_calculation(self):
        size=self.grid_size.value()
        side=self.angular_side.value()
        if self.mode.currentData()=="controlled":
            self.calculate_button.setEnabled(True)
            self._queued_calculation=False
            self.model_summary.setText("Расчёт управления для выбранной сетки пока не подготовлен")
            reason=("исторический случай 4 КА имеет внешнюю сторону 1,8183°, фактически приложенную тягу до 0,510 Н при принятой массе 22 кг, но не имеет согласованной UTC-эпохи, модели J₂+сопротивление и подтверждённого двигателя 16U"
                    if size==2 else "для N>2 рассчитанных данных управления нет")
            self._clear_result(f"С управлением: для N={size}, внешней стороны {side:g}° проверенного решения нет; {reason}.")
            return
        self.calculate_button.setEnabled(True)
        self.model_summary.setText("Каждый КА: собственная орбита; центральное поле + J₂ + атмосферное сопротивление; тяга отсутствует")
        if self._future is not None and not self._future.done():
            self.status.setText("Новый запрос ожидает завершения предыдущего; устаревший результат будет отброшен…")
            self._poll.start()
            return
        self._cancel_event=Event()
        self._queued_calculation=False
        self.cancel_button.setEnabled(True)
        self._future=self._executor.submit(self._calculate_task,self._request_parameters(),
                                           self._cancel_event,True,max(16,4*size),
                                           lambda message,generation=self._generation:
                                               self._progress.put((generation,message)))
        self._future_generation=self._generation
        self.status.setText(f"N={size}, сторона {side:g}°: загружаю проверенный результат или подбираю новые орбиты в фоновом потоке…")
        self._poll.start()

    def _collect_calculation(self):
        try:
            while True:
                generation,message=self._progress.get_nowait()
                if generation==self._generation:self.status.setText(message)
        except Empty:pass
        if self._future is None or not self._future.done(): return
        future=self._future
        self._future=None
        self.cancel_button.setEnabled(False)
        if self._future_generation!=self._generation:
            if self.mode.currentData()=="passive" and self._queued_calculation:
                self._start_calculation()
            else: self._poll.stop()
            return
        self._poll.stop()
        try:
            scenario,candidate,audit,source,result=future.result()
            if ("saved_duration_s" in audit
                    and abs(result.metadata["quality"]["longest_contiguous_duration_s"]
                            -audit["saved_duration_s"])>1e-8):
                raise ValueError("Повторный пролёт не совпадает с сохранённой проверкой")
        except Exception as exc:
            print(f"square_projection calculation_error: {exc}",flush=True)
            self._clear_result(str(exc))
            return
        size=self.grid_size.value(); side=self.angular_side.value()
        self.timer.stop()
        self.play_button.setText("Пуск")
        self.scenario,self.study_record,self.study_data=scenario,audit,candidate
        self.result=result
        print(f"square_projection selected: N={candidate['N']} side={scenario['angular_side_deg']:g}°; "
              f"{source}; square_1s={audit.get('verified_duration_1s_s',0):g}s; "
              f"max_normal_orbit={audit['orbital_direction']['full_orbit']['max_angle_deg']:.6f}°",
              flush=True)
        self._grid_artists={}
        self._space_artists=None
        self.time_slider.setEnabled(True)
        self.play_button.setEnabled(True)
        self.elements_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.long_button.setEnabled(True)
        self.long_status.setText("Долгосрочный расчёт не выполнен. Отдельный отчёт по запросу сохранит исходные данные и причину ограничения.")
        duration=result.metadata["quality"]["longest_contiguous_duration_s"]
        safe=audit["closest_pair"]["distance_km"]>=audit["threshold_km"]
        density_range=scenario["atmosphere"]["documented_range_km"]
        model_valid=(audit["minimum_altitude_km"]>=density_range[0] and
                     audit["maximum_altitude_km"]<=density_range[1])
        full_orbit_ok=bool(audit["full_orbit_ok"] and safe and model_valid)
        checked_duration=audit.get("verified_duration_1s_s",duration)
        resolved_duration=audit.get("verified_resolution_2deg_1s_s",audit["instrument_duration_2deg_s"])
        accepted=checked_duration>0 and full_orbit_ok
        state=("наблюдаемый квадрат и физическая проверка витка пройдены" if accepted else
               "часть условий демонстрационного случая не выполнена")
        problems=[]
        if checked_duration<=0: problems.append("не выполнены видимость, форма или внешний размер всей сетки")
        if not safe: problems.append("порог сближения нарушен")
        if not model_valid: problems.append(f"высота {audit['minimum_altitude_km']:.0f}…{audit['maximum_altitude_km']:.0f} км выходит за область плотности {density_range[0]:g}…{density_range[1]:g} км")
        problems.extend(audit["violations"])
        problem_text=(" Нарушения: "+"; ".join(problems)+"." if problems else "")
        problem_text+=f" Отдельный сценарий разрешения ≥2°: {resolved_duration:g} с."
        normal_angle=audit["orbital_direction"]["full_orbit"]["max_angle_deg"]
        self.status.setText(f"N={size}, внешняя сторона {side:g}°: {state}; {source}. Проверка проекции через 1 с: {checked_duration:g} с; показанные кадры через {scenario['display']['step_s']:g} с: {duration:g} с. Минимум всех {audit['pair_count']} пар за полный виток {audit['closest_pair']['distance_km']:.3f} км; максимальный угол нормалей {normal_angle:.3f}° (предел 5°, hᵢ·h₀>0).{problem_text} Многосуточная повторяемость не проверена.")
        self.status.setStyleSheet("background:#e3f2e5;color:#245a30;padding:5px;" if accepted else "background:#ffe3df;color:#8b241d;padding:5px;")
        observer=scenario["observer"]
        orbit=scenario["orbit"]
        self.orbit_parameters.setText(f"{result.metadata['reference_period_s']:.1f} с")
        self.epoch.setText(scenario["epoch_utc"])
        craft=scenario["spacecraft"]; atmosphere=scenario["atmosphere"]
        self.model_details.setText(f"ECI средней даты, GMST≈{observer['greenwich_angle_at_zenith_deg']:.3f}° (UT1≈UTC); μ={orbit['mu_km3_s2']:g} км³/с²; J₂={orbit['j2']:.11f}. 16U: m={craft['mass_kg']:g} кг (допущение для этой группы), A={craft['effective_area_m2']:g} м², CD={craft['drag_coefficient']:g} (допущение), широкая грань к потоку. Плотность при 700 км {atmosphere['rho_reference_kg_m3']:.2e} кг/м³, H={atmosphere['scale_height_km']:g} км; приближённая модель. Локальное начало — центр {size*size} КА. Разрешение прибора при истинном угле соседей ≥2° исследуется отдельно от квадратности. Гармоники выше J₂, Солнце, Луна, световое давление и тяга не включены.")
        self.elements_table.setRowCount(size*size)
        self.elements_table.setVisible(self.elements_button.isChecked())
        for row,entry in enumerate(result.metadata["orbital_elements"]):
            values=[str(row),f"{entry['a_km']:.6f}",f"{entry['e']:.9f}",f"{entry['inclination_deg']:.6f}",f"{entry['raan_deg']:.6f}",f"{entry['arg_perigee_deg']:.6f}",f"{entry['true_anomaly_deg']:.6f}"]
            for col,value in enumerate(values): self.elements_table.setItem(row,col,QTableWidgetItem(value))
        self.time_slider.blockSignals(True)
        self.time_slider.setRange(0,len(result.t_s)-1)
        self.time_slider.setValue(int(np.argmin(abs(result.t_s))))
        self.time_slider.blockSignals(False)
        self.timeline.set_result(result.t_s,result.metadata["observability"]["eligible"])
        self.update_frame()

    def _toggle_elements(self,expanded):
        self.elements_table.setVisible(expanded and self.result is not None)
        self.elements_button.setText("Элементы орбит ▾" if expanded else "Элементы орбит ▸")

    def _choose_elements_path(self):
        if self.result is None: return
        path,_=QFileDialog.getSaveFileName(self,"Сохранить оскулирующие элементы",
            f"grid_n{self.grid_size.value()}_elements.json","JSON (*.json)")
        if path: self.export_orbit_elements(path)

    def export_orbit_elements(self,path):
        if self.result is None: raise ValueError("Нет выбранного набора")
        m=self.result.metadata
        payload={"epoch_utc":self.scenario["epoch_utc"],"grid_size":m["grid_size"],
            "index_map":m["grid_indices"],"model":self.scenario["model_id"],
            "elements_at_epoch":m["orbital_elements"],"reference_orbit":self.scenario["orbit"],
            "study_provenance":{"scope":"one_pass","included_forces":m["included_forces"]},
            "units":{"a":"km","angles":"deg","e":"dimensionless"}}
        Path(path).write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    def _choose_long_analysis_dir(self):
        if self.result is None: return
        folder=QFileDialog.getExistingDirectory(self,"Папка отдельного отчёта",
            str(Path("orbit_study/user_runs").resolve()))
        if folder: self.start_long_analysis(folder)

    def start_long_analysis(self,output_parent=None):
        if self.result is None: return
        try:
            saved=create_long_report(self.scenario,self.study_data,self.study_record,
                                     self.result,self.long_horizon.value(),output_parent)
            self.long_status.setText(f"Отдельный отчёт: {saved['folder']}. Многосуточный расчёт не выполнен: {saved['reason']}")
        except (ValueError,OSError) as exc:
            self.long_status.setText(f"Отчёт не сохранён: {exc}")

    def toggle_play(self):
        if self.result is None: return
        if self.timer.isActive():
            self.timer.stop(); self.play_button.setText("Пуск")
        else:
            if self.time_slider.value()==self.time_slider.maximum(): self.time_slider.setValue(0)
            self.timer.start(); self.play_button.setText("Пауза")

    def advance(self):
        if self.result is None: return
        k=self.time_slider.value()
        t=self.result.t_s
        target=t[k]+self.timer.interval()/1000*self.speed.currentData()*10
        next_index=int(np.searchsorted(t,target,side="left"))
        self.time_slider.setValue(min(max(k+1,next_index),len(t)-1))
        if self.time_slider.value()==len(t)-1:
            self.timer.stop(); self.play_button.setText("Пуск")

    def update_frame(self,*_):
        r=self.result
        if r is None: return
        k=self.time_slider.value()
        self.time_label.setText(f"t = {r.t_s[k]:.1f} с")
        self.timeline.set_index(k)
        self._draw_projection(k)
        self._draw_yoz(k)
        self._draw_space(k)
        obs=r.metadata["observability"]
        quality=r.metadata["quality"]
        good=bool(obs["eligible"][k])
        longest=quality["longest_interval_s"]
        interval=(f"{longest[0]:g}…{longest[1]:g} с ({quality['longest_contiguous_duration_s']:.0f} с)" if quality["has_positive_duration_interval"] else "отсутствует")
        visible=obs["visibility_intervals_s"]
        vis_text=", ".join(f"{a:g}…{b:g} с" for a,b in visible) or "нет"
        reason=obs['reasons'][k]
        self.observable_badge.setText(f"{'● Наблюдаемый квадрат сейчас: да' if good else '● Наблюдаемый квадрат сейчас: нет — '+reason}. Минимальный угол места {obs['min_elevation_deg'][k]:.2f}°. Видимость всех КА: {vis_text}.")
        self.observable_badge.setStyleSheet("background:#d9f4e3;color:#145b32;padding:5px;" if good else "background:#ffe3df;color:#8b241d;padding:5px;")
        audit=self.study_record
        verified=audit.get("verified_intervals_1s")
        if verified:
            best=max(verified,key=lambda item:item["duration_s"])
            precise=f" Проверка с шагом 1 с: {best['start_s']:g}…{best['end_s']:g} с ({best['duration_s']:g} с)."
        else:precise=""
        instrument=audit.get("verified_resolution_2deg_1s_s",audit['instrument_duration_2deg_s'])
        self.pass_badge.setText(f"Интервал на кадрах с шагом {self.scenario['display']['step_s']:g} с: {interval}.{precise} Отдельный сценарий разрешения 2°: {instrument:g} с (узлы 1 с); {audit['instrument_duration_2deg_s']:g} с на кадрах.")
        self.pass_badge.setStyleSheet("background:#d9f4e3;color:#145b32;padding:5px;" if quality["has_positive_duration_interval"] else "background:#ffe3df;color:#8b241d;padding:5px;")
        orbit_ok=audit["full_orbit_ok"]
        issues="; ".join(audit["full_orbit_violations"])
        normal_angle=audit["orbital_direction"]["full_orbit"]["max_angle_deg"]
        direction_text=f" Максимальный угол нормалей {normal_angle:.3f}° ≤ 5°, hᵢ·h₀>0."
        self.orbit_badge.setText("Полный виток: " + ("орбиты и все пары прошли принятые проверки." if orbit_ok else f"проверка не пройдена — {issues}.")+direction_text)
        self.orbit_badge.setStyleSheet("background:#d9f4e3;color:#145b32;padding:5px;" if orbit_ok else "background:#fff0d9;color:#805015;padding:5px;")
        closest=audit["closest_pair"]
        threshold=audit["threshold_km"]
        self.safety_badge.setText(f"Минимум между КА сейчас: {r.min_pair_distance_km[k]:.3f} км. Минимум за проверенный интервал: {closest['distance_km']:.3f} км, пара {closest['pair']}, t={closest['t_s']:.1f} с, {closest['latitude_deg']:.1f}° / {closest['longitude_deg']:.1f}°. Порог: размеры КА + принятый запас = {threshold:g} км (условный). Горизонт проверки: {audit['horizon_s'][0]:g}…{audit['horizon_s'][1]:g} с; все {audit['pair_count']} пар.")
        self.safety_badge.setStyleSheet("background:#e7edf8;color:#1c3764;padding:5px;" if closest["distance_km"]>=threshold else "background:#fff0d9;color:#805015;padding:5px;")
        gm=r.metadata["grid_metrics"]
        errors=r.shape_errors[k]
        resolution=("да" if obs["resolution_ok"][k] else "нет")
        self.metrics.setText(f"Внешняя сторона: {obs['outer_side_deg'][k]:.3f}° (цель {self.angular_side.value():g}°); истинный минимальный угол соседей: {gm['min_true_neighbor_angle_deg'][k]:.3f}°. Отдельный сценарий разрешения ≥2°: {resolution}. Ошибки формы: εb={errors[0]:.3f}, εd={errors[1]:.3f}, εγ={np.rad2deg(errors[2]):.2f}°; отклонение узлов {gm['point_error_steps'][k]:.3f} шага, неравномерность {gm['neighbor_step_error'][k]:.3f}.")
        sp=r.metadata["spatial_size_metrics"]
        corners=", ".join(f"{x:.2f}" for x in sp["corner_sides_km"][k])
        diagonals=", ".join(f"{x:.2f}" for x in sp["corner_diagonals_km"][k])
        self.geometry.setText(f"В пространстве: соседние стороны {sp['min_neighbor_km'][k]:.2f}…{sp['max_neighbor_km'][k]:.2f} км; стороны между крайними углами [{corners}] км; диагонали крайних углов [{diagonals}] км; max любой пары {sp['max_pair_km'][k]:.2f} км.")
        self.note.setText("Внешняя угловая сторона = atan(средней длины 4 сторон крайнего контура на гномонической плоскости). Расстояния в км относятся к КА между собой; дальность до наблюдателя сюда не входит. Центр локальной системы — расчётная точка, не дополнительный КА.")

    def _draw_grid(self,canvas,points,target,all_points,title,xlabel,ylabel):
        ax=canvas.axes
        ids=np.asarray(self.result.metadata["grid_indices"])
        size=len(ids)
        cache=self._grid_artists.get(canvas)
        if cache is None:
            ax.clear(); ax.set_axis_on()
            colors=plt.get_cmap("viridis",size)
            row_lines=[]; col_lines=[]
            for row in range(size):
                ax.plot(*target[ids[row]].T,"--",color="#a7a7a7",linewidth=.7,alpha=.7)
                row_lines.append(ax.plot([],[],color="#165f84",linewidth=1.1,alpha=.85)[0])
            for col in range(size):
                ax.plot(*target[ids[:,col]].T,"--",color="#a7a7a7",linewidth=.7,alpha=.7)
                col_lines.append(ax.plot([],[],color="#bd6a24",linewidth=1.1,alpha=.85)[0])
            rgba=np.empty((size*size,4))
            for row in range(size): rgba[ids[row]]=colors(row)
            scatter=ax.scatter(points[:,0],points[:,1],c=rgba,s=19,zorder=3)
            labels={j:ax.annotate(str(j),(0,0),xytext=(4,4),
                textcoords="offset points",fontsize=8) for j in self.result.metadata["corner_indices"]}
            finite=all_points[np.isfinite(all_points).all(axis=1)]
            combined=np.vstack((target,finite)) if len(finite) else target
            low,high=np.min(combined,axis=0),np.max(combined,axis=0)
            center=(low+high)/2
            radius=max(float(np.max(high-low))*.56,.5 if "atan" in xlabel else 1.)
            ax.set_xlim(center[0]-radius,center[0]+radius)
            ax.set_ylim(center[1]-radius,center[1]+radius)
            ax.set_aspect("equal",adjustable="box")
            ax.set(title=title,xlabel=xlabel,ylabel=ylabel)
            ax.grid(alpha=.3)
            cache=(row_lines,col_lines,scatter,labels)
            self._grid_artists[canvas]=cache
            canvas._animated_artists=[*row_lines,*col_lines,scatter,*labels.values()]
            for artist in canvas._animated_artists: artist.set_animated(True)
            canvas.draw()
        row_lines,col_lines,scatter,labels=cache
        for row,line in enumerate(row_lines): line.set_data(*points[ids[row]].T)
        for col,line in enumerate(col_lines): line.set_data(*points[ids[:,col]].T)
        scatter.set_offsets(points)
        for j,label in labels.items():
            label.xy=points[j]
            label.set_visible(bool(np.isfinite(points[j]).all()))
        canvas.fast_draw()

    def _draw_projection(self,k):
        r=self.result
        angle=lambda x:np.rad2deg(np.arctan(x))
        self._draw_grid(self.proj,angle(r.image_gnomonic[k]),angle(r.target_image_gnomonic),
            angle(r.image_gnomonic.reshape(-1,2)),"Гномоническая проекция",
            "atan(ξ), ° — координатный угол","atan(η), ° — координатный угол")

    def _draw_yoz(self,k):
        r=self.result
        self._draw_grid(self.yoz,r.rho_oso_km[k,:,1:3],r.target_yoz_km,
            r.rho_oso_km[:,:,1:3].reshape(-1,2),"Конфигурация yOz от центра группы",
            "y, км","z, км")

    def _draw_space(self,k):
        r=self.result
        ax=self.space.axes
        mode=self.view.currentData()
        if self._space_artists is None or self._space_artists["mode"]!=mode:
            ax.clear(); ax.set_axis_on()
            self._space_artists={"mode":mode}
            rebuild=True
        else:
            rebuild=False
        if mode=="local":
            xyz=r.rho_oso_km
            low=np.min(xyz,axis=(0,1)); high=np.max(xyz,axis=(0,1))
            center=(low+high)/2
            radius=max(1.,float(np.max(high-low))*.575)
            bounds=[(c-radius,c+radius) for c in center]
            labels=("x, км","y, км","z, км")
            if rebuild: ax.set_title("Центр группы\nСтрелки: относительная скорость",fontsize=10)
        else:
            xyz=r.r_eci_km
            labels=("X ECI, км","Y ECI, км","Z ECI, км")
            if mode in ("eci_close","eci_fixed"):
                center=xyz[k,0] if mode=="eci_close" else xyz[int(np.argmin(abs(r.t_s))),0]
                # Fixed span for the entire pass: contain the displayed 200 s
                # history AND every current spacecraft about the chosen chief.
                if rebuild:
                    extent=0.
                    for frame in range(len(xyz)):
                        origin=xyz[frame,0] if mode=="eci_close" else center
                        extent=max(extent,float(np.max(abs(xyz[max(0,frame-40):frame+1]-origin))))
                    self._space_artists["camera_radius_km"]=max(20.,extent*1.12)
                radius=self._space_artists["camera_radius_km"]
                bounds=[(c-radius,c+radius) for c in center]
                if rebuild: ax.set_title("ECI: камера следует за КА 0" if mode=="eci_close" else
                                         "ECI: камера неподвижна у КА 0 в эпоху")
            else:
                re=r.metadata["orbit_radius_km"]
                u=np.linspace(0,2*np.pi,25); v=np.linspace(0,np.pi,13)
                earth=np.array([re*np.outer(np.cos(u),np.sin(v)),
                    re*np.outer(np.sin(u),np.sin(v)),re*np.outer(np.ones_like(u),np.cos(v))])
                if rebuild:
                    ax.plot_surface(*earth,color="#9fc4e9",alpha=.38,linewidth=0)
                    ax.plot(*r.reference_orbit_eci_km.T,color="#888",linewidth=.9)
                limit=max(re,float(np.max(abs(xyz))),
                          float(np.max(abs(r.reference_orbit_eci_km))))*1.08
                bounds=[(-limit,limit)]*3
                if rebuild: ax.set_title("ECI: общий вид с Землёй")
        ids=np.asarray(r.metadata["grid_indices"])
        size=len(ids)
        corners=r.metadata["corner_indices"]
        selected=list(dict.fromkeys(corners+[int(ids[size//2,size//2])]))
        if rebuild:
            self._space_artists["traces"]={j:ax.plot([],[],[],color="#165f84",linewidth=1.4)[0]
                                           for j in selected}
        for j,line in self._space_artists["traces"].items():
            start=max(0,k-40)
            line.set_data_3d(*xyz[start:k+1,j].T)
        # Each arrow is tangent to its trace in that trace's coordinate frame.
        velocity=(r.metadata["local_velocity_oso_km_s"] if mode=="local"
                  else r.metadata["velocity_eci_km_s"])
        if "velocity_arrows" in self._space_artists:
            self._space_artists["velocity_arrows"].remove()
        selected_index=np.asarray(selected,dtype=int)
        self._space_artists["velocity_arrows"]=ax.quiver(
            *xyz[k,selected_index].T,*velocity[k,selected_index].T,
            length=radius*.12 if mode=="local" else (bounds[0][1]-bounds[0][0])*.06,
            normalize=True,color="#a32929",linewidth=.9,arrow_length_ratio=.28)
        self._space_artists["arrow_frame_index"]=k
        self._space_artists["trace_end_time_s"]=float(r.t_s[k])
        self._space_artists["arrow_vectors_km_s"]=velocity[k,selected_index].copy()
        self._space_artists["arrow_spacecraft_indices"]=selected_index.copy()
        rows=np.empty(size*size,dtype=int)
        for row in range(size): rows[ids[row]]=row
        if rebuild:
            self._space_artists["scatter"]=ax.scatter(*xyz[k].T,c=rows,cmap="viridis",s=28 if mode=="local" else 18,depthshade=False)
            self._space_artists["labels"]={j:ax.text(*xyz[k,j],f" {j}",fontsize=8) for j in corners}
        self._space_artists["scatter"]._offsets3d=tuple(xyz[k].T)
        for j,label in self._space_artists["labels"].items():
            label.set_position((xyz[k,j,0],xyz[k,j,1]))
            label.set_3d_properties(xyz[k,j,2])
        ax.set(xlim=bounds[0],ylim=bounds[1],zlim=bounds[2],
               xlabel=labels[0],ylabel=labels[1],zlabel=labels[2])
        ax.set_box_aspect((1,1,1))
        self.space.draw_idle()

    def closeEvent(self,event):
        if self._cancel_event is not None: self._cancel_event.set()
        self._executor.shutdown(wait=False,cancel_futures=True)
        super().closeEvent(event)


def run():
    app=QApplication.instance() or QApplication([])
    plt.rcParams["font.family"]="DejaVu Sans"
    window=DemoWindow()
    window.show()
    return app.exec()
