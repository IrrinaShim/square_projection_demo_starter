"""Окно одного пролёта: готовая геометрия и траектории из расчётного слоя."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QSlider, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QToolButton,
    QVBoxLayout, QWidget)

from .providers import calculate
from .study import CandidateUnavailable, load_study_case, study_status
from orbit_study.user_run import run_selected_case


class Plot(FigureCanvasQTAgg):
    def __init__(self, projection=None):
        self.figure=Figure(figsize=(5,4),constrained_layout=projection!="3d")
        self.axes=self.figure.add_subplot(111,projection=projection)
        if projection=="3d":
            self.figure.subplots_adjust(left=.02,right=.98,bottom=.03,top=.92)
        super().__init__(self.figure)
        self.setMinimumWidth(300)


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
        self.setWindowTitle("Наблюдаемая сетка КА — один пролёт")
        self.resize(1530,900)
        self.result=None
        self.scenario=None
        self.study_record=None
        self.study_data=None
        self.timer=QTimer(self)
        self.timer.setInterval(110)
        self.timer.timeout.connect(self.advance)
        self.analysis_executor=ThreadPoolExecutor(max_workers=1)
        self.analysis_future=None
        self.analysis_timer=QTimer(self)
        self.analysis_timer.setInterval(200)
        self.analysis_timer.timeout.connect(self._poll_long_analysis)
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
        self.angular_side.setDecimals(3)
        self.angular_side.setValue(20)
        self.angular_side.setSuffix(" °")
        cf.addRow("Внешняя сторона (цель)",self.angular_side)
        self.resolution=QComboBox()
        self.resolution.addItem("1°",1.)
        self.resolution.addItem("2°",2.)
        self.resolution.setCurrentIndex(1)
        cf.addRow("Истинный угол соседей ≥",self.resolution)
        groups.addWidget(config,2)
        observer=QGroupBox("Наблюдатель")
        of=QFormLayout(observer)
        self.observer_location=QLabel("—")
        self.min_elevation=QLabel("—")
        of.addRow("Широта / долгота",self.observer_location)
        of.addRow("Мин. угол места",self.min_elevation)
        groups.addWidget(observer,2)
        orbit=QGroupBox("Орбита")
        orb=QFormLayout(orbit)
        self.orbit_parameters=QLabel("—")
        self.epoch=QLabel("—")
        orb.addRow("Опорный КА 0",self.orbit_parameters)
        orb.addRow("Общая эпоха",self.epoch)
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
        groups.addWidget(model,2)
        self.grid_size.valueChanged.connect(self.calculate_current)
        self.angular_side.valueChanged.connect(self.calculate_current)
        self.resolution.currentIndexChanged.connect(self.calculate_current)
        self.mode.currentIndexChanged.connect(self.calculate_current)

        self.status=QLabel()
        self.status.setWordWrap(True)
        root.addWidget(self.status)
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
        self.long_button=QPushButton("Рассчитать и сохранить")
        self.long_button.clicked.connect(self._choose_long_analysis_dir)
        technical.addWidget(self.long_button)
        self.elements_table=QTableWidget(0,7)
        self.elements_table.setHorizontalHeaderLabels(["КА","a, км","e","i, °","Ω, °","ω, °","ν, °"])
        self.elements_table.verticalHeader().setVisible(False)
        self.elements_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.elements_table.setFixedHeight(160)
        self.elements_table.setVisible(False)
        root.addWidget(self.elements_table)
        self.long_status=QLabel("Дальнейшие пролёты рассчитываются только по отдельному запросу.")
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
        self.timeline.setToolTip("Зелёные участки: вся сетка удовлетворяет условиям")
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

    def _clear_result(self,message):
        self.timer.stop()
        self.play_button.setText("Пуск")
        self.result=None
        self.scenario=None
        self.study_record=None
        self.study_data=None
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
        self.orbit_parameters.setText("Нет выбранной орбиты")
        self.epoch.setText("—")
        self.observer_location.setText("—")
        self.min_elevation.setText("—")
        self.model_details.clear()
        for canvas in (self.proj,self.yoz,self.space):
            canvas.axes.clear()
            canvas.draw_idle()

    def calculate_current(self,*_):
        size=self.grid_size.value()
        side=self.angular_side.value()
        if self.mode.currentData()=="controlled":
            self.model_summary.setText("Расчёт управления для выбранной сетки пока не подготовлен")
            self._clear_result(f"С управлением: для N={size}, внешней стороны {side:g}° совместимого рассчитанного случая нет. Историческая траектория не является подобранной наблюдаемой сеткой.")
            return
        self.model_summary.setText("Каждый КА: собственная орбита; центральная гравитация + J₂; тяга отсутствует")
        try:
            scenario,record,data=load_study_case(size,float(self.resolution.currentData()),
                                                   outer_side_deg=side)
            result=calculate(scenario,include_audit=False)
            key=str(int(self.resolution.currentData()))
            saved=record["verification"]["repeat"][key]["best_pass"]
            actual=result.metadata["quality"]["longest_contiguous_duration_s"]
            if abs(actual-(saved["duration_s"] if saved else 0))>scenario["time"]["step_s"]+1e-6:
                raise ValueError("Сохранённый и текущий расчёты пролёта расходятся")
        except (CandidateUnavailable,ValueError,OSError,KeyError) as exc:
            self._clear_result(str(exc))
            return
        self.timer.stop()
        self.play_button.setText("Пуск")
        self.scenario,self.study_record,self.study_data=scenario,record,data
        self.result=result
        self.time_slider.setEnabled(True)
        self.play_button.setEnabled(True)
        self.elements_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.long_button.setEnabled(self.analysis_future is None)
        audit=record["verification"]
        state=study_status(record,float(self.resolution.currentData()),data["search_config"])
        reasons="; ".join(record["unmet_constraints"]) or "нет зафиксированных нарушений"
        self.status.setText(f"N={size}, внешняя сторона цели {side:g}°: {state}. Проверено {audit['horizon_days']:g} сут.; подходящих пролётов при шаге ≥{key}°: {audit['repeat'][key]['qualifying_passes']}. Ограничения: {reasons}.")
        self.status.setStyleSheet("background:#d9f4e3;color:#145b32;padding:5px;" if state=="требования выполнены" else "background:#fff0d9;color:#805015;padding:5px;" if state=="выполнены частично" else "background:#ffe3df;color:#8b241d;padding:5px;")
        observer=scenario["observer"]
        orbit=scenario["orbit"]
        self.observer_location.setText(f"{observer['latitude_deg']:.4f}° / {observer['longitude_deg']:.4f}°")
        self.min_elevation.setText(f"≥{scenario['visibility']['min_elevation_deg']:g}°")
        self.orbit_parameters.setText(f"h₀={orbit['altitude_km']:g} км; i₀={orbit['inclination_deg']:g}°; T₀={result.metadata['reference_period_s']:.1f} с")
        self.epoch.setText(scenario["epoch_utc"])
        self.model_details.setText(f"Условная ECI: Z вдоль оси Земли, Гринвич=0° на эпохе. Rₑ={orbit['earth_radius_km']:g} км; μ={orbit['mu_km3_s2']:g} км³/с²; J₂={orbit['j2']:.11f}. Сферическая вращающаяся Земля. Локальное начало — среднее положение {size*size} КА; оси ориентированы по КА 0, скорость включает вращение осей. Условный порог: сумма радиусов 0,02 км + запас 1 км. Атмосфера и ошибки состояния не включены.")
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
            "index_map":m["grid_indices"],"model":"independent_j2",
            "elements_at_epoch":m["orbital_elements"],"reference_orbit":self.scenario["orbit"],
            "study_provenance":self.scenario["study_provenance"],
            "units":{"a":"km","angles":"deg","e":"dimensionless"}}
        Path(path).write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    def _choose_long_analysis_dir(self):
        if self.result is None: return
        folder=QFileDialog.getExistingDirectory(self,"Папка отдельных результатов",
            str(Path("orbit_study/user_runs").resolve()))
        if folder: self.start_long_analysis(folder)

    def start_long_analysis(self,output_parent=None):
        if self.result is None or self.analysis_future is not None: return
        size=self.grid_size.value()
        angle=float(self.resolution.currentData())
        days=self.long_horizon.value()
        self.long_button.setEnabled(False)
        self.long_status.setText(f"Расчёт N={size}, {days:g} сут. выполняется…")
        self.analysis_future=self.analysis_executor.submit(run_selected_case,size,angle,days,output_parent)
        self.analysis_timer.start()

    def _poll_long_analysis(self):
        if self.analysis_future is None or not self.analysis_future.done(): return
        future=self.analysis_future
        self.analysis_future=None
        self.analysis_timer.stop()
        self.long_button.setEnabled(self.result is not None)
        try:
            out=future.result()
            self.long_status.setText(f"{out['status']}; подходящих пролётов {out['qualifying_passes']}; min {out['closest_pair']['distance_km']:.3f} км; файлы: {out['folder']}")
        except Exception as exc:
            self.long_status.setText(f"Отдельный анализ не завершён: {exc}")

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
        self.observable_badge.setText(f"{'● Вся сетка приемлема сейчас' if good else '● Сетка сейчас неприемлема: '+obs['reasons'][k]}. Видимость всех КА: {vis_text}. Подходящий интервал пролёта: {interval}.")
        self.observable_badge.setStyleSheet("background:#d9f4e3;color:#145b32;padding:5px;" if good else "background:#ffe3df;color:#8b241d;padding:5px;")
        audit=self.study_record["verification"]
        closest=audit["closest_pair"]
        threshold=audit["absolute_minimum_requirement_km"]
        self.safety_badge.setText(f"Минимум между КА сейчас: {r.min_pair_distance_km[k]:.3f} км. Минимум за проверенный интервал: {closest['distance_km']:.3f} км, пара {closest['pair']}, t={closest['t_s']:.1f} с, {closest['latitude_deg']:.1f}° / {closest['longitude_deg']:.1f}°. Порог: размеры КА + принятый запас = {threshold:g} км (условный). Горизонт проверки: {audit['horizon_days']:g} сут. от эпохи, плюс начало первого пролёта; все {audit['pair_count']} пар.")
        self.safety_badge.setStyleSheet("background:#e7edf8;color:#1c3764;padding:5px;" if closest["distance_km"]>=threshold else "background:#fff0d9;color:#805015;padding:5px;")
        gm=r.metadata["grid_metrics"]
        errors=r.shape_errors[k]
        self.metrics.setText(f"Внешняя сторона: {obs['outer_side_deg'][k]:.3f}° (цель {self.angular_side.value():g}°); min истинный угол соседей: {gm['min_true_neighbor_angle_deg'][k]:.3f}°; min угол места: {obs['min_elevation_deg'][k]:.2f}°. Ошибки формы: εb={errors[0]:.3f}, εd={errors[1]:.3f}, εγ={np.rad2deg(errors[2]):.2f}°; отклонение узлов {gm['point_error_steps'][k]:.3f} шага, неравномерность {gm['neighbor_step_error'][k]:.3f}.")
        sp=r.metadata["spatial_size_metrics"]
        corners=", ".join(f"{x:.2f}" for x in sp["corner_sides_km"][k])
        diagonals=", ".join(f"{x:.2f}" for x in sp["corner_diagonals_km"][k])
        self.geometry.setText(f"В пространстве: соседние стороны {sp['min_neighbor_km'][k]:.2f}…{sp['max_neighbor_km'][k]:.2f} км; стороны между крайними углами [{corners}] км; диагонали крайних углов [{diagonals}] км; max любой пары {sp['max_pair_km'][k]:.2f} км.")
        self.note.setText("Внешняя угловая сторона = atan(средней длины 4 сторон крайнего контура на гномонической плоскости). Расстояния в км относятся к КА между собой; дальность до наблюдателя сюда не входит. Центр локальной системы — расчётная точка, не дополнительный КА.")

    def _draw_grid(self,canvas,points,target,all_points,title,xlabel,ylabel):
        ax=canvas.axes
        ax.clear()
        ids=np.asarray(self.result.metadata["grid_indices"])
        size=len(ids)
        colors=plt.get_cmap("viridis",size)
        for row in range(size):
            ax.plot(*target[ids[row]].T,"--",color="#a7a7a7",linewidth=.7,alpha=.7)
            ax.plot(*points[ids[row]].T,color="#165f84",linewidth=1.1,alpha=.85)
        for col in range(size):
            ax.plot(*target[ids[:,col]].T,"--",color="#a7a7a7",linewidth=.7,alpha=.7)
            ax.plot(*points[ids[:,col]].T,color="#bd6a24",linewidth=1.1,alpha=.85)
        rgba=np.empty((size*size,4))
        for row in range(size): rgba[ids[row]]=colors(row)
        ax.scatter(points[:,0],points[:,1],c=rgba,s=19,zorder=3)
        for j in self.result.metadata["corner_indices"]:
            if np.isfinite(points[j]).all():
                ax.annotate(str(j),points[j],xytext=(4,4),textcoords="offset points",fontsize=8)
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
        canvas.draw_idle()

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
        ax.clear()
        mode=self.view.currentData()
        if mode=="local":
            xyz=r.rho_oso_km
            low=np.min(xyz,axis=(0,1)); high=np.max(xyz,axis=(0,1))
            center=(low+high)/2
            radius=max(1.,float(np.max(high-low))*.575)
            bounds=[(c-radius,c+radius) for c in center]
            labels=("x, км","y, км","z, км")
            ax.set_title("Локальная система: центр группы")
        else:
            xyz=r.r_eci_km
            labels=("X ECI, км","Y ECI, км","Z ECI, км")
            if mode=="eci_close":
                center=xyz[k,0]
                radius=max(20.,float(np.max(np.linalg.norm(xyz-xyz[:,0:1],axis=2)))*.75)
                bounds=[(c-radius,c+radius) for c in center]
                ax.set_title("ECI: приближение к КА 0")
            else:
                re=r.metadata["orbit_radius_km"]
                u=np.linspace(0,2*np.pi,25); v=np.linspace(0,np.pi,13)
                earth=np.array([re*np.outer(np.cos(u),np.sin(v)),
                    re*np.outer(np.sin(u),np.sin(v)),re*np.outer(np.ones_like(u),np.cos(v))])
                ax.plot_surface(*earth,color="#9fc4e9",alpha=.38,linewidth=0)
                ax.plot(*r.reference_orbit_eci_km.T,color="#888",linewidth=.9)
                limit=re+r.metadata["reference_altitude_km"]+150
                bounds=[(-limit,limit)]*3
                ax.set_title("ECI: общий вид с Землёй")
        ids=np.asarray(r.metadata["grid_indices"])
        size=len(ids)
        corners=r.metadata["corner_indices"]
        selected=list(dict.fromkeys(corners+[int(ids[size//2,size//2])]))
        for j in selected:
            start=max(0,k-40)
            ax.plot(*xyz[start:k+1,j].T,color="#165f84",linewidth=1.4)
        rows=np.empty(size*size,dtype=int)
        for row in range(size): rows[ids[row]]=row
        ax.scatter(*xyz[k].T,c=rows,cmap="viridis",s=28 if mode=="local" else 18,depthshade=False)
        for j in corners: ax.text(*xyz[k,j],f" {j}",fontsize=8)
        ax.set(xlim=bounds[0],ylim=bounds[1],zlim=bounds[2],
               xlabel=labels[0],ylabel=labels[1],zlabel=labels[2])
        ax.set_box_aspect((1,1,1))
        self.space.draw_idle()

    def closeEvent(self,event):
        self.analysis_timer.stop()
        self.analysis_executor.shutdown(wait=False,cancel_futures=True)
        super().closeEvent(event)


def run():
    app=QApplication.instance() or QApplication([])
    plt.rcParams["font.family"]="DejaVu Sans"
    window=DemoWindow()
    window.show()
    return app.exec()
