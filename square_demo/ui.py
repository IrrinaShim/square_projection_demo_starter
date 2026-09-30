"""One-pass UI for the unchanged c910c70 J2/CSV providers."""
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
from PySide6.QtWidgets import (QApplication,QComboBox,QDoubleSpinBox,QFileDialog,
    QFormLayout,QGroupBox,QHBoxLayout,QLabel,QMainWindow,QPushButton,QSlider,
    QSpinBox,QSplitter,QTableWidget,QTableWidgetItem,QToolButton,QVBoxLayout,QWidget)

from .providers import SCENARIOS,load_scenario,calculate,audit_independent_grid
from .display import prepare_display

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
        self.setWindowTitle('Наблюдаемая сетка КА — J₂ — расчёт c910c70')
        self.resize(1530,900)
        self.result=None;self.display=None;self.scenario=None
        self._grid_artists={};self._space_artists=None
        self._generation=0;self._loading=False
        self._executor=ThreadPoolExecutor(max_workers=2)
        self._calculation=None;self._audit=None
        self.timer=QTimer(self);self.timer.setInterval(110);self.timer.timeout.connect(self.advance)
        self._poll=QTimer(self);self._poll.setInterval(80);self._poll.timeout.connect(self._collect)
        self._poll.start()
        central=QWidget();self.setCentralWidget(central);root=QVBoxLayout(central)
        top=QHBoxLayout();root.addLayout(top)
        self.mode=QComboBox();self.mode.addItem('Без управления','independent_j2');self.mode.addItem('С управлением','legacy_replay')
        top.addWidget(QLabel('Режим:'));top.addWidget(self.mode,1)
        self.view=QComboBox()
        for title,key in (('3D: локальная система центра группы','local'),
            ('3D: инерциальная система — приблизить группу','eci_close'),
            ('3D: инерциальная система — Земля целиком','eci'),
            ('3D: инерциальная система — неподвижная камера','eci_fixed')):
            self.view.addItem(title,key)
        self.view.currentIndexChanged.connect(self.update_frame);top.addWidget(self.view,1)
        groups=QHBoxLayout();root.addLayout(groups)
        forms=[]
        for name in ('Конфигурация','Наблюдатель','Орбита'):
            group=QGroupBox(name);form=QFormLayout(group);groups.addWidget(group,1);forms.append(form)
        cf,obs,orb=forms
        self.grid_size=QSpinBox();self.grid_size.setRange(2,10);self.grid_size.setValue(2)
        cf.addRow('КА на стороне N',self.grid_size)
        self.fields={}
        for key,label,lo,hi,decimals,form in (
            ('angular_side_deg','Исходная сторона, °',0,20,9,cf),
            ('latitude_deg','Широта, °',-90,90,4,obs),
            ('longitude_deg','Долгота, °',-180,180,4,obs),
            ('altitude_km','Высота КА 0, км',0,100000,2,orb),
            ('inclination_deg','Наклонение, °',0,180,3,orb)):
            field=QDoubleSpinBox();field.setRange(lo,hi);field.setDecimals(decimals)
            self.fields[key]=field;form.addRow(label,field)
        for key in ('angular_side_deg','altitude_km','inclination_deg'):
            self.fields[key].setEnabled(False)
            self.fields[key].setToolTip('Задано исходными орбитами/CSV. Автоматического подбора новых угловых орбит в этой ветке нет.')
        self.elevation_label=QLabel();obs.addRow('Минимальный угол места',self.elevation_label)
        self.epoch_label=QLabel();self.epoch_label.setWordWrap(True);orb.addRow('Эпоха',self.epoch_label)
        self.period_label=QLabel('—');orb.addRow('Период КА 0',self.period_label)
        self.model_summary=QLabel();self.model_summary.setWordWrap(True)
        model=QGroupBox('Модель');ml=QVBoxLayout(model);ml.addWidget(self.model_summary);groups.addWidget(model,1)
        self.input_note=QLabel();self.input_note.setWordWrap(True);root.addWidget(self.input_note)
        self.status=QLabel();self.status.setWordWrap(True);root.addWidget(self.status)
        actions=QHBoxLayout();root.addLayout(actions)
        self.calculate_button=QPushButton('Рассчитать');self.calculate_button.clicked.connect(self.calculate_current)
        actions.addWidget(self.calculate_button)
        self.elements_button=QToolButton();self.elements_button.setText('Элементы орбит ▸');self.elements_button.setCheckable(True)
        self.elements_button.toggled.connect(lambda on:self.elements_table.setVisible(on and self.result is not None and 'orbital_elements' in self.result.metadata))
        actions.addWidget(self.elements_button)
        self.export_elements_button=QPushButton('Сохранить элементы JSON');self.export_elements_button.clicked.connect(self._choose_elements_path)
        actions.addWidget(self.export_elements_button);actions.addStretch()
        self.elements_table=QTableWidget(0,7);self.elements_table.setHorizontalHeaderLabels(['КА','a, км','e','i, °','Ω, °','ω, °','ν, °'])
        self.elements_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.elements_table.setFixedHeight(155);self.elements_table.hide();root.addWidget(self.elements_table)
        splitter=QSplitter(Qt.Orientation.Horizontal)
        self.proj=Plot();self.yoz=Plot();self.space=Plot('3d')
        for canvas in (self.proj,self.yoz,self.space):splitter.addWidget(canvas)
        splitter.setSizes([500,500,500]);root.addWidget(splitter,1)
        transport=QHBoxLayout();root.addLayout(transport)
        self.start_button=QPushButton('В начало');self.start_button.clicked.connect(lambda:self.time_slider.setValue(0))
        self.play_button=QPushButton('Пуск');self.play_button.clicked.connect(self.toggle_play)
        transport.addWidget(self.start_button);transport.addWidget(self.play_button)
        self.time_slider=QSlider(Qt.Orientation.Horizontal);self.time_slider.valueChanged.connect(self.update_frame)
        self.timeline=ObservableTimeline();self.timeline.setToolTip('Зелёный: исходные критерии всей сетки, включая угловой шаг 0,1° в J₂-сценарии.')
        tc=QVBoxLayout();tc.addWidget(self.time_slider);tc.addWidget(self.timeline);transport.addLayout(tc,1)
        self.time_label=QLabel('t = — с');transport.addWidget(self.time_label)
        self.speed=QComboBox()
        for label,value in (('0,5×',.5),('1×',1),('2×',2),('5×',5)):self.speed.addItem(label,value)
        self.speed.setCurrentIndex(1);transport.addWidget(self.speed)
        for name in ('observable_badge','pass_badge','safety_badge','metrics','spatial_badge'):
            label=QLabel();label.setWordWrap(True);setattr(self,name,label);root.addWidget(label)
        self.info=QLabel('Стрелки показывают направление производной отображаемых координат (по узлам времени); длина нормирована. ОСО центрирована только для рисунка.')
        self.info.setWordWrap(True);root.addWidget(self.info)
        for field in (self.grid_size,*self.fields.values()):field.valueChanged.connect(self._parameters_changed)
        self.mode.currentIndexChanged.connect(self.reload_scenario)
        self.reload_scenario()
        QTimer.singleShot(0,self.calculate_current)

    def _clear_result(self,message):
        self.timer.stop();self.play_button.setText('Пуск')
        self.result=None;self.display=None;self._grid_artists={};self._space_artists=None
        for canvas in (self.proj,self.yoz,self.space):
            canvas._animated_artists=[];canvas._background=None;canvas.axes.clear();canvas.draw_idle()
        self.time_slider.setEnabled(False);self.play_button.setEnabled(False)
        self.elements_button.setEnabled(False);self.export_elements_button.setEnabled(False);self.elements_table.hide()
        self.timeline.times=None;self.timeline.update();self.time_label.setText('t = — с')
        for label in (self.observable_badge,self.pass_badge,self.safety_badge,self.metrics,self.spatial_badge):label.clear()
        self.status.setText(message);self.status.setStyleSheet('background:#fff0d9;color:#805015;padding:5px;')

    def _parameters_changed(self,*_):
        if self._loading:return
        self._generation+=1
        self._clear_result('Параметры изменены. Нажмите «Рассчитать»; прежняя траектория снята.')

    def reload_scenario(self,*_):
        self._loading=True
        try:
            self.scenario=load_scenario(SCENARIOS[self.mode.currentData()]);s=self.scenario
            if s['mode']=='independent_j2':
                orbit=s['orbit'];side=s['angular_side_deg']
                self.epoch_label.setText(s['epoch_utc'])
                self.model_summary.setText('Независимые орбиты с J₂, без атмосферного сопротивления. Четыре исходных угла; дополнительные КА распространяются независимо. Тяга отсутствует.')
                self.input_note.setText('Сторона, высота, наклонение и эпоха заданы четырьмя орбитами из independent_j2_4sat.json. Подбор для других значений здесь не реализован. Исходные критерии сохранены; допуск ±10% на внешний размер не добавлен.')
            else:
                path=Path(s['_path']).parent/s['data_directory']/'config.json'
                old=json.loads(path.read_text());o=old['orbit']
                orbit=dict(altitude_km=o['h_km'],inclination_deg=o['i_deg'])
                side=float(np.rad2deg(np.arctan(old['formation']['d_m']/1000/o['h_km'])))
                self.epoch_label.setText('Зенит t=600 с — допущение')
                self.model_summary.setText('С управлением: воспроизведение исторических CSV только для четырёх КА. Это отдельная реконструкция, не управляемое решение независимой J₂-сетки.')
                self.input_note.setText('Доступен только N=2, пространственная сторона 20 км. Для N>2 управляемый случай не рассчитан. Положение наблюдателя используется только при реконструкции проекции.')
            self.fields['angular_side_deg'].setValue(side)
            for key in ('altitude_km','inclination_deg'):self.fields[key].setValue(orbit[key])
            for key in ('latitude_deg','longitude_deg'):self.fields[key].setValue(s['observer'][key])
            self.elevation_label.setText(f"{s['visibility']['min_elevation_deg']:g}° (из JSON)")
        except Exception as exc:
            self.scenario=None;self.status.setText(str(exc))
        finally:self._loading=False
        self._parameters_changed()

    @staticmethod
    def _calculate(scenario):
        result=calculate(scenario,include_audit=False)
        return result,prepare_display(result)

    def calculate_current(self):
        if self.scenario is None:return
        self._generation+=1;generation=self._generation
        n=self.grid_size.value()
        if self.mode.currentData()=='legacy_replay' and n!=2:
            self._clear_result(f'Для N={n} случай «С управлением» не рассчитан. Исторические CSV содержат только 4 КА (N=2).')
            return
        s=deepcopy(self.scenario)
        if s['mode']=='independent_j2':s['grid_size']=n
        for key in ('latitude_deg','longitude_deg'):s['observer'][key]=self.fields[key].value()
        self._clear_result(f'Рассчитываю один пролёт, N={n}; исходная модель c910c70…')
        self._calculation=(generation,s,self._executor.submit(self._calculate,s))

    def _collect(self):
        if self._calculation is not None:
            generation,s,future=self._calculation
            if future.done():
                self._calculation=None
                if generation==self._generation:
                    try:
                        self.result,self.display=future.result();self.scenario=s
                        self._accept_result()
                        if s['mode']=='independent_j2':
                            self._audit=(generation,self._executor.submit(audit_independent_grid,
                                deepcopy(s),deepcopy(self.result.metadata['orbital_elements'])))
                    except Exception as exc:self._clear_result(f'Расчёт не выполнен: {exc}')
        if self._audit is not None:
            generation,future=self._audit
            if future.done():
                self._audit=None
                if generation==self._generation and self.result is not None:
                    try:self.result.metadata.update(future.result())
                    except Exception as exc:self.result.metadata.update(audit_status='failed',audit_error=str(exc))
                    self.update_frame()

    def _accept_result(self):
        r=self.result;m=r.metadata
        self.status.setText(f"Один пролёт: {r.r_eci_km.shape[1]} КА, {r.t_s[0]:g}…{r.t_s[-1]:g} с, шаг {np.max(np.diff(r.t_s)):g} с. {m['source']}. Исходные орбиты и критерии c910c70.")
        self.status.setStyleSheet('background:#e7edf8;color:#1c3764;padding:5px;')
        self.period_label.setText(f"{m['reference_period_s']:.1f} с")
        self.time_slider.setEnabled(True);self.play_button.setEnabled(True)
        self.time_slider.blockSignals(True);self.time_slider.setRange(0,len(r.t_s)-1)
        self.time_slider.setValue(int(np.argmin(abs(r.t_s-m['zenith_s']))));self.time_slider.blockSignals(False)
        self.timeline.set_result(r.t_s,m['observability']['eligible'])
        entries=m.get('orbital_elements',[])
        self.elements_button.setEnabled(bool(entries));self.export_elements_button.setEnabled(bool(entries))
        self.elements_table.setRowCount(len(entries))
        for row,e in enumerate(entries):
            values=[str(row)]+[f'{e[k]:.9f}' for k in ('a_km','e','inclination_deg','raan_deg','arg_perigee_deg','true_anomaly_deg')]
            for col,value in enumerate(values):self.elements_table.setItem(row,col,QTableWidgetItem(value))
        self.elements_table.setVisible(bool(entries) and self.elements_button.isChecked())
        self.update_frame()

    def _choose_elements_path(self):
        path,_=QFileDialog.getSaveFileName(self,'Сохранить элементы','j2_elements.json','JSON (*.json)')
        if path:self.export_orbit_elements(path)

    def export_orbit_elements(self,path):
        if self.result is None or 'orbital_elements' not in self.result.metadata:return
        m=self.result.metadata
        payload=dict(epoch_utc=m['epoch_utc'],grid_size=m['grid_size'],index_map=m['grid_indices'],
            model='independent_j2_without_drag',elements_at_epoch=m['orbital_elements'],
            units={'a':'km','angles':'deg','e':'dimensionless'})
        Path(path).write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')

    def toggle_play(self):
        if self.result is None:return
        if self.timer.isActive():self.timer.stop();self.play_button.setText('Пуск')
        else:self.timer.start();self.play_button.setText('Пауза')

    def advance(self):
        if self.result is None:return
        k=self.time_slider.value();t=self.result.t_s
        target=t[k]+self.timer.interval()/1000*self.speed.currentData()*10
        self.time_slider.setValue(min(max(k+1,int(np.searchsorted(t,target))),len(t)-1))
        if self.time_slider.value()==len(t)-1:self.timer.stop();self.play_button.setText('Пуск')

    @staticmethod
    def _interval(quality):
        interval=quality['longest_interval_s']
        if not quality['has_positive_duration_interval']:return 'нет положительного интервала (0 с)'
        return f"{interval[0]:g}…{interval[1]:g} с ({quality['longest_contiguous_duration_s']:g} с)"

    def update_frame(self,*_):
        r=self.result
        if r is None:return
        k=self.time_slider.value();d=self.display;m=r.metadata;obs=m['observability']
        self.time_label.setText(f't = {r.t_s[k]:.1f} с');self.timeline.set_index(k)
        self._draw_projection(k);self._draw_yoz(k);self._draw_space(k)
        good=bool(obs['eligible'][k]);outer=bool(d.outer_observability['eligible'][k])
        self.observable_badge.setText(f"Сейчас: внешний квадрат — {'да' if outer else 'нет'}; вся сетка — {'да' if good else 'нет'}. Минимальный угол места {obs['min_elevation_deg'][k]:.2f}°. "+('' if good else obs['reasons'][k]))
        self.observable_badge.setStyleSheet('background:#d9f4e3;color:#145b32;padding:5px;' if good else 'background:#ffe3df;color:#8b241d;padding:5px;')
        self.pass_badge.setText(f"Внешний квадрат (форма + видимость четырёх углов): {self._interval(d.outer_quality)}. Вся сетка (исходные критерии): {self._interval(m['quality'])}. Узлы {np.max(np.diff(r.t_s)):g} с.")
        angle=np.rad2deg(np.arctan(obs['mean_side_gnomonic'][k]));err=r.shape_errors[k]
        grid=m.get('grid_metrics')
        self.metrics.setText(f"Видимая внешняя сторона {angle:.4f}°; исходная {m['angular_side_deg']:.4f}°; отклонение размера {obs['size_deviation_percent'][k]:+.2f}% (справочно). εb={err[0]:.4f}, εd={err[1]:.4f}, εγ={np.rad2deg(err[2]):.2f}°."+(f" Истинный шаг соседей {grid['min_true_neighbor_angle_deg'][k]:.4f}°; предел {m['grid_tolerances']['min_neighbor_angle_deg']:g}°." if grid else ''))
        sp=d.spatial
        self.spatial_badge.setText(f"Расстояния: соседи {sp['min_neighbor_km'][k]:.3f}…{sp['max_neighbor_km'][k]:.3f} км; стороны между крайними КА "+', '.join(f'{x:.3f}' for x in sp['corner_sides_km'][k])+" км; диагонали "+', '.join(f'{x:.3f}' for x in sp['corner_diagonals_km'][k])+f" км; максимум любой пары {sp['max_pair_km'][k]:.3f} км.")
        threshold=m['illustrative_min_separation_km'];status=m.get('audit_status')
        safety=f"Сейчас min всех пар {r.min_pair_distance_km[k]:.3f} км; порог {threshold:g} км (условный). "
        if status=='pending':safety+='Проверка полного витка выполняется.'
        elif status=='failed':safety+='Проверка витка не завершена: '+m.get('audit_error','')
        elif 'full_orbit_audit' in m:
            audit=m['full_orbit_audit'];closest=min(audit['pair_minima'].values(),key=lambda x:x['distance_km'])
            safety+=f"Полный виток: {len(audit['pair_minima'])} пар, min {closest['distance_km']:.3f} км, пара {closest['pair']}, t={closest['t_s']:.1f} с. "+('Порог пройден.' if m['full_orbit_meets_illustrative_threshold'] else 'Есть сближение ниже порога.')
        else:safety+=f"Проверено только окно CSV: min {r.min_pair_distance_km.min():.3f} км; аудит витка отсутствует."
        self.safety_badge.setText(safety)

    def _draw_projection(self,k):
        r=self.result;d=self.display
        self._draw_grid(self.proj,d.image_deg[k],np.rad2deg(np.arctan(r.target_image_gnomonic)),
            d.image_deg.reshape(-1,2),'Гномоническая проекция','atan(ξ), ° — координатный угол','atan(η), ° — координатный угол')

    def _draw_yoz(self,k):
        d=self.display
        self._draw_grid(self.yoz,d.local_km[k,:,1:],d.initial_local_yoz_km,
            d.local_km[:,:,1:].reshape(-1,2),'Конфигурация yOz от центра группы','y, км','z, км')

    def closeEvent(self,event):
        self._generation+=1
        self._executor.shutdown(wait=False,cancel_futures=True)
        super().closeEvent(event)

    def _draw_grid(self,canvas,points,target,all_points,title,xlabel,ylabel):
        ax=canvas.axes
        ids=np.asarray(self.display.indices)
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
                textcoords="offset points",fontsize=8) for j in self.display.corners}
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

    def _draw_space(self,k):
        r=self.result
        ax=self.space.axes
        mode=self.view.currentData()
        # A short, explicitly labelled history lets the ECI camera show the
        # group at its actual kilometre scale without clipping a long arc.
        trace_count=2 if mode=="eci_close" else 40
        if self._space_artists is None or self._space_artists["mode"]!=mode:
            ax.clear(); ax.set_axis_on()
            self._space_artists={"mode":mode}
            rebuild=True
        else:
            rebuild=False
        if mode=="local":
            xyz=self.display.local_km
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
                # Fixed span: contain the chosen history and every current KA.
                if rebuild:
                    extent=0.
                    for frame in range(len(xyz)):
                        origin=xyz[frame,0] if mode=="eci_close" else center
                        extent=max(extent,float(np.max(abs(xyz[max(0,frame-trace_count):frame+1]-origin))))
                    self._space_artists["camera_radius_km"]=max(20.,extent*1.12)
                radius=self._space_artists["camera_radius_km"]
                bounds=[(c-radius,c+radius) for c in center]
                if rebuild: ax.set_title(f"ECI: за КА 0; след до {trace_count*np.max(np.diff(r.t_s)):g} с" if mode=="eci_close" else
                                         "ECI: камера неподвижна у КА 0 в эпоху",fontsize=10)
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
        ids=np.asarray(self.display.indices)
        size=len(ids)
        corners=self.display.corners
        selected=list(dict.fromkeys(corners+[int(ids[size//2,size//2])]))
        if rebuild:
            self._space_artists["traces"]={j:ax.plot([],[],[],color="#165f84",linewidth=1.4)[0]
                                           for j in selected}
        for j,line in self._space_artists["traces"].items():
            start=max(0,k-trace_count)
            line.set_data_3d(*xyz[start:k+1,j].T)
        # Each arrow is tangent to its trace in that trace's coordinate frame.
        velocity=(self.display.local_velocity_km_s if mode=="local"
                  else self.display.eci_velocity_km_s)
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


def run():
    app=QApplication.instance() or QApplication([])
    plt.rcParams["font.family"]="DejaVu Sans"
    window=DemoWindow();window.show()
    return app.exec()
