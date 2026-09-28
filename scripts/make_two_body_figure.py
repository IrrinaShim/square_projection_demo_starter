"""Рисунок расстояний для TeX: .venv/bin/python scripts/make_two_body_figure.py"""
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from two_body_check import initial_states, propagate
from square_demo.providers import SCENARIOS, load_scenario


scenario = load_scenario(SCENARIOS["free_hcw_two_body"])
initial,n = initial_states(scenario)
period = 2*np.pi/n
states,dt = propagate(initial,float(scenario["orbit"]["mu_km3_s2"]),period)
t = np.arange(len(states))*dt
colors = ("#204b8e","#dc4e41","#2b9a69","#ad6c23","#894f9b","#278f9b")
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9})
fig,(main,zoom) = plt.subplots(2,1,figsize=(7.1,5.6),layout="constrained")
for color,(i,j) in zip(colors,combinations(range(4),2)):
    d = np.linalg.norm(states[:,i,0]-states[:,j,0],axis=1)
    main.plot(t/period,d,color=color,lw=1,label=f"{i}–{j}")
    zoom.plot((t-period/4),d,color=color,lw=1)
main.set(xlim=(0,3),ylim=(0,32),xlabel="Время / T₀",ylabel="Расстояние, км",
         title="Все шесть пар: точные двухтельные орбиты")
main.legend(ncol=6,loc="upper center",frameon=False)
main.grid(alpha=.25)
zoom.set(xlim=(-60,60),ylim=(0,22),xlabel="Время относительно T₀/4, с",ylabel="Расстояние, км",
         title="Окрестность первого сближения")
zoom.grid(alpha=.25)
output = Path(__file__).resolve().parent.parent/"docs/two_body_distances.pdf"
fig.savefig(output)
print(output)
