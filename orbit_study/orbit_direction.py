"""Сонаправленность абсолютных орбит относительно КА 0; км, км/с, градусы."""
from __future__ import annotations

import numpy as np


MAX_NORMAL_ANGLE_DEG = 5.0


def audit_orbit_normals(states, times_s, limit_deg=MAX_NORMAL_ANGLE_DEG):
    """Проверить h_i=r_i×v_i на одних и тех же возрастающих временах.

    ``states`` имеет размер (T,S,2,3), ``times_s`` — (T,). Ни знак
    наклонения, ни близость чисел i сами по себе не определяют направление.
    """
    values=np.asarray(states,dtype=float)
    times=np.asarray(times_s,dtype=float)
    if values.ndim!=4 or values.shape[2:]!=(2,3) or values.shape[0]!=len(times):
        raise ValueError("Для проверки нормалей нужны состояния (T,S,2,3) и времена (T,)")
    if not len(times) or np.any(np.diff(times)<=0):
        raise ValueError("Время проверки нормалей должно строго возрастать")
    h=np.cross(values[:,:,0],values[:,:,1])
    norms=np.linalg.norm(h,axis=2)
    if not np.all(np.isfinite(norms)) or np.any(norms<=0):
        return {"ok":False,"limit_deg":float(limit_deg),"max_angle_deg":None,
                "min_dot":None,"reason":"нулевая или неопределённая нормаль орбиты"}
    unit=h/norms[:,:,None]
    dot=np.einsum("tsd,td->ts",unit,unit[:,0])
    sine=np.linalg.norm(np.cross(unit,unit[:,0,None,:]),axis=2)
    angles=np.rad2deg(np.arctan2(sine,dot))
    index=np.unravel_index(np.argmax(angles),angles.shape)
    minimum=float(np.min(dot))
    maximum=float(angles[index])
    ok=bool(minimum>0 and maximum<=limit_deg)
    return {"ok":ok,"limit_deg":float(limit_deg),"max_angle_deg":maximum,
            "min_dot":minimum,"max_angle_time_s":float(times[index[0]]),
            "max_angle_spacecraft":int(index[1]),
            "reason":("" if ok else
                f"нормали орбит: max {maximum:.6f}° > {limit_deg:g}° или h_i·h_0 ≤ 0")}
