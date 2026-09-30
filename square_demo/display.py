"""Read-only presentation of c910c70 trajectories; no dynamics or new criteria.

The provider result remains in the chief's frame. Only display coordinates
are translated to the centroid. Arrow directions are finite-difference
velocities of the displayed samples, in km/s, not control accelerations.
"""
from dataclasses import dataclass, replace

import numpy as np

from .core import observability, visibility_summary
from .grid import grid_indices


@dataclass
class DisplayData:
    local_km: np.ndarray
    local_velocity_km_s: np.ndarray
    eci_velocity_km_s: np.ndarray
    center_eci_km: np.ndarray
    image_deg: np.ndarray
    initial_local_yoz_km: np.ndarray
    outer_observability: dict
    outer_quality: dict
    spatial: dict
    indices: np.ndarray
    corners: list


def prepare_display(result):
    """Translate copies, reuse the old outer-square predicate unchanged."""
    n=result.metadata.get('grid_size',2)
    ids=grid_indices(n)
    corners=result.metadata.get('corner_indices',[0,1,2,3])
    local=result.rho_oso_km-result.rho_oso_km.mean(axis=1,keepdims=True)
    outer_meta={k:v for k,v in result.metadata.items() if k not in ('grid_metrics','grid_size')}
    outer_meta['corner_indices']=[0,1,2,3]
    outer=replace(result,image_gnomonic=result.image_gnomonic[:,corners],
        image_valid=result.image_valid[:,corners],elevation_deg=result.elevation_deg[:,corners],
        target_image_gnomonic=result.target_image_gnomonic[corners],metadata=outer_meta)
    outer_obs=observability(outer)
    outer_meta['observability']=outer_obs
    outer_quality=visibility_summary(outer)
    points=result.r_eci_km[:,ids]
    neighbor=np.concatenate((np.linalg.norm(points[:,:,1:]-points[:,:,:-1],axis=3).reshape(len(points),-1),
        np.linalg.norm(points[:,1:]-points[:,:-1],axis=3).reshape(len(points),-1)),axis=1)
    edge=result.r_eci_km[:,corners]
    sides=np.linalg.norm(np.roll(edge,-1,axis=1)-edge,axis=2)
    diagonals=np.stack((np.linalg.norm(edge[:,2]-edge[:,0],axis=1),
                        np.linalg.norm(edge[:,3]-edge[:,1],axis=1)),axis=1)
    max_pair=np.zeros(len(points))
    for i in range(result.r_eci_km.shape[1]-1):
        max_pair=np.maximum(max_pair,np.linalg.norm(result.r_eci_km[:,i+1:]-result.r_eci_km[:,i:i+1],axis=2).max(axis=1))
    epoch=int(np.argmin(abs(result.t_s-result.metadata['zenith_s'])))
    return DisplayData(local,np.gradient(local,result.t_s,axis=0,edge_order=2),
        np.gradient(result.r_eci_km,result.t_s,axis=0,edge_order=2),
        result.r_eci_km.mean(axis=1),np.rad2deg(np.arctan(result.image_gnomonic)),
        local[epoch,:,1:].copy(),outer_obs,outer_quality,
        dict(min_neighbor_km=neighbor.min(axis=1),max_neighbor_km=neighbor.max(axis=1),
             corner_sides_km=sides,corner_diagonals_km=diagonals,max_pair_km=max_pair),ids,list(corners))
