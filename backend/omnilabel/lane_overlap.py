"""Estimate a fixed partial lane correspondence from the initial projection."""
from __future__ import annotations

import numpy as np

from .lane_optimizer import resample
from .projection_calibration import project_camera_points


def match_overlap(observation: dict, camera: dict, matrix, max_distance_px=60) -> dict:
    source = np.asarray(observation['world_points'], dtype=np.float64)
    total = float(np.linalg.norm(np.diff(source, axis=0), axis=1).sum())
    start, end = observation.get('world_range', [0, 1])
    world = resample(source, 4097, start, end)
    pose = np.asarray(observation['world_from_lidar'], dtype=np.float64)
    local = (world-pose[:3, 3]) @ pose[:3, :3]
    uv, depth = project_camera_points(local, np.asarray(matrix), np.asarray(camera['intrinsics']), np.asarray(camera.get('distortion', [])))
    valid = np.isfinite(uv).all(1) & (depth > .05) & (uv[:,0] >= 0) & (uv[:,0] < camera['width']) & (uv[:,1] >= 0) & (uv[:,1] < camera['height'])
    edges = np.flatnonzero(valid[:-1] & valid[1:] & (np.linalg.norm(np.diff(uv, axis=0), axis=1) > 1e-8))
    if len(edges) < 2: raise ValueError('当前初始投影中没有足够的连续可见三维线段，请先检查外参和位姿。')
    image = resample(observation['image_points'], 64)
    if np.linalg.norm(np.diff(image, axis=0), axis=1).sum() < 4:
        raise ValueError('图像线过短（小于 4 像素），无法可靠自动匹配，请选取更长的可见线段或更近的帧。')
    a, direction = uv[edges], uv[edges+1]-uv[edges]
    alpha = np.clip(((image[:,None]-a)*direction).sum(2)/(direction**2).sum(1), 0, 1)
    distance = np.linalg.norm(image[:,None]-(a+alpha[:,:,None]*direction), axis=2)
    best = distance.argmin(1)
    errors = distance[np.arange(len(image)), best]
    if np.quantile(errors, .9) > max_distance_px:
        raise ValueError('图像线与初始投影相距过大，无法可靠自动匹配；请检查配对、外参或调整匹配距离。')
    # A screen-space interpolation fraction is not a metric fraction along a
    # perspective-projected 3D segment. Recover it using endpoint depths.
    t = alpha[np.arange(len(image)), best]
    segment = edges[best]
    t = t*depth[segment]/((1-t)*depth[segment+1]+t*depth[segment])
    stations = (start+(end-start)*(segment+t)/4096)*total
    low, high = float(stations.min()), float(stations.max())
    span = high-low
    if span < .05: raise ValueError('自动匹配区间不足 5 厘米，请检查图像线和三维线是否对应。')
    # Reject order reversals/loops rather than joining separate visible pieces.
    delta = np.diff(stations)
    direction_sign = 1 if stations[-1] >= stations[0] else -1
    if np.maximum(-direction_sign*delta, 0).sum() > .1*span:
        raise ValueError('存在不连续或往返的匹配区间，请缩小三维取样范围或重新检查配对。')
    if not np.all(valid[int(segment.min()):int(segment.max())+2]):
        raise ValueError('匹配区间跨越不可见区域，请先缩小三维取样范围。')
    # A folded map can project several distant road segments onto the same line.
    edge_stations = (start+(end-start)*(edges+.5)/4096)*total
    far = np.abs(edge_stations[None,:]-stations[:,None]) > max(span*1.5, total*.05, 1)
    alternate = np.where(far, distance, np.inf).min(1)
    if np.mean(alternate <= errors+2) > .25:
        raise ValueError('长线中有多个相似投影区间，自动匹配有歧义；请先缩小三维取样范围。')
    return {'id': observation['id'], 'world_range': [low/total, high/total],
        'start_m': low, 'end_m': high, 'length_m': span, 'total_length_m': total,
        'mean_distance_px': float(errors.mean()), 'p90_distance_px': float(np.quantile(errors, .9))}
