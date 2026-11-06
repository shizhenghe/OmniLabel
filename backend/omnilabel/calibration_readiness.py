"""Read-only diagnostics for one real PJLab image/scan pair.

Numerical checks cannot establish sensor synchronization, measured intensity
provenance, or the physical correctness of imported calibration values.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .models import rigid_transform
from .pjlab_runtime import mask_command, pjlab_status, validate_mask
from .projection_calibration import _image_path, document_for


def inspect_time_basis(clip, frame, has_scan: bool) -> dict:
    """Never subtract an epoch scan time from clip-relative playback time.

    For old 4D imports only the relative sampling intervals can be checked.
    Removing an inferred offset is NOT an estimate of image/LiDAR latency.
    """
    result = dict(status='warning', time_basis='unverified', nominal_time_delta_ms=None,
                  relative_interval_residual_ms=None,
                  detail='没有可用的独立扫描时间记录，无法核验图像与点云的时间关系。',
                  action='当前格式没有逐相机曝光时间，名义时间差为零也不能证明物理同步；请核对采集时间戳。PJLab 不估计时间偏移或运动补偿。')
    if not np.isfinite(frame.timestamp) or (frame.calibration_timestamp is not None and not np.isfinite(frame.calibration_timestamp)):
        result.update(status='missing', detail='时间记录包含无效数值。')
        return result
    if not has_scan or frame.calibration_timestamp is None:
        return result
    t, scan = frame.timestamp, frame.calibration_timestamp
    mixed = abs(scan-t) > 100_000_000 and min(abs(t), abs(scan)) < 100_000_000
    if mixed:
        known_import = clip.source == '4d-clip-import' and abs(t) < 100_000_000 and scan > 100_000_000
        result.update(time_basis='clip_relative_vs_scan_epoch' if known_import else 'incomparable',
                      detail=(f'帧时间 {t:.6f}s 为 clip 相对时间，扫描时间 {scan:.6f}s 为 Unix 绝对时间。两者起点不同，不能直接相减；这不表示扫描缺失或存在数十年的不同步。'
                              if known_import else f'帧时间 {t:.6f}s 与扫描时间 {scan:.6f}s 疑似使用不同时间基准，当前不计算时间差。'),
                      action='尚无图像曝光时间，无法测量真实图像/扫描延迟。可先试运行并复核投影，不要把推测的时间原点当成同步校正。')
        # Report only interval consistency, explicitly NOT an absolute offset.
        anchor = next((f for f in clip.frames if f.calibration_pointcloud_file and f.calibration_timestamp is not None
                       and np.isfinite(f.calibration_timestamp) and np.isfinite(f.timestamp)), None)
        if known_import and anchor is not None and anchor is not frame:
            residual = ((scan-anchor.calibration_timestamp) - (t-anchor.timestamp))*1000
            if np.isfinite(residual):
                result['relative_interval_residual_ms'] = residual
                result['detail'] += f' 相对于首个可用扫描的帧间隔残差为 {residual:.3f}ms；只反映间隔一致性，不证明曝光同步。'
        return result
    result.update(time_basis='declared_comparable', nominal_time_delta_ms=(scan-t)*1000,
                  detail=f'帧声明时间 {t:.6f}s，扫描声明时间 {scan:.6f}s，按已声明时间基准的名义差 {(scan-t)*1000:.3f}ms；平台没有独立曝光时间可验证该声明。')
    return result


def inspect_pjlab_inputs(store, clip, frame, camera, mask_path: Path) -> dict:
    checks = []
    def add(key, title, status, detail, action=''):
        checks.append(dict(key=key, title=title, status=status, detail=detail, action=action))

    runtime = pjlab_status()
    add('runtime', 'PJLab 运行入口', 'ready' if runtime['available'] else 'missing', runtime['detail'],
        '' if runtime['available'] else '在平台后端所在机器运行 bash scripts/setup-pjlab.sh，设置 OMNILABEL_PJLAB_BINARY 后重启后端。')
    image_path = None
    try:
        image_path = _image_path(store, frame, camera.id)
        with Image.open(image_path) as image:
            if image.size != (camera.width, camera.height):
                raise ValueError(f'原图尺寸 {image.width}×{image.height} 与相机声明 {camera.width}×{camera.height} 不一致。')
            image.verify()
        add('image', '原始图像', 'ready', f'{camera.width} × {camera.height}，已找到当前帧 {camera.id} 的本地图像。')
    except (ValueError, OSError, Image.DecompressionBombError) as exc:
        add('image', '原始图像', 'missing', str(exc), '重新导入该相机的原始图像，保持尺寸和像素坐标，不预先缩放或去畸变。')

    k = np.asarray(camera.intrinsics, dtype=float)
    add('intrinsics', '相机内参 K', 'ready', f'已导入，fx={k[0,0]:.3f}、fy={k[1,1]:.3f}、cx={k[0,2]:.3f}、cy={k[1,2]:.3f}。数值格式有效，物理准确性需核对采集标定。')
    distortion = np.asarray(camera.distortion, dtype=float)
    if not np.isfinite(distortion).all():
        add('distortion', '畸变参数', 'missing', '畸变参数包含无效数值。', '重新导入真实相机畸变参数。')
    elif not len(distortion):
        add('distortion', '畸变参数', 'warning', '未提供畸变参数，运行时按零畸变处理。', '原始相机若有畸变，请重新导入 [k1,k2,p1,p2,k3]；不要凭空填写。')
    else:
        add('distortion', '畸变参数', 'ready',
            '已导入 [k1,k2,p1,p2,k3]：' + ', '.join(f'{v:.6g}' for v in distortion),
            '' if np.any(distortion) else '这是已提供的零畸变模型，适用于已去畸变/矫正图像；请使用配套的矫正后内参 K。未经矫正的原图需导入真实畸变参数。')
    try:
        document = document_for(store, clip.id, frame.id, camera.id, context=(clip, frame, camera))
        rigid_transform(document['effective_matrix'])
        add('initial', '当前有效初始外参', 'ready', '已找到有效刚体矩阵，算法会使用当前已保存修正或导入外参作为初值。',
            '先查看当前图像的点云投影；偏差很大时使用逐帧校准调整，再保存初值。矩阵格式正确不等于物理对齐。')
    except (ValueError, OSError, KeyError) as exc:
        add('initial', '当前有效初始外参', 'missing', str(exc), '重新导入有效 LiDAR/相机外参，或先保存当前帧的手动校准。')

    raw = frame.calibration_pointcloud_file is not None
    source = 'synchronized_sensor_scan' if raw else 'primary_point_cloud'
    count, intensity_min, intensity_max = 0, None, None
    try:
        points = store.calibration_point_array(frame) if raw else store.frame_point_array(frame)
        if points.ndim != 2 or points.shape[1] != 4 or not len(points):
            raise ValueError('没有有效的 N×4 XYZI 点云。')
        count = len(points)
        low, high = float('inf'), float('-inf')
        # Inspect the entire scan in bounded temporary chunks; no point quota.
        for offset in range(0, count, 262144):
            chunk = points[offset:offset+262144]
            if not np.isfinite(chunk).all():
                raise ValueError('点云坐标或反射强度包含 NaN/Inf。')
            low, high = min(low, float(chunk[:,3].min())), max(high, float(chunk[:,3].max()))
        intensity_min, intensity_max = low, high
        add('scan', '标定点云来源', 'ready' if raw else 'warning',
            f'{count:,} 点，' + ('使用独立的同步单帧扫描，不使用融合地图替代。' if raw else '当前回退使用主点云，未提供独立同步扫描。'),
            '' if raw else '若主点云是融合地图，请用 scripts/import_4d_clip.py 的 --calibration-pointcloud-dir 和 --install-calibration-scans 给现有 clip 补齐原始扫描及其位姿、时间。')
        add('intensity', 'XYZI 反射强度', 'ready' if high > low else 'missing', f'完整扫描检查：最小值 {low:.6g}，最大值 {high:.6g}。',
            '数值有变化，但来源是否为实测强度需核对采集文件；[0,1] 强度会自动映射到 [0,255]。' if high > low else '强度恒定，PJLab 无法提取可靠道路标线。请导入真实 intensity，不能用高度、颜色或常数代替。')
    except (ValueError, OSError, KeyError) as exc:
        add('scan', '标定点云与反射强度', 'missing', str(exc), '重新导入带真实 intensity 的当前帧原始扫描，并提供其位姿。')
    timing = inspect_time_basis(clip, frame, raw)
    delta_ms = timing['nominal_time_delta_ms']
    add('synchronization', '采集时刻与时间基准', timing['status'], timing['detail'], timing['action'])

    try:
        if mask_path.is_file():
            if image_path is None:
                raise ValueError('缺少原图，无法核对掩码来源。')
            stat = image_path.stat()
            expected = dict(path=str(image_path), inode=stat.st_ino, size=stat.st_size, mtime_ns=stat.st_mtime_ns)
            metadata = json.loads(mask_path.with_suffix('.json').read_text())
            if metadata.get('image') != expected:
                raise ValueError('掩码对应的原图已变化。')
            validate_mask(mask_path.read_bytes(), camera)
            add('mask', '当前帧线特征掩码', 'ready', '已保存同尺寸掩码，原图版本匹配；' + ('来源 SAM2。' if metadata.get('source') == 'sam2' else '来源上传。'),
                '确认白色只包含道路标线或杆体，不是整片路面。')
        elif mask_command() is not None:
            add('mask', '当前帧线特征掩码', 'warning', '已配置自动掩码命令，运行时将生成并验证。', '自动生成命令是否成功需实际执行验证。')
        else:
            add('mask', '当前帧线特征掩码', 'missing', '尚未保存这张图像的线特征掩码。', '在下方调用 SAM2，勾选标线/杆体区域后保存掩码；也可上传原图尺寸的黑底白色 PNG。')
    except (ValueError, OSError, KeyError) as exc:
        add('mask', '当前帧线特征掩码', 'missing', str(exc), '使用 SAM2 重新生成并保存当前帧掩码，或重新上传对应原图的 PNG。')
    if clip.source == 'synthetic' or clip.id == 'demo':
        add('dataset', '数据用途', 'warning', '当前为合成演示数据，不代表真实传感器标定效果。', '实际使用请切换到已导入的真实 clip。')
    return dict(clip_id=clip.id, frame_id=frame.id, camera_id=camera.id,
                ready_to_run=not any(c['status']=='missing' for c in checks),
                checks=checks, input_source=source, point_count=count,
                intensity_min=intensity_min, intensity_max=intensity_max, nominal_time_delta_ms=delta_ms, time_basis=timing['time_basis'],
                relative_interval_residual_ms=timing['relative_interval_residual_ms'],
                note='检查的是输入可用性，不证明真实同步、初值准确或最终标定精度；启动时会重新核验数据。')
