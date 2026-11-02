import {useEffect, useRef, useState} from 'react';
import {download, jsonRequest, request} from '../api';
import {applyCalibration, calibrationWorldPoses, correctExtrinsic, correctFrameCameraExtrinsic, ZERO_CORRECTION,
  type CalibrationDocument, type ExtrinsicCorrection, type Matrix4} from '../geometry/calibration';
import type {Clip} from '../types';
import type {ProjectionCalibrationDocument} from './ProjectionCalibrationEditor';
export type {CalibrationDocument} from '../geometry/calibration';

export interface CalibrationEditorProps {
  clip: Clip;
  frameId: string;
  cameraId: string;
  disabled?: boolean;
  onPreview: (clip: Clip | null, document?: CalibrationDocument) => void;
  onSaved: (document: CalibrationDocument, options: {lidarChanged: boolean; allFrames: boolean}) => Promise<void> | void;
  onFramePreview: (matrix: Matrix4 | null, cameraId: string) => void;
  onFrameSaved: (document: ProjectionCalibrationDocument) => Promise<void> | void;
  onCameraChange: (cameraId: string) => void;
  onClose: () => void;
  onError?: (message: string) => void;
  onDirtyChange?: (dirty: boolean) => void;
  onBusyChange?: (busy: boolean) => void;
}

const message = (error: unknown) => error instanceof Error ? error.message : String(error);
const copy = <T,>(value: T): T => structuredClone(value);
const sameCalibration = (a: CalibrationDocument, b: CalibrationDocument) =>
  JSON.stringify([a.cameras, a.lidar_to_ego]) === JSON.stringify([b.cameras, b.lidar_to_ego]);

export function CalibrationEditor(props: CalibrationEditorProps) {
  const [baseline, setBaseline] = useState<CalibrationDocument | null>(null);
  const [draft, setDraft] = useState<CalibrationDocument | null>(null);
  const [target, setTarget] = useState(props.cameraId || props.clip.cameras[0]?.id || '');
  const [allFrames, setAllFrames] = useState(false);
  const [correction, setCorrection] = useState<ExtrinsicCorrection>(copy(ZERO_CORRECTION));
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [status, setStatus] = useState('');
  const [frameBaseline, setFrameBaseline] = useState<ProjectionCalibrationDocument | null>(null);
  const [frameDraft, setFrameDraft] = useState<Matrix4 | null>(null);
  const [frameLoading, setFrameLoading] = useState(false);
  const [reload, setReload] = useState(0);
  const targetBase = useRef<Record<string, Matrix4>>({});
  const originalClip = useRef(props.clip);
  const latest = useRef(props);
  latest.current = props;
  const lidarTarget = target === '__lidar__';
  const frameCamera = !lidarTarget && !allFrames;
  const frameReady = frameBaseline?.clip_id === props.clip.id && frameBaseline.frame_id === props.frameId && frameBaseline.camera_id === target;
  const dirty = frameCamera ? Boolean(frameReady && frameDraft && JSON.stringify(frameDraft) !== JSON.stringify(frameBaseline.effective_matrix))
    : Boolean(baseline && draft && !sameCalibration(baseline, draft));
  const switchLocked = Boolean(props.disabled || loading || saving);
  const locked = switchLocked || frameLoading || !baseline || (frameCamera && !frameReady);
  const endpoint = `/clips/${encodeURIComponent(props.clip.id)}/frames/${encodeURIComponent(props.frameId)}/projection-calibration/${encodeURIComponent(target)}`;

  useEffect(() => {props.onDirtyChange?.(dirty);}, [dirty, props.onDirtyChange]);
  useEffect(() => () => latest.current.onDirtyChange?.(false), []);
  useEffect(() => {latest.current.onBusyChange?.(loading || frameLoading || saving);}, [loading, frameLoading, saving]);
  useEffect(() => () => {latest.current.onBusyChange?.(false); latest.current.onPreview(null); latest.current.onFramePreview(null, target);}, []);

  useEffect(() => {
    if (target !== '__lidar__' && props.cameraId && props.cameraId !== target) {
      setTarget(props.cameraId); setAllFrames(false);
    }
  }, [props.cameraId]);

  useEffect(() => {
    let active = true;
    setLoading(true);
    request<CalibrationDocument>(`/clips/${encodeURIComponent(props.clip.id)}/calibration`).then(value => {
      if (!active) return;
      originalClip.current = latest.current.clip;
      setBaseline(value); setDraft(copy(value));
      setTarget(latest.current.cameraId || latest.current.clip.cameras[0]?.id || '');
      targetBase.current = copy(value.cameras);
      setCorrection(copy(ZERO_CORRECTION));
    }).catch(cause => {if (active) setError(`读取标定失败：${message(cause)}`);})
      .finally(() => {if (active) setLoading(false);});
    return () => {active = false;};
  }, [props.clip.id]);

  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    setFrameBaseline(null); setFrameDraft(null); setCorrection(copy(ZERO_CORRECTION));
    setError(''); setStatus(''); setFrameLoading(frameCamera);
    originalClip.current = latest.current.clip;
    latest.current.onPreview(null); latest.current.onFramePreview(null, target);
    if (frameCamera) {
      request<ProjectionCalibrationDocument>(endpoint, {signal: controller.signal}).then(value => {
        if (active) {setFrameBaseline(value); setFrameDraft(copy(value.effective_matrix));}
      }).catch(cause => {if (active) setError(`读取本帧相机外参失败：${message(cause)}`);})
        .finally(() => {if (active) setFrameLoading(false);});
    }
    return () => {active = false; controller.abort();};
  }, [endpoint, frameCamera, reload]);

  useEffect(() => {
    setDraft(baseline ? copy(baseline) : null);
    targetBase.current = copy((lidarTarget ? baseline?.lidar_to_ego : baseline?.cameras) ?? {});
    setCorrection(copy(ZERO_CORRECTION));
  }, [props.frameId, props.cameraId, target, allFrames]);

  useEffect(() => {
    if (!dirty) return;
    const guard = (event: BeforeUnloadEvent) => {event.preventDefault(); event.returnValue = '';};
    window.addEventListener('beforeunload', guard);
    return () => window.removeEventListener('beforeunload', guard);
  }, [dirty]);

  const preview = (value: CalibrationDocument) => {
    if (!baseline) return;
    const next = {...value, world_from_lidar: calibrationWorldPoses(baseline, value)};
    setDraft(next); setError(''); setStatus('');
    if (sameCalibration(baseline, next)) props.onPreview(null);
    else props.onPreview(applyCalibration(originalClip.current, next, baseline), next);
  };
  const chooseTarget = (next: string) => {
    if (!draft || (dirty && !window.confirm('切换传感器会放弃尚未保存的外参修改，确定继续？'))) return;
    resetAll(); setAllFrames(false);
    setTarget(next); setCorrection(copy(ZERO_CORRECTION));
    targetBase.current = copy(next === '__lidar__' ? baseline!.lidar_to_ego : baseline!.cameras);
    if (next !== '__lidar__') props.onCameraChange(next);
  };
  const chooseScope = (value: boolean) => {
    if (!draft || (dirty && !window.confirm('切换应用范围会放弃尚未保存的外参修改，确定继续？'))) return;
    resetAll();
    setAllFrames(value); setCorrection(copy(ZERO_CORRECTION));
    targetBase.current = copy(lidarTarget ? baseline!.lidar_to_ego : baseline!.cameras);
  };
  const update = (kind: 'translation' | 'rotation', axis: number, value: number) => {
    if (!draft || !Number.isFinite(value) || locked) return;
    const values = [...correction[kind]] as [number, number, number];
    values[axis] = value;
    const nextCorrection = {...correction, [kind]: values};
    setCorrection(nextCorrection);
    if (frameCamera && frameReady) {
      const matrix = correctFrameCameraExtrinsic(frameBaseline.effective_matrix, nextCorrection);
      setFrameDraft(matrix); setError(''); setStatus('');
      props.onFramePreview(JSON.stringify(matrix) === JSON.stringify(frameBaseline.effective_matrix) ? null : matrix, target);
    } else if (lidarTarget) {
      const matrices = {...draft.lidar_to_ego};
      for (const id of allFrames ? Object.keys(matrices) : [props.frameId]) {
        matrices[id] = correctExtrinsic(targetBase.current[id] ?? baseline!.lidar_to_ego[id], nextCorrection);
      }
      preview({...draft, lidar_to_ego: matrices});
    } else {
      preview({...draft, cameras: {...draft.cameras,
        [target]: correctExtrinsic(targetBase.current[target] ?? baseline!.cameras[target], nextCorrection)}});
    }
  };
  const resetTarget = () => {
    if (!baseline || !draft) return;
    if (frameCamera) {resetAll(); return;}
    setCorrection(copy(ZERO_CORRECTION));
    if (lidarTarget) {
      const matrices = {...draft.lidar_to_ego};
      for (const id of allFrames ? Object.keys(matrices) : [props.frameId]) matrices[id] = copy(baseline.lidar_to_ego[id]);
      targetBase.current = copy(matrices); preview({...draft, lidar_to_ego: matrices});
    } else {
      const cameras = {...draft.cameras, [target]: copy(baseline.cameras[target])};
      targetBase.current = copy(cameras); preview({...draft, cameras});
    }
  };
  const resetAll = () => {
    if (!baseline) return;
    setDraft(copy(baseline)); setCorrection(copy(ZERO_CORRECTION)); setError(''); setStatus('');
    targetBase.current = copy(lidarTarget ? baseline.lidar_to_ego : baseline.cameras);
    setFrameDraft(frameReady ? copy(frameBaseline.effective_matrix) : null);
    props.onPreview(null); props.onFramePreview(null, target);
  };
  const close = () => {
    if (saving || (dirty && !window.confirm('放弃尚未保存的外参修改？'))) return;
    props.onPreview(null); props.onFramePreview(null, target); props.onClose();
  };
  const save = async () => {
    if (!baseline || !draft || !dirty || locked) return;
    setSaving(true); setError('');
    try {
      if (frameCamera && frameReady && frameDraft) {
        const value = await jsonRequest<ProjectionCalibrationDocument>(endpoint, {
          revision: frameBaseline.revision, matrix: frameDraft, scope: 'frame', method: 'manual',
          correspondences: frameBaseline.correspondences,
        }, 'PUT');
        setFrameBaseline(value); setFrameDraft(copy(value.effective_matrix)); setCorrection(copy(ZERO_CORRECTION));
        try {await props.onFrameSaved(value);}
        catch (cause) {setError(`本帧外参已保存，但刷新失败：${message(cause)}。请重新加载场景。`);}
        props.onFramePreview(null, target);
        setStatus('已保存当前帧、当前相机外参，其他帧和相机保持不变。');
        return;
      }
      const value = await jsonRequest<CalibrationDocument>(`/clips/${encodeURIComponent(props.clip.id)}/calibration`,
        {revision: baseline.revision, cameras: draft.cameras, lidar_to_ego: draft.lidar_to_ego}, 'PUT');
      originalClip.current = applyCalibration(originalClip.current, value, value);
      setBaseline(value); setDraft(copy(value)); setCorrection(copy(ZERO_CORRECTION));
      targetBase.current = copy(lidarTarget ? value.lidar_to_ego : value.cameras);
      setStatus(value.map_rebuild_required ? '本帧 LiDAR 外参已保存；请重建场景地图并复核世界坐标标注。'
        : allFrames ? '共享安装外参已保存，所有帧投影已更新。' : '本帧 LiDAR 外参已保存，其他帧的安装外参保持不变。');
      await props.onSaved(value, {lidarChanged: value.lidar_changed ?? false, allFrames});
    } catch (cause) {setError(`保存外参失败：${message(cause)}`); props.onError?.(message(cause));}
    finally {setSaving(false);}
  };
  const currentMatrix = frameCamera ? frameReady ? frameDraft : null
    : draft ? lidarTarget ? draft.lidar_to_ego[props.frameId] : draft.cameras[target] : undefined;
  const exportDocument = () => {
    if (frameCamera) {
      if (frameReady && frameDraft) download(`${props.clip.id}-${props.frameId}-${target}-calibration${dirty ? '-draft' : ''}.json`,
        JSON.stringify({...frameBaseline, effective_matrix: frameDraft, scope: 'frame',
          ...(dirty ? {candidate_matrix: frameDraft, candidate_method: 'manual'} : {}), draft: dirty}, null, 2));
    } else if (draft) download(`${props.clip.id}-calibration${dirty ? '-draft' : ''}.json`, JSON.stringify({...draft, draft: dirty}, null, 2));
  };

  return <div className="calibration-editor" role="dialog" aria-modal="false" aria-label="标定外参编辑" data-testid="calibration-editor">
    <div className="calibration-heading"><div><h2>标定外参</h2><p>当前帧 {props.frameId} · 默认只调整本帧与所选相机，平移单位米、旋转单位度。</p></div>
      <button className="icon-button" aria-label="关闭外参编辑" disabled={saving} onClick={close}>×</button></div>
    {loading ? <p role="status">正在读取标定…</p> : <>
      <label className="field-label">传感器变换<select aria-label="外参目标" value={target} disabled={switchLocked} onChange={event => chooseTarget(event.target.value)}>
        {props.clip.cameras.map(camera => <option key={camera.id} value={camera.id}>{camera.name} · {allFrames ? '共享 ego → camera' : '本帧 LiDAR → camera'}</option>)}
        <option value="__lidar__">LiDAR → ego</option></select></label>
      <label className="closed-field"><input type="checkbox" aria-label={lidarTarget ? 'LiDAR外参应用全帧' : '相机外参应用全帧'} checked={allFrames} disabled={switchLocked} onChange={event => chooseScope(event.target.checked)}/>
        应用整个 clip（默认仅当前帧 {props.frameId}）</label>
      <p className="micro-copy" role="status">{allFrames ? '当前范围：整个 clip。修改共享安装外参将影响所有帧，请确认数据需要相同的修正。'
        : lidarTarget ? `当前范围：仅帧 ${props.frameId} 的 LiDAR 外参。` : `当前范围：仅帧 ${props.frameId} · 相机 ${target}，其他帧和相机不变。`}</p>
      {frameLoading && <p role="status">正在读取本帧相机外参…</p>}
      <div className="calibration-controls">{(['translation', 'rotation'] as const).map(kind => <fieldset key={kind} disabled={locked}><legend>{kind === 'translation' ? '平移修正 Δ · 米' : '旋转修正 Δ · 度'}</legend>
        {correction[kind].map((value, axis) => {const name = kind === 'translation' ? ['X', 'Y', 'Z'][axis] : ['Roll', 'Pitch', 'Yaw'][axis];
          return <div className="calibration-axis" key={name}><span>{name}</span><input type="range" aria-label={`外参${name}滑块`} min={kind === 'translation' ? -Math.max(2, Math.abs(value)) : -180} max={kind === 'translation' ? Math.max(2, Math.abs(value)) : 180}
            step={kind === 'translation' ? .005 : .05} value={value} onChange={event => update(kind, axis, event.target.valueAsNumber)}/>
            <input type="number" aria-label={`外参${name}`} step={kind === 'translation' ? .005 : .05} value={value} onChange={event => update(kind, axis, event.target.valueAsNumber)}/></div>;})}
      </fieldset>)}</div>
      {currentMatrix && <div className="calibration-matrix"><strong>当前 4×4 外参矩阵</strong><pre data-testid="calibration-matrix">{currentMatrix.map(row => row.map(value => value.toFixed(6).padStart(11)).join(' ')).join('\n')}</pre></div>}
      {lidarTarget && <p className="micro-copy">{props.clip.source === '4d-clip-import'
        ? '预建地图保留世界坐标，外参调整更新各帧投影；已有标注保留并需复核。'
        : 'LiDAR 外参改动会改变场景点位置，保存后需重建地图；已有世界坐标标注保留并需复核。'}</p>}
      <div className="calibration-actions"><button className="button primary" disabled={locked || !dirty} aria-label="保存外参" onClick={() => void save()}>{saving ? '保存中…' : '保存外参'}</button>
        <button className="button subtle" disabled={locked || !draft} aria-label="导出外参" onClick={exportDocument}>导出外参</button>
        <button className="button ghost" disabled={locked || !dirty} onClick={resetTarget}>还原当前</button>
        <button className="button ghost" disabled={locked || !dirty} onClick={resetAll}>还原全部</button></div>
    </>}
    {error && <div className="calibration-error" role="alert"><p>{error}</p>{frameCamera && !frameReady && <button className="button subtle" disabled={switchLocked} onClick={() => setReload(value => value + 1)}>重新读取当前帧外参</button>}</div>}{status && <p role="status">{status}</p>}
  </div>;
}
