import {useEffect, useRef, useState} from 'react';
import {download, jsonRequest, request} from '../api';
import {applyCalibration, calibrationWorldPoses, correctExtrinsic, ZERO_CORRECTION,
  type CalibrationDocument, type ExtrinsicCorrection, type Matrix4} from '../geometry/calibration';
import type {Clip} from '../types';
export type {CalibrationDocument} from '../geometry/calibration';

export interface CalibrationEditorProps {
  clip: Clip;
  frameId: string;
  disabled?: boolean;
  onPreview: (clip: Clip | null, document?: CalibrationDocument) => void;
  onSaved: (document: CalibrationDocument, options: {lidarChanged: boolean}) => void;
  onClose: () => void;
  onError?: (message: string) => void;
  onDirtyChange?: (dirty: boolean) => void;
}

const message = (error: unknown) => error instanceof Error ? error.message : String(error);
const copy = <T,>(value: T): T => structuredClone(value);
const sameCalibration = (a: CalibrationDocument, b: CalibrationDocument) =>
  JSON.stringify([a.cameras, a.lidar_to_ego]) === JSON.stringify([b.cameras, b.lidar_to_ego]);

export function CalibrationEditor(props: CalibrationEditorProps) {
  const [baseline, setBaseline] = useState<CalibrationDocument | null>(null);
  const [draft, setDraft] = useState<CalibrationDocument | null>(null);
  const [target, setTarget] = useState(props.clip.cameras[0]?.id ?? '');
  const [allFrames, setAllFrames] = useState(false);
  const [correction, setCorrection] = useState<ExtrinsicCorrection>(copy(ZERO_CORRECTION));
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [status, setStatus] = useState('');
  const targetBase = useRef<Record<string, Matrix4>>({});
  const originalClip = useRef(props.clip);
  const latest = useRef(props);
  latest.current = props;
  const lidarTarget = target === '__lidar__';
  const dirty = Boolean(baseline && draft && !sameCalibration(baseline, draft));
  const locked = props.disabled || loading || saving;

  useEffect(() => {props.onDirtyChange?.(dirty);}, [dirty, props.onDirtyChange]);
  useEffect(() => () => latest.current.onDirtyChange?.(false), []);

  useEffect(() => {
    let active = true;
    setLoading(true);
    request<CalibrationDocument>(`/clips/${encodeURIComponent(props.clip.id)}/calibration`).then(value => {
      if (!active) return;
      originalClip.current = latest.current.clip;
      setBaseline(value); setDraft(copy(value));
      setTarget(latest.current.clip.cameras[0]?.id ?? '');
      targetBase.current = copy(value.cameras);
      setCorrection(copy(ZERO_CORRECTION));
    }).catch(cause => {if (active) setError(`读取标定失败：${message(cause)}`);})
      .finally(() => {if (active) setLoading(false);});
    return () => {active = false;};
  }, [props.clip.id]);

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
    if (!draft) return;
    setTarget(next); setCorrection(copy(ZERO_CORRECTION));
    targetBase.current = copy(next === '__lidar__' ? draft.lidar_to_ego : draft.cameras);
  };
  const chooseScope = (value: boolean) => {
    if (!draft) return;
    setAllFrames(value); setCorrection(copy(ZERO_CORRECTION));
    targetBase.current = copy(draft.lidar_to_ego);
  };
  const update = (kind: 'translation' | 'rotation', axis: number, value: number) => {
    if (!draft || !Number.isFinite(value) || locked) return;
    const values = [...correction[kind]] as [number, number, number];
    values[axis] = value;
    const nextCorrection = {...correction, [kind]: values};
    setCorrection(nextCorrection);
    if (lidarTarget) {
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
    props.onPreview(null);
  };
  const close = () => {
    if (saving || (dirty && !window.confirm('放弃尚未保存的外参修改？'))) return;
    props.onPreview(null); props.onClose();
  };
  const save = async () => {
    if (!baseline || !draft || !dirty || locked) return;
    setSaving(true); setError('');
    try {
      const value = await jsonRequest<CalibrationDocument>(`/clips/${encodeURIComponent(props.clip.id)}/calibration`,
        {revision: baseline.revision, cameras: draft.cameras, lidar_to_ego: draft.lidar_to_ego}, 'PUT');
      originalClip.current = applyCalibration(originalClip.current, value, value);
      setBaseline(value); setDraft(copy(value)); setCorrection(copy(ZERO_CORRECTION));
      targetBase.current = copy(lidarTarget ? value.lidar_to_ego : value.cameras);
      setStatus(value.map_rebuild_required ? '外参已保存；请重建场景地图并复核世界坐标标注。' : '外参已保存，所有帧投影已更新。');
      props.onSaved(value, {lidarChanged: value.lidar_changed ?? false});
    } catch (cause) {setError(`保存外参失败：${message(cause)}`); props.onError?.(message(cause));}
    finally {setSaving(false);}
  };
  const currentMatrix = draft ? lidarTarget ? draft.lidar_to_ego[props.frameId] : draft.cameras[target] : undefined;

  return <div className="calibration-editor" role="dialog" aria-modal="false" aria-label="标定外参编辑" data-testid="calibration-editor">
    <div className="calibration-heading"><div><h2>标定外参</h2><p>平移单位为米，旋转修正单位为度；调节后立即预览相机投影。</p></div>
      <button className="icon-button" aria-label="关闭外参编辑" disabled={saving} onClick={close}>×</button></div>
    {loading ? <p role="status">正在读取标定…</p> : <>
      <label className="field-label">传感器变换<select aria-label="外参目标" value={target} disabled={locked} onChange={event => chooseTarget(event.target.value)}>
        {props.clip.cameras.map(camera => <option key={camera.id} value={camera.id}>{camera.name} · ego → camera</option>)}
        <option value="__lidar__">LiDAR → ego</option></select></label>
      {lidarTarget && <label className="closed-field"><input type="checkbox" aria-label="LiDAR外参应用全帧" checked={allFrames} disabled={locked} onChange={event => chooseScope(event.target.checked)}/>
        应用整个 clip（否则仅 {props.frameId}）</label>}
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
        <button className="button subtle" disabled={locked || !draft} aria-label="导出外参" onClick={() => draft && download(`${props.clip.id}-calibration${dirty ? '-draft' : ''}.json`, JSON.stringify({...draft, draft: dirty}, null, 2))}>导出外参</button>
        <button className="button ghost" disabled={locked || !dirty} onClick={resetTarget}>还原当前</button>
        <button className="button ghost" disabled={locked || !dirty} onClick={resetAll}>还原全部</button></div>
    </>}
    {error && <p className="calibration-error" role="alert">{error}</p>}{status && <p role="status">{status}</p>}
  </div>;
}
