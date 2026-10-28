import {useEffect, useRef, useState} from 'react';
import {download, jsonRequest, request} from '../api';
import type {Matrix4} from '../geometry/calibration';
import type {Camera, Vec3} from '../types';
import './ProjectionCalibrationEditor.css';

export type ProjectionPickMode = '3d' | '2d' | null;
export interface ProjectionCorrespondence {
  id: string;
  point: Vec3;
  pixel: [number, number];
  enabled?: boolean;
}
export interface ProjectionPointError {
  id: string;
  error_px: number | null;
  baseline_error_px?: number | null;
  inlier: boolean;
  enabled?: boolean;
  projected?: [number, number] | null;
}
export interface ProjectionCalibrationMetrics {
  rmse_px?: number | null;
  baseline_rmse_px?: number | null;
  median_px?: number | null;
  p90_px?: number | null;
  inlier_count?: number;
  total_count?: number;
  inlier_rmse_px?: number | null;
  point_errors?: ProjectionPointError[];
  score_name?: string;
  holdout_before?: {nid?: number; overlap?: number};
  holdout_after?: {nid?: number; overlap?: number};
  holdout_improvement?: number;
  [key: string]: unknown;
}
export interface ProjectionCalibrationDocument {
  clip_id: string;
  frame_id: string;
  camera_id: string;
  revision: string;
  base_matrix: Matrix4;
  effective_matrix: Matrix4;
  saved_matrix: Matrix4 | null;
  correspondences: ProjectionCorrespondence[];
  method?: string | null;
  metrics?: ProjectionCalibrationMetrics | null;
  automatic_available?: boolean;
  automatic_detail?: string;
  projection_matrix?: Matrix4 | null;
  stale?: boolean;
  status_reason?: string;
  [key: string]: unknown;
}
export interface ProjectionCalibrationSolveResult extends ProjectionCalibrationDocument {
  candidate_matrix: Matrix4;
  usable: boolean;
  rejection_reasons?: string[];
}
export interface ProjectionCalibrationEditorProps {
  clipId: string;
  frameId: string;
  cameras: Camera[];
  cameraId: string;
  disabled?: boolean;
  pickedPoint?: Vec3 | null;
  pickedPixel?: [number, number] | null;
  onCameraChange: (cameraId: string) => void;
  onPickModeChange: (mode: ProjectionPickMode) => void;
  onConsumedPick: () => void;
  onPreview: (matrix: Matrix4 | null) => void;
  onSaved: (document: ProjectionCalibrationDocument) => void;
  onClose: () => void;
  onDirtyChange?: (dirty: boolean) => void;
  onBusyChange?: (busy: boolean) => void;
  onError?: (message: string) => void;
  onAnchorsChange?: (anchors: ProjectionCorrespondence[], pendingPoint: Vec3 | null, selectedId: string | null) => void;
  onAutoRegister?: () => Promise<ProjectionCalibrationSolveResult>;
}

const errorMessage = (error: unknown) => error instanceof Error ? error.message : String(error);
const clone = <T,>(value: T): T => structuredClone(value);
const pixels = (value?: number | null) => typeof value === 'number' && Number.isFinite(value) ? `${value.toFixed(1)} px` : '—';
const score = (value?: number) => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(4) : '—';
const percentage = (value?: number) => typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
const validMatrix = (matrix: Matrix4 | null | undefined): matrix is Matrix4 => Boolean(matrix && matrix.length === 4 && matrix.every(row => row.length === 4 && row.every(Number.isFinite)));
const sameAnchors = (a: ProjectionCorrespondence[], b: ProjectionCorrespondence[]) => JSON.stringify(a) === JSON.stringify(b);

/** Per-frame camera registration. Three-dimensional picks are in the frame's LiDAR coordinates. */
export function ProjectionCalibrationEditor(props: ProjectionCalibrationEditorProps) {
  const [baseline, setBaseline] = useState<ProjectionCalibrationDocument | null>(null);
  const [anchors, setAnchors] = useState<ProjectionCorrespondence[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [pendingPoint, setPendingPoint] = useState<Vec3 | null>(null);
  const [pickMode, setPickMode] = useState<ProjectionPickMode>(null);
  const [pickingId, setPickingId] = useState<string | null>(null);
  const [candidate, setCandidate] = useState<Matrix4 | null>(null);
  const [candidateMethod, setCandidateMethod] = useState<string>('correspondences');
  const [metrics, setMetrics] = useState<ProjectionCalibrationMetrics | null>(null);
  const [metricsKind, setMetricsKind] = useState<'saved' | 'candidate' | 'rejected'>('saved');
  const [showCandidate, setShowCandidate] = useState(false);
  const [loading, setLoading] = useState(true);
  const [operation, setOperation] = useState<'solve' | 'auto' | 'save' | 'reset' | null>(null);
  const [error, setError] = useState('');
  const [status, setStatus] = useState('');
  const [reload, setReload] = useState(0);
  const sequence = useRef(0);
  const anchorNumber = useRef(1);
  const latest = useRef(props);
  latest.current = props;
  const scope = `${props.clipId}/${props.frameId}/${props.cameraId}`;
  const activeScope = useRef(scope);
  activeScope.current = scope;
  const endpoint = `/clips/${encodeURIComponent(props.clipId)}/frames/${encodeURIComponent(props.frameId)}/projection-calibration/${encodeURIComponent(props.cameraId)}`;
  const camera = props.cameras.find(value => value.id === props.cameraId);
  const enabledCount = anchors.filter(anchor => anchor.enabled !== false).length;
  const dirty = Boolean(baseline && (candidate || pendingPoint || !sameAnchors(anchors, baseline.correspondences)));
  const locked = Boolean(props.disabled || loading || operation);
  const unchangedScope = (token: number, startScope: string) => sequence.current === token && activeScope.current === startScope;

  useEffect(() => {
    const token = ++sequence.current;
    let active = true;
    setLoading(true); setBaseline(null); setAnchors([]); setCandidate(null); setMetrics(null);
    setPendingPoint(null); setPickMode(null); setPickingId(null); setSelectedId(null);
    setShowCandidate(false); setError(''); setStatus(''); setOperation(null);
    latest.current.onPreview(null); latest.current.onPickModeChange(null); latest.current.onConsumedPick();
    request<ProjectionCalibrationDocument>(endpoint).then(value => {
      if (!active || !unchangedScope(token, scope)) return;
      setBaseline(value); setAnchors(clone(value.correspondences ?? [])); setMetrics(value.metrics ?? null); setMetricsKind('saved');
      const ids = (value.correspondences ?? []).map(anchor => Number(anchor.id.match(/^anchor-(\d+)$/)?.[1] ?? 0));
      anchorNumber.current = Math.max(0, ...ids) + 1;
    }).catch(cause => {
      if (active && unchangedScope(token, scope)) setError(`读取当前帧修正失败：${errorMessage(cause)}`);
    }).finally(() => {if (active && unchangedScope(token, scope)) setLoading(false);});
    return () => {active = false;};
  }, [scope, endpoint, reload]);

  useEffect(() => {latest.current.onDirtyChange?.(dirty);}, [dirty]);
  useEffect(() => {latest.current.onBusyChange?.(loading || Boolean(operation));}, [loading, operation]);
  useEffect(() => {latest.current.onAnchorsChange?.(anchors, pendingPoint, selectedId);}, [anchors, pendingPoint, selectedId]);
  useEffect(() => () => {
    ++sequence.current;
    latest.current.onDirtyChange?.(false); latest.current.onPickModeChange(null); latest.current.onPreview(null);
    latest.current.onAnchorsChange?.([], null, null);
    latest.current.onBusyChange?.(false);
  }, []);
  useEffect(() => {
    if (!dirty) return;
    const guard = (event: BeforeUnloadEvent) => {event.preventDefault(); event.returnValue = '';};
    window.addEventListener('beforeunload', guard);
    return () => window.removeEventListener('beforeunload', guard);
  }, [dirty]);

  const changePickMode = (mode: ProjectionPickMode) => {setPickMode(mode); props.onPickModeChange(mode);};
  const invalidate = () => {
    setCandidate(null); setShowCandidate(false); setMetrics(null); setError(''); setStatus('对应点已修改，请重新求解。');
    props.onPreview(null);
  };
  const updateAnchors = (value: ProjectionCorrespondence[]) => {setAnchors(value); invalidate();};

  useEffect(() => {
    if (!props.pickedPoint) return;
    if (locked || pickMode !== '3d' || !baseline) {props.onConsumedPick(); return;}
    const point = [...props.pickedPoint] as Vec3;
    props.onConsumedPick();
    if (!point.every(Number.isFinite)) {setError('选取的三维点无效，请重新选择。'); return;}
    if (pickingId) {
      updateAnchors(anchors.map(anchor => anchor.id === pickingId ? {...anchor, point} : anchor));
      setPickingId(null); changePickMode(null);
    } else {
      setPendingPoint(point); setStatus('三维点已选中，请在当前相机图像上点击同一位置。'); changePickMode('2d');
    }
  // A pick is a one-shot event consumed by the parent. Other state changes must not replay it.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.pickedPoint]);

  useEffect(() => {
    if (!props.pickedPixel) return;
    if (locked || pickMode !== '2d' || !baseline) {props.onConsumedPick(); return;}
    const pixel = [...props.pickedPixel] as [number, number];
    props.onConsumedPick();
    if (!pixel.every(Number.isFinite) || (camera && (pixel[0] < 0 || pixel[1] < 0 || pixel[0] >= camera.width || pixel[1] >= camera.height))) {
      setError('请在当前相机图像内部选择像素位置。'); return;
    }
    if (pickingId) {
      updateAnchors(anchors.map(anchor => anchor.id === pickingId ? {...anchor, pixel} : anchor));
    } else if (pendingPoint) {
      const id = `anchor-${anchorNumber.current++}`;
      updateAnchors([...anchors, {id, point: pendingPoint, pixel, enabled: true}]); setSelectedId(id);
    } else {setError('请先在点云中选择三维点。'); changePickMode(null); return;}
    setPendingPoint(null); setPickingId(null); changePickMode(null);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.pickedPixel]);

  const startPick = (mode: Exclude<ProjectionPickMode, null>, id: string | null = null) => {
    if (locked || !baseline) return;
    setSelectedId(id); setPickingId(id); setPendingPoint(null); setError('');
    setStatus(mode === '3d' ? '请在点云中点击清晰的静态特征点。' : '请在当前相机图像中点击对应位置。');
    changePickMode(mode);
  };
  const cancelPick = () => {setPendingPoint(null); setPickingId(null); changePickMode(null); setStatus('已取消取点。');};
  const reportFailure = (cause: unknown, prefix: string) => {
    const detail = errorMessage(cause);
    const conflict = /409|revision|版本|已被.*修改/i.test(detail);
    setError(`${prefix}：${detail}${conflict ? '。请先导出当前对应点，再重新读取最新版本。' : ''}`);
    props.onError?.(detail);
  };
  const solve = async (automatic: boolean) => {
    if (!baseline || locked || (!automatic && enabledCount < 6)) return;
    const token = sequence.current, startScope = scope;
    setOperation(automatic ? 'auto' : 'solve'); setError(''); setStatus(''); changePickMode(null);
    try {
      const value = automatic && props.onAutoRegister ? await props.onAutoRegister() :
        await jsonRequest<ProjectionCalibrationSolveResult>(`${endpoint}/${automatic ? 'auto' : 'solve'}`,
          {revision: baseline.revision, initial_matrix: candidate ?? baseline.effective_matrix, ...(!automatic ? {correspondences: anchors} : {})});
      if (!unchangedScope(token, startScope)) return;
      if (!value.usable || !validMatrix(value.candidate_matrix)) {
        setError(`本次${automatic ? '自动配准' : '对应点求解'}未通过质量检查：${(value.rejection_reasons ?? []).join('；') || '未得到可靠的投影修正，请补充不同位置与距离的静态特征。'}`);
        if (!candidate) {setMetrics(value.metrics ?? null); setMetricsKind('rejected');}
        return;
      }
      setCandidate(clone(value.candidate_matrix)); setCandidateMethod(value.method ?? (automatic ? 'direct_nid' : 'correspondences'));
      setMetrics(value.metrics ?? null); setMetricsKind('candidate'); setShowCandidate(true); props.onPreview(value.candidate_matrix);
      setStatus('候选修正已生成。请切换前后预览，检查不同距离的车道线与静态物体，再保存。');
    } catch (cause) {if (unchangedScope(token, startScope)) reportFailure(cause, automatic ? '自动配准失败' : '对应点求解失败');}
    finally {if (unchangedScope(token, startScope)) setOperation(null);}
  };
  const preview = (next: boolean) => {
    setShowCandidate(next); props.onPreview(next ? candidate : null);
  };
  const discardCandidate = () => {
    setCandidate(null); setShowCandidate(false); setMetrics(sameAnchors(anchors, baseline?.correspondences ?? []) ? baseline?.metrics ?? null : null);
    setMetricsKind('saved');
    props.onPreview(null); setError(''); setStatus('已丢弃候选修正，对应点保留。');
  };
  const save = async (reset = false) => {
    if (!baseline || locked || (!reset && !candidate)) return;
    const token = sequence.current, startScope = scope;
    setOperation(reset ? 'reset' : 'save'); setError(''); changePickMode(null);
    try {
      const value = await jsonRequest<ProjectionCalibrationDocument>(endpoint, {
        revision: baseline.revision, matrix: reset ? null : candidate,
        correspondences: anchors, method: reset ? 'reset' : candidateMethod, ...(!reset ? {metrics} : {}),
      }, 'PUT');
      if (!unchangedScope(token, startScope)) return;
      setBaseline(value); setAnchors(clone(value.correspondences ?? [])); setMetrics(value.metrics ?? null); setMetricsKind('saved');
      setCandidate(null); setShowCandidate(false); setPendingPoint(null); setPickingId(null);
      props.onSaved(value); props.onPreview(null);
      setStatus(reset ? '已清除本帧本相机的已保存修正，恢复原始投影。' : '已保存，仅应用于当前帧和当前相机。');
    } catch (cause) {if (unchangedScope(token, startScope)) reportFailure(cause, reset ? '清除修正失败' : '保存修正失败');}
    finally {if (unchangedScope(token, startScope)) setOperation(null);}
  };
  const close = () => {
    if (operation || (dirty && !window.confirm('当前对应点或候选修正尚未保存，确定放弃并关闭？'))) return;
    props.onPreview(null); props.onPickModeChange(null); props.onClose();
  };
  const chooseCamera = (id: string) => {
    if (id === props.cameraId || locked || (dirty && !window.confirm('切换相机会丢弃当前未保存的对应点和候选修正，确定切换？'))) return;
    props.onPickModeChange(null); props.onPreview(null); props.onCameraChange(id);
  };
  const exportDocument = () => {
    if (!baseline) return;
    download(`${props.clipId}-${props.frameId}-${props.cameraId}-projection${dirty ? '-draft' : ''}.json`, JSON.stringify({
      ...baseline, correspondences: anchors, ...(candidate ? {candidate_matrix: candidate, candidate_method: candidateMethod, candidate_metrics: metrics} : {}),
      ...(pendingPoint ? {pending_point: pendingPoint} : {}), draft: dirty,
    }, null, 2));
  };
  const errors = new Map((metrics?.point_errors ?? []).map(value => [value.id, value]));

  return <div className="projection-calibration-editor" role="dialog" aria-modal="false" aria-label="逐帧投影校正" data-testid="projection-calibration-editor">
    <div className="projection-calibration-heading"><div><h2>逐帧投影校正</h2><p>当前帧与当前相机独立求解，三维标注坐标保持不变。</p></div>
      <button className="icon-button" aria-label="关闭投影校正" disabled={Boolean(operation)} onClick={close}>×</button></div>
    <div className="projection-calibration-scope"><span>帧 <strong title={props.frameId}>{props.frameId}</strong></span>{baseline?.saved_matrix && <span className="projection-saved-tag">{baseline.stale ? '修正待重算' : '已有修正'}</span>}</div>
    {baseline?.status_reason && <p className="projection-calibration-notice" role="status">{baseline.status_reason}</p>}
    <label className="field-label">当前相机<select aria-label="投影校正相机" value={props.cameraId} disabled={locked} onChange={event => chooseCamera(event.target.value)}>
      {props.cameras.map(value => <option key={value.id} value={value.id}>{value.name}</option>)}
    </select></label>
    {loading ? <p className="projection-progress" role="status">正在读取当前帧投影…</p> : baseline && <>
      <section className="projection-calibration-section"><h3>图像与点云自动配准</h3>
        <p>利用当前图像与静态点云直接估计候选修正，无需手工对应点。</p>
        <button className="button subtle full-width" aria-label="自动配准当前帧" disabled={locked || baseline.automatic_available === false} onClick={() => void solve(true)}>{operation === 'auto' ? '正在配准，请稍候…' : '自动配准当前帧'}</button>
        {baseline.automatic_detail && <p className="projection-help">{baseline.automatic_detail}</p>}
      </section>
      <section className="projection-calibration-section"><div className="projection-section-title"><h3>对应点辅助求解</h3><span>{enabledCount} 个启用</span></div>
        <p>在点云与图像中选择同一静态位置，至少 6 对。选点应覆盖画面左右、近处和远处，优先选择虚线端点、路缘转角、杆体等清晰特征。</p>
        <div className="projection-pick-actions"><button className={`button subtle${pickMode === '3d' && !pickingId ? ' active' : ''}`} disabled={locked} onClick={() => startPick('3d')}>＋ 添加对应点</button>
          {pickMode && <button className="button ghost" disabled={locked} onClick={cancelPick}>取消取点</button>}</div>
        {pickMode && <div className="projection-pick-hint" role="status"><span className="status-dot"/>{pickMode === '3d' ? '在点云中点击三维位置' : `在 ${camera?.name ?? props.cameraId} 图像中点击同一位置`}{pickingId ? ' · 更新所选点' : pendingPoint ? ' · 三维点已选中' : ''}</div>}
        <ol className="projection-anchor-list" aria-label="投影校正对应点">
          {anchors.map((anchor, index) => {const residual = errors.get(anchor.id); return <li key={anchor.id} className={`projection-anchor${selectedId === anchor.id ? ' selected' : ''}${anchor.enabled === false ? ' disabled' : ''}`}>
            <div className="projection-anchor-heading"><label><input type="checkbox" aria-label={`启用对应点 ${index + 1}`} checked={anchor.enabled !== false} disabled={locked} onChange={event => updateAnchors(anchors.map(value => value.id === anchor.id ? {...value, enabled: event.target.checked} : value))}/>
              <button type="button" className="projection-anchor-select" aria-pressed={selectedId === anchor.id} onClick={() => setSelectedId(anchor.id)}>对应点 {index + 1}</button></label>
              {residual && <span className={`projection-residual${residual.inlier ? '' : ' outlier'}`} title={residual.inlier ? '内点' : '离群点或未参与求解'}>{pixels(residual.error_px)}{anchor.enabled !== false && !residual.inlier ? ' · 离群' : ''}</span>}
              <button className="icon-button" aria-label={`删除对应点 ${index + 1}`} disabled={locked} onClick={() => {updateAnchors(anchors.filter(value => value.id !== anchor.id)); if (selectedId === anchor.id) setSelectedId(null); if (pickingId === anchor.id) cancelPick();}}>×</button></div>
            <div className="projection-anchor-values"><span title="当前帧 LiDAR 坐标，单位米">3D <code>{anchor.point.map(value => value.toFixed(2)).join(', ')}</code></span><span title="原始图像像素坐标">2D <code>{anchor.pixel.map(value => value.toFixed(1)).join(', ')}</code></span></div>
            <div className="projection-anchor-actions"><button disabled={locked} onClick={() => startPick('3d', anchor.id)}>重选三维点</button><button disabled={locked} onClick={() => startPick('2d', anchor.id)}>重选图像点</button></div>
          </li>;})}
        </ol>
        {!anchors.length && <p className="projection-empty">还没有对应点。添加后，点的编号会显示在点云与图像上。</p>}
        <button className="button subtle full-width" aria-label="根据对应点求解投影" disabled={locked || enabledCount < 6 || Boolean(pickMode)} onClick={() => void solve(false)}>{operation === 'solve' ? '正在求解…' : `根据对应点自动求解${enabledCount < 6 ? `（还需 ${6 - enabledCount} 对）` : ''}`}</button>
      </section>
      {metrics && <section className="projection-calibration-section"><h3>{metricsKind === 'candidate' ? '候选修正质量' : metricsKind === 'rejected' ? '未通过质量检查' : '已保存修正质量'}</h3>
        {metrics.score_name === 'normalized_information_distance' && <><dl className="projection-metrics"><div><dt>修正前 NID</dt><dd>{score(metrics.holdout_before?.nid)}</dd></div><div><dt>修正后 NID</dt><dd>{score(metrics.holdout_after?.nid)}</dd></div><div><dt>验证覆盖率</dt><dd>{percentage(metrics.holdout_after?.overlap)}</dd></div><div><dt>NID 改善</dt><dd>{score(metrics.holdout_improvement)}</dd></div></dl><p className="projection-help">NID 衡量图像与点云的纹理一致性，越低越好；此指标不代表像素误差。</p></>}
        {(metrics.score_name !== 'normalized_information_distance' || typeof metrics.rmse_px === 'number') && <dl className="projection-metrics"><div><dt>修正前 RMSE</dt><dd>{pixels(metrics.baseline_rmse_px)}</dd></div><div><dt>修正后 RMSE</dt><dd>{pixels(metrics.rmse_px)}</dd></div>
          <div><dt>内点 / 参与点</dt><dd>{typeof metrics.inlier_count === 'number' ? `${metrics.inlier_count} / ${metrics.total_count ?? enabledCount}` : '—'}</dd></div><div><dt>内点 RMSE</dt><dd>{pixels(metrics.inlier_rmse_px)}</dd></div></dl>
        }
        {candidateMethod === 'direct_nid' && candidate && <p className="projection-help">自动配准质量取决于点云与图像的共同结构。请检查近远处静态物体是否同时对齐。</p>}
      </section>}
      {candidate && <section className="projection-calibration-section"><h3>预览与保存</h3><div className="projection-preview-toggle" role="group" aria-label="修正前后预览"><button className={!showCandidate ? 'active' : ''} aria-pressed={!showCandidate} disabled={locked} onClick={() => preview(false)}>修正前</button><button className={showCandidate ? 'active' : ''} aria-pressed={showCandidate} disabled={locked} onClick={() => preview(true)}>候选修正</button></div>
        <p className="projection-help">修正前为当前已保存的投影。保存后同时应用于点云与三维标注的图像投影。</p>
        <div className="projection-calibration-actions"><button className="button primary full-width" aria-label="保存本帧本相机投影修正" disabled={locked} onClick={() => void save()}>{operation === 'save' ? '正在保存…' : '保存本帧本相机'}</button><button className="button ghost" disabled={locked} onClick={discardCandidate}>丢弃候选</button></div>
      </section>}
      <div className="projection-calibration-actions"><button className="button subtle" disabled={locked} onClick={exportDocument}>导出校正 JSON</button><button className="button ghost" disabled={locked || !baseline.saved_matrix || Boolean(candidate) || Boolean(pendingPoint)} onClick={() => void save(true)}>{operation === 'reset' ? '正在清除…' : '清除已保存修正'}</button></div>
    </>}
    {operation === 'auto' && <p className="projection-progress" role="status">正在比较图像与点云结构，通常需要数十秒。完成后仍需检查候选投影。</p>}
    {error && <div className="projection-calibration-error" role="alert"><p>{error}</p>{!locked && <button className="button ghost" onClick={() => {if (!dirty || window.confirm('重新读取会丢弃未保存的对应点和候选修正，确定继续？')) setReload(value => value + 1);}}>重新读取</button>}</div>}
    {status && <p className="projection-calibration-status" role="status">{status}</p>}
  </div>;
}
