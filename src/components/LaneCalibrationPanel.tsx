import {useEffect, useRef, useState} from 'react';
import {jsonRequest, request} from '../api';
import {annotationId} from '../ids';
import type {Camera, Frame, Vector} from '../types';
import './LaneCalibrationPanel.css';

export interface LaneCalibrationOverlay {label: string; color: string; points: ([number, number] | null)[]; vertices?: boolean;}
interface Pair {
  id: string; frame_id: string; lane_id: string; image_points: number[][]; image_size: number[];
  world_range: number[]; image_world_range?: number[]; image_lane_id?: string | null;
  split: 'train' | 'validation'; weight: number;
}
interface Pairing {revision: number; pairs: Pair[];}
interface Metrics {mean_px: number; rmse_px: number; median_px: number; p95_px: number; frames: number; samples: number;}
interface Statistics {
  splits: {train: Metrics | null; validation: Metrics | null};
  pairs: {id: string; mean_px: number; rmse_px: number; p95_px: number}[];
  overlays: {id: string; frame_id: string; pixels: ([number, number] | null)[]}[];
}
interface Result {
  initial_matrix: number[][]; optimized_matrix: number[][];
  before: Statistics; after: Statistics; usable: boolean; rejection_reasons: string[];
  error_reduction: {train: number | null; validation: number | null};
}
interface Job {
  id: string; status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  iteration: number; loss?: number; error?: string; can_apply?: boolean; applied?: boolean; result: Result | null;
  apply_block_reason?: string;
  request: {pairing_revision: number};
}
interface State {
  runtime: {available: boolean; detail: string}; pairing: Pairing; lanes: Vector[]; latest_job: Job | null;
  coordinate_info: {fused_map: boolean; sensor_mount_available: boolean};
}
interface Props {
  clipId: string; camera: Camera; frame: Frame; selectedId: string | null; disabled: boolean;
  onSelectLane: (id: string) => void; onOverlays: (lines: LaneCalibrationOverlay[]) => void;
  onBusy: (busy: boolean) => void; onApplied: () => Promise<void>; onClose: () => void;
  pickedPixel: [number, number] | null;
  onPickModeChange: (enabled: boolean) => void;
  onConsumedPick: () => void;
}
const errorText = (error: unknown) => error instanceof Error ? error.message : String(error);
const running = (job: Job | null) => job?.status === 'queued' || job?.status === 'running';
const metric = (value: number | undefined) => value === undefined ? '—' : `${value.toFixed(2)} px`;

export function LaneCalibrationPanel(props: Props) {
  const [state, setState] = useState<State | null>(null);
  const [pairing, setPairing] = useState<Pairing>({revision: 0, pairs: []});
  const [job, setJob] = useState<Job | null>(null);
  const [before, setBefore] = useState<Statistics | null>(null);
  const [showResult, setShowResult] = useState(true);
  const [chosenLane, setChosenLane] = useState('');
  const [chosenImageLane, setChosenImageLane] = useState('');
  const [sourceMode, setSourceMode] = useState<'cloud' | 'image'>('cloud');
  const [drawing, setDrawing] = useState(false);
  const [imagePoints, setImagePoints] = useState<[number, number][]>([]);
  const [range, setRange] = useState([0, 100]);
  const [imageRange, setImageRange] = useState([0, 100]);
  const [role, setRole] = useState<'train' | 'validation'>('train');
  const [mode, setMode] = useState('multi');
  const [poseMode, setPoseMode] = useState('auto');
  const [confirmed, setConfirmed] = useState(false);
  const [synchronized, setSynchronized] = useState(false);
  const [advanced, setAdvanced] = useState('');
  const [iterations, setIterations] = useState(180);
  const [loading, setLoading] = useState(true);
  const [operation, setOperation] = useState('');
  const [error, setError] = useState('');
  const latest = useRef(props); latest.current = props;
  const endpoint = `/clips/${encodeURIComponent(props.clipId)}/lane-calibration/${encodeURIComponent(props.camera.id)}`;
  const active = running(job);
  const locked = props.disabled || loading || Boolean(operation);
  const selected = state?.lanes.find(lane => lane.id === props.selectedId);
  const imageUnavailable = (id: string) => pairing.pairs.some(pair => pair.image_lane_id === id && (pair.frame_id === props.frame.id || pair.lane_id !== chosenLane));
  const result = showResult && job?.request.pairing_revision === pairing.revision ? job.result : null;

  useEffect(() => {
    let alive = true; const controller = new AbortController();
    setLoading(true); setError('');
    request<State>(endpoint, {signal: controller.signal}).then(value => {
      if (alive) {setState(value); setPairing(value.pairing); setJob(value.latest_job);}
    }).catch(cause => {if (alive) setError(errorText(cause));}).finally(() => {if (alive) setLoading(false);});
    return () => {alive = false; controller.abort();};
  }, [endpoint]);
  useEffect(() => {latest.current.onBusy(loading || Boolean(operation));}, [loading, operation]);
  useEffect(() => () => {latest.current.onBusy(false); latest.current.onOverlays([]); latest.current.onPickModeChange(false); latest.current.onConsumedPick();}, []);
  useEffect(() => {latest.current.onPickModeChange(sourceMode === 'image' && drawing && Boolean(chosenLane) && !locked && !active);}, [sourceMode, drawing, chosenLane, locked, active]);
  useEffect(() => {
    const point = props.pickedPixel;
    if (!point) return;
    if (sourceMode === 'image' && drawing && chosenLane && !locked && !active) {
      if (point.every(Number.isFinite) && point[0] >= 0 && point[0] < props.camera.width && point[1] >= 0 && point[1] < props.camera.height) {
        setImagePoints(points => points.length < 512 ? [...points, point] : points);
      } else setError('请在当前相机原图范围内选点。');
    }
    latest.current.onConsumedPick();
  }, [props.pickedPixel]);
  useEffect(() => {
    setChosenLane(''); setChosenImageLane(''); setRange([0, 100]); setImageRange([0, 100]);
    setImagePoints([]); setDrawing(false);
    setRole(pairing.pairs.find(pair => pair.frame_id === props.frame.id)?.split ?? 'train');
  }, [props.frame.id]);
  useEffect(() => {
    let alive = true; let timer: ReturnType<typeof setTimeout>;
    if (!job || !running(job)) return;
    const poll = async () => {
      try {
        const next = await request<Job>(`${endpoint}/jobs/${job.id}`);
        if (!alive) return; setJob(next);
        if (running(next)) timer = setTimeout(() => void poll(), 600);
      } catch (cause) {if (alive) {setError(errorText(cause)); timer = setTimeout(() => void poll(), 2000);}}
    };
    timer = setTimeout(() => void poll(), 300);
    return () => {alive = false; clearTimeout(timer);};
  }, [job?.id, job?.status, endpoint]);
  useEffect(() => {
    const lines: LaneCalibrationOverlay[] = [];
    if (sourceMode === 'image' && imagePoints.length) lines.push({label: `手绘图像线 · ${imagePoints.length} 点`, color: '#facc15', points: imagePoints, vertices: true});
    for (const pair of pairing.pairs.filter(p => p.frame_id === props.frame.id)) {
      const index = pairing.pairs.indexOf(pair) + 1;
      lines.push({label: `${index} 固定图像线${pair.image_lane_id ? ` ${pair.image_lane_id}` : ''}`, color: '#4ade80', points: pair.image_points as [number, number][]});
      const a = (result?.before ?? before)?.overlays.find(p => p.id === pair.id);
      const b = result?.after.overlays.find(p => p.id === pair.id);
      if (a) lines.push({label: `${index} 优化前`, color: '#fb923c', points: a.pixels});
      if (b) lines.push({label: `${index} 优化后`, color: '#38bdf8', points: b.pixels});
    }
    latest.current.onOverlays(lines);
  }, [pairing, before, result, props.frame.id, sourceMode, imagePoints]);

  const persist = async (pairs: Pair[]) => {
    setOperation('pairs'); setError('');
    try {
      const value = await jsonRequest<Pairing>(`${endpoint}/pairs`, {revision: pairing.revision, pairs}, 'PUT');
      setPairing(value); setBefore(null); setShowResult(false); return true;
    } catch (cause) {setError(errorText(cause)); return false;}
    finally {setOperation('');}
  };
  const addPair = async () => {
    if (sourceMode === 'image') {
      if (!chosenLane || imagePoints.length < 2 || locked || active) return;
      const pairs = pairing.pairs.map(pair => pair.frame_id === props.frame.id ? {...pair, split: role} : pair);
      const saved = await persist([...pairs, {id: annotationId('lane-pair'), frame_id: props.frame.id, lane_id: chosenLane,
        image_points: imagePoints, image_size: [props.camera.width, props.camera.height],
        world_range: range.map(value => value / 100), split: role, weight: 1}]);
      if (saved) {setImagePoints([]); setDrawing(false);}
      return;
    }
    if (!chosenLane || !chosenImageLane || chosenLane === chosenImageLane || locked || active) return;
    setOperation('pairs'); setError('');
    try {
      const value = await jsonRequest<Pairing>(`${endpoint}/pairs/from-lanes`, {
        revision: pairing.revision, id: annotationId('lane-pair'), frame_id: props.frame.id,
        lane_id: chosenLane, image_lane_id: chosenImageLane, world_range: range.map(v => v / 100),
        image_world_range: imageRange.map(v => v / 100), split: role, weight: 1,
      });
      setPairing(value); setBefore(null); setShowResult(false); setChosenImageLane('');
    } catch (cause) {setError(errorText(cause));} finally {setOperation('');}
  };
  const settings = () => {
    const extra = advanced.trim() ? JSON.parse(advanced) as Record<string, unknown> : {};
    return {...extra, pairing_revision: pairing.revision, mode, ...(mode === 'single' ? {frame_ids: [props.frame.id]} : {}), pose_mode: poseMode,
      conventions_confirmed: confirmed, time_alignment_confirmed: synchronized,
      config: {...(extra.config as Record<string, unknown> ?? {}), adam_iterations: iterations}};
  };
  const calculate = async (preview: boolean) => {
    setOperation(preview ? 'preview' : 'start'); setError('');
    try {
      if (preview) {
        const value = await jsonRequest<{statistics: Statistics}>(`${endpoint}/preview`, settings());
        setBefore(value.statistics); setShowResult(false);
      } else {setJob(await jsonRequest<Job>(`${endpoint}/jobs`, settings())); setShowResult(true);}
    } catch (cause) {setError(errorText(cause));}
    finally {setOperation('');}
  };
  const cancel = async () => {
    if (!job) return; setOperation('cancel');
    try {setJob(await request<Job>(`${endpoint}/jobs/${job.id}/cancel`, {method: 'POST'}));}
    catch (cause) {setError(errorText(cause));} finally {setOperation('');}
  };
  const apply = async () => {
    if (!job || !window.confirm(`将优化外参应用于整个 clip 的 ${props.camera.name}，替换该相机已有的逐帧投影修正？原始安装外参及导入修正会备份，其他相机保持不变。`)) return;
    setOperation('apply'); setError('');
    try {setJob(await jsonRequest<Job>(`${endpoint}/jobs/${job.id}/apply`, {confirm_shared_update: true})); await props.onApplied();}
    catch (cause) {setError(errorText(cause));} finally {setOperation('');}
  };
  const discard = async () => {
    if (locked || active || !window.confirm('丢弃当前相机的优化候选并清空所有帧的配对，重新选择？已保存外参和原始标注会保留。')) return;
    setOperation('discard'); setError('');
    try {
      const value = await jsonRequest<State>(`${endpoint}/discard`, {revision: pairing.revision, pairs: []});
      setState(value); setPairing(value.pairing); setJob(null); setBefore(null); setShowResult(false);
      setChosenLane(''); setChosenImageLane(''); setRange([0, 100]); setImageRange([0, 100]); setRole('train');
      setImagePoints([]); setDrawing(false);
      props.onSelectLane(''); props.onOverlays([]);
    } catch (cause) {setError(errorText(cause));} finally {setOperation('');}
  };

  return <div className="lane-calibration-panel" data-testid="lane-calibration-panel" role="region" aria-label="车道线外参优化">
    <div className="lane-calibration-heading"><h2>车道线外参优化</h2><button className="icon-button" aria-label="关闭车道线优化" disabled={Boolean(operation)} onClick={props.onClose}>×</button></div>
    <p>{props.camera.name} · 当前帧 <strong>{props.frame.id}</strong></p>
    <p>先在点云中选择三维真值线，再在图像上逐点选择对应线，或在点云中选择已有图像投影线。两种方式可以混合配对，共用预览与优化；真实的逐帧相对震动仍需逐帧校准。</p>
    {loading ? <p role="status">正在读取世界车道线与已保存配对…</p> : state && <>
      <section><h3>1. 选择三维真值线</h3><p>在点云中点击已保存的线，或使用下方 ID 列表选择并在点云中确认高亮，再点击按钮。</p>
        <button className="button subtle full-width" aria-label="使用选中三维车道线" disabled={locked || active || !selected} onClick={() => {setChosenLane(selected!.id); if (selected!.id === chosenImageLane) setChosenImageLane('');}}>使用选中三维线：{selected?.id ?? '尚未选中'}</button>
        <label className="field-label">已选真值线 / 按 ID 选择<select aria-label="优化三维车道线" value={chosenLane} disabled={locked || active} onChange={e => {setChosenLane(e.target.value); props.onSelectLane(e.target.value); if (e.target.value === chosenImageLane) setChosenImageLane('');}}>
          <option value="">请选择已保存的三维线</option>{state.lanes.map(lane => <option key={lane.id} value={lane.id}>{lane.id} · {lane.label} · {lane.points.length} 顶点</option>)}</select></label>
        <div className="lane-calibration-range">三维线取样范围（%）{range.map((value, axis) => <input key={axis} aria-label={axis ? '三维线范围终点' : '三维线范围起点'} type="number" min="0" max="100" value={value} disabled={locked || active} onChange={e => setRange(range.map((v, i) => i === axis ? Number(e.target.value) : v))}/>)}</div>
      </section>
      <section><h3>2. 选择对应图像线</h3>
        <label className="field-label">图像约束选择方式<select aria-label="图像约束选择方式" value={sourceMode} disabled={locked || active} onChange={e => {setSourceMode(e.target.value as typeof sourceMode); setDrawing(false);}}><option value="cloud">点云中选择投影车道线</option><option value="image">图像上手动选线</option></select></label>
        {sourceMode === 'cloud' ? <><p>在点云中点击另一条已保存的图像投影线，再点击下方按钮；也可按 ID 选择。请选择来自当前帧 {props.frame.id}、当前相机 {props.camera.name} 的线。</p>
        <button className="button subtle full-width" aria-label="使用选中图像投影车道线" disabled={locked || active || !chosenLane || !selected || selected.id === chosenLane || imageUnavailable(selected.id)} onClick={() => setChosenImageLane(selected!.id)}>使用选中图像投影线：{selected?.id ?? '尚未选中'}</button>
        <label className="field-label">已选图像投影线 / 按 ID 选择<select aria-label="优化图像投影车道线" value={chosenImageLane} disabled={locked || active || !chosenLane} onChange={e => {setChosenImageLane(e.target.value); props.onSelectLane(e.target.value);}}>
          <option value="">请选择点云中的图像投影线</option>{state.lanes.filter(lane => lane.id !== chosenLane).map(lane => <option key={lane.id} value={lane.id} disabled={imageUnavailable(lane.id)}>{lane.id} · {lane.label} · {lane.points.length} 顶点{imageUnavailable(lane.id) ? ' · 已配对' : ''}</option>)}</select></label>
        <div className="lane-calibration-range">图像投影线取样范围（%）{imageRange.map((value, axis) => <input key={axis} aria-label={axis ? '图像投影线范围终点' : '图像投影线范围起点'} type="number" min="0" max="100" value={value} disabled={locked || active} onChange={e => setImageRange(imageRange.map((v, i) => i === axis ? Number(e.target.value) : v))}/>)}</div>
        <p>保存配对时按当前已保存的投影参数固定图像位置，绿色叠加可供核对。若当初使用了其他帧或标定，请先恢复对应参考。</p>
        <p role="status">{chosenLane && chosenImageLane ? `待添加：${chosenLane} ↔ ${chosenImageLane}` : '请选择两条不同的线。'}</p>
        </> : <>
          <p>在当前帧 {props.frame.id}、相机 {props.camera.name} 的原图上沿对应车道线依次点击，至少两个不同点，最多 512 点。缩放后的显示会换算为原图像素，不修改原始图像标注。</p>
          <button className="button subtle full-width" aria-label="绘制优化图像车道线" disabled={locked || active || !chosenLane || !props.frame.images[props.camera.id]} onClick={() => setDrawing(value => !value)}>{drawing ? '暂停图像选点' : '开始图像选点'}</button>
          <p role="status">手绘图像线：{imagePoints.length} 点{drawing ? ' · 请点击左侧当前相机图像' : ''}</p>
          <button className="button ghost" aria-label="撤回优化图像线一点" disabled={locked || active || !imagePoints.length} onClick={() => setImagePoints(points => points.slice(0, -1))}>撤回一点</button>
          <button className="button ghost" aria-label="清空优化图像线" disabled={locked || active || !imagePoints.length} onClick={() => setImagePoints([])}>清空手绘线</button>
        </>}
        <label className="field-label">本帧用途<select aria-label="优化当前帧用途" value={role} disabled={locked || active} onChange={e => setRole(e.target.value as typeof role)}><option value="train">训练帧</option><option value="validation">独立验证帧</option></select></label>
        <button className="button primary full-width" aria-label="添加车道线配对" disabled={locked || active || !chosenLane || (sourceMode === 'image' ? imagePoints.length < 2 : !chosenImageLane || chosenLane === chosenImageLane || imageUnavailable(chosenImageLane)) || !props.frame.images[props.camera.id]} onClick={() => void addPair()}>添加配对 #{pairing.pairs.length + 1}</button>
        <p>一条真值线可对应多条图像投影线；每条图像投影线只能归属一条真值线，同帧不能重复添加。</p>
      </section>
      <section><h3>已保存配对（{pairing.pairs.length}）</h3><p>同一帧的所有线共用训练/验证用途，防止验证泄漏。至少需要一组训练配对；建议增加训练帧和独立验证帧，是否采纳由你决定。</p>
        <ol className="lane-calibration-pairs">{pairing.pairs.map((pair, index) => <li key={pair.id}><strong>#{index + 1} {pair.frame_id} / {pair.lane_id} ↔ {pair.image_lane_id ?? '图像手绘线'}</strong>
          <select aria-label={`配对${index + 1}用途`} value={pair.split} disabled={locked || active} onChange={e => void persist(pairing.pairs.map(p => p.frame_id === pair.frame_id ? {...p, split: e.target.value as Pair['split']} : p))}><option value="train">训练</option><option value="validation">验证</option></select>
          <button aria-label={`上移配对${index + 1}`} disabled={locked || active || index === 0} onClick={() => {const next = [...pairing.pairs]; [next[index-1], next[index]] = [next[index], next[index-1]]; void persist(next);}}>↑</button>
          <button aria-label={`删除配对${index + 1}`} disabled={locked || active} onClick={() => void persist(pairing.pairs.filter(p => p.id !== pair.id))}>×</button>
          {(before || result) && <span className="lane-calibration-pair-error">Mean {metric((result?.before ?? before)?.pairs.find(p => p.id === pair.id)?.mean_px)} → {metric(result?.after.pairs.find(p => p.id === pair.id)?.mean_px)}</span>}</li>)}</ol>
      </section>
      <section><h3>3. 坐标与时间确认</h3>
        <p>三维线使用世界坐标；每帧先执行 inverse(T_WL)，再用同一个待优化的 T_CL 投影，包含 Brown 畸变。</p>
        {state.coordinate_info.fused_map && (state.coordinate_info.sensor_mount_available
          ? <p>已读取全局地图和真实 LiDAR 安装矩阵；自动模式按每帧车辆位姿恢复雷达 pose。</p>
          : <p className="lane-calibration-warning">这是全局地图导入。单位地图注册矩阵不是车辆 pose；自动模式需要真实 LiDAR 安装变换。</p>)}
        <label className="field-label">Pose 来源<select aria-label="车道线优化Pose来源" value={poseMode} disabled={locked || active} onChange={e => {setPoseMode(e.target.value); setBefore(null);}}><option value="auto">已有真实传感器 / 注册位姿</option><option value="ego">已确认 LiDAR 坐标与 ego 重合</option><option value="provided">外部带时间戳 Pose（高级配置）</option></select></label>
        <label><input type="checkbox" aria-label="确认车道线坐标约定" checked={confirmed} disabled={locked || active} onChange={e => setConfirmed(e.target.checked)}/>已确认世界坐标、变换方向、原图分辨率与畸变状态</label>
        <label><input type="checkbox" aria-label="确认图像Pose同步" checked={synchronized} disabled={locked || active} onChange={e => setSynchronized(e.target.checked)}/>已确认每帧图像与 Pose 时间对齐</label>
        <details><summary>外参、时间戳与约束高级配置</summary><p>可提供 initial_matrix / initial_direction、sensor_lidar_to_ego，或 pose_samples、pose_direction、image_timestamps；不允许时间外推。</p><textarea aria-label="车道线优化高级配置" value={advanced} placeholder="JSON 配置（可留空）" disabled={locked || active} onChange={e => setAdvanced(e.target.value)}/></details>
      </section>
      <section><h3>4. 预览与优化</h3><p>{state.runtime.detail}</p>
        <button className="button ghost full-width" aria-label="丢弃当前优化并重新选择" disabled={locked || active} onClick={() => void discard()}>丢弃当前优化并重新选择</button>
        <p>清空当前相机所有帧的配对与候选，保留已保存外参和原始标注。{active ? '请先取消优化，等待结束后再丢弃。' : ''}</p>
        <label className="field-label">优化模式<select aria-label="车道线优化模式" value={mode} disabled={locked || active} onChange={e => setMode(e.target.value)}><option value="multi">多帧共享外参（正式）</option><option value="single">单帧调试（只导出，不应用）</option></select></label>
        <label className="field-label">Adam 迭代次数<input type="number" aria-label="车道线优化迭代次数" min="1" max="1000" value={iterations} disabled={locked || active} onChange={e => setIterations(Number(e.target.value))}/></label>
        <button className="button subtle full-width" aria-label="预览车道线投影误差" disabled={locked || active || !pairing.pairs.length || !confirmed || !synchronized} onClick={() => void calculate(true)}>预览当前投影与误差</button>
        <button className="button primary full-width" aria-label="运行车道线外参优化" disabled={locked || active || !state.runtime.available || !pairing.pairs.length || !confirmed || !synchronized} onClick={() => void calculate(false)}>{operation === 'start' ? '正在提交…' : '运行共享外参优化'}</button>
        <p className="lane-calibration-legend"><span style={{color:'#4ade80'}}>● 固定图像线</span><span style={{color:'#fb923c'}}>● 优化前</span><span style={{color:'#38bdf8'}}>● 优化后</span></p>
      </section>
    </>}
    {job && <section aria-label="车道线优化任务"><h3>任务：{job.status}</h3><p role="status">迭代 {job.iteration} {job.loss === undefined ? '' : `· 损失 ${job.loss.toFixed(4)}`}</p>
      {active && <button className="button ghost" disabled={locked} onClick={() => void cancel()}>取消优化</button>}
      {job.error && <p role="alert">{job.error}</p>}
      {job.status === 'completed' && <a className="button subtle" aria-label="下载车道线优化结果" href={`/api${endpoint}/jobs/${job.id}/download`}>下载矩阵、对比图、损失曲线与误差报告</a>}
    </section>}
    {(before || result) && <section><h3>像素误差（前 → 后）</h3>{(['train', 'validation'] as const).map(split => {
      const a = (result?.before ?? before)?.splits[split], b = result?.after.splits[split];
      return <div key={split} className="lane-calibration-metrics"><h4>{split === 'train' ? '训练帧' : '独立验证帧'} {a?.frames ?? 0} 帧</h4><p>Mean {metric(a?.mean_px)} → {metric(b?.mean_px)}</p><p>RMSE {metric(a?.rmse_px)} → {metric(b?.rmse_px)}</p><p>Median {metric(a?.median_px)} → {metric(b?.median_px)}</p><p>P95 {metric(a?.p95_px)} → {metric(b?.p95_px)}</p>{result?.error_reduction[split] !== null && result?.error_reduction[split] !== undefined && <p>误差下降 {(result.error_reduction[split]! * 100).toFixed(1)}%</p>}</div>;
    })}</section>}
    {result && <section><h3>优化后的 T_CL</h3><pre>{result.optimized_matrix.map(row => row.map(v => v.toFixed(7)).join(' ')).join('\n')}</pre>
      <p>请根据前后投影和误差自行决定是否采纳，质量指标不限制应用。</p>
      {result.rejection_reasons.length > 0 && <p>复核提示：{result.rejection_reasons.join('；')}</p>}
      {job?.apply_block_reason && <p>{job.apply_block_reason}</p>}
      <button className="button primary full-width" aria-label="应用车道线优化外参" disabled={locked || active || !job?.can_apply || job.applied} onClick={() => void apply()}>{job?.applied ? '已应用' : '复核后应用到整段当前相机'}</button>
    </section>}
    {error && <p className="lane-calibration-error" role="alert">{error}</p>}
  </div>;
}
