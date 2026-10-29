import {useEffect, useRef, useState} from 'react';
import {request} from '../api';
import type {Clip, Vector} from '../types';
import './LatrBatchPanel.css';

type BatchStatus = 'queued' | 'running' | 'completed' | 'cancelled' | 'failed';
type ItemStatus = 'pending' | 'running' | 'succeeded' | 'failed' | 'skipped';
interface BatchParameters {
  camera_id: string;
  frame_ids?: string[] | null;
  label: string;
  score_threshold: number;
  weight_variant: 'once' | 'openlane';
}
interface BatchItem {frame_id: string; status: ItemStatus; vector_count: number; retryable?: boolean; error?: string;}
interface BatchSummary {
  id: string;
  clip_id: string;
  camera_id: string;
  status: BatchStatus;
  total: number;
  processed: number;
  succeeded: number;
  failed: number;
  skipped: number;
  vector_count: number;
  created_at: string;
}
interface BatchJob extends BatchSummary {
  cancel_requested?: boolean;
  scene_revision: number;
  parameters: BatchParameters;
  items: BatchItem[];
  updated_at: string;
  error?: string;
}
interface BatchOverview {latest_job: BatchJob | null; jobs: BatchSummary[];}
export interface BatchProposal {
  job_id: string;
  clip_id: string;
  camera_id: string;
  coordinate_frame: 'world';
  scene_revision: number;
  vectors: Vector[];
  source_frame_ids: string[];
  warnings: string[];
}
export interface LatrBatchPanelProps {
  clip: Clip;
  cameraId: string;
  label: string;
  scoreThreshold: number;
  weightVariant: 'once' | 'openlane';
  disabled: boolean;
  modelAvailable: boolean;
  onApply: (proposal: BatchProposal) => Promise<boolean> | boolean;
  onClose: () => void;
  onError: (message: string) => void;
}

const active = (job: {status: BatchStatus} | null) => job?.status === 'queued' || job?.status === 'running';
const jobLabels: Record<BatchStatus, string> = {queued: '等待预标注', running: '批量预标注中', completed: '批量任务完成', cancelled: '批量任务已取消', failed: '批量任务失败'};
const itemLabels: Record<ItemStatus, string> = {pending: '等待', running: '推理中', succeeded: '成功', failed: '失败', skipped: '跳过'};
const errorText = (cause: unknown) => cause instanceof Error ? cause.message : String(cause);

/** The server owns inference progress. Closing this dialog aborts only local
 * requests, so reopening it can recover both active and partial results. */
export function LatrBatchPanel(props: LatrBatchPanelProps) {
  const [selection, setSelection] = useState<'all' | 'range'>('all');
  const [start, setStart] = useState(1);
  const [count, setCount] = useState(Math.min(50, props.clip.frames.length));
  const [step, setStep] = useState(1);
  const [job, setJob] = useState<BatchJob | null>(null);
  const [history, setHistory] = useState<BatchSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [operation, setOperation] = useState<'start' | 'cancel' | 'apply' | 'load' | null>(null);
  const [error, setError] = useState('');
  const [pollError, setPollError] = useState('');
  const [status, setStatus] = useState('');
  const [filter, setFilter] = useState<'all' | 'succeeded' | 'attention'>('all');
  const [reload, setReload] = useState(0);
  const [appliedId, setAppliedId] = useState<string | null>(null);
  const panel = useRef<HTMLDivElement>(null);
  const closeButton = useRef<HTMLButtonElement>(null);
  const operations = useRef<AbortController | null>(null);
  const mounted = useRef(true);
  const scope = `${props.clip.id}/${props.cameraId}`;
  const latest = useRef({props, scope, operation});
  latest.current = {props, scope, operation};
  const endpoint = `/clips/${encodeURIComponent(props.clip.id)}/latr-batches`;
  const frameCount = props.clip.frames.length;
  const maximumCount = Math.max(1, Math.ceil((frameCount - start + 1) / step));
  const requestedCount = Math.min(count, maximumCount);
  const selectedFrames = selection === 'all' ? props.clip.frames
    : Array.from({length: requestedCount}, (_, index) => props.clip.frames[start - 1 + index * step]).filter(Boolean);
  const missingCount = selectedFrames.filter(frame => !frame.images[props.cameraId]).length;
  const cameraName = (id: string) => props.clip.cameras.find(camera => camera.id === id)?.name ?? id;
  const running = active(job);
  const anotherRunning = history.some(value => value.id !== job?.id && active(value));
  const locked = props.disabled || loading || Boolean(operation);
  const filteredItems = (job?.items ?? []).filter(item => filter === 'all' || (filter === 'succeeded'
    ? item.status === 'succeeded' : item.status === 'failed' || item.status === 'skipped'));
  const failedFrames = (job?.items ?? []).filter(item => item.status === 'failed' || item.retryable).map(item => item.frame_id);
  const progress = job?.total ? Math.min(100, Math.round(job.processed / job.total * 100)) : 0;

  const acceptJob = (value: BatchJob) => {
    setJob(value);
    setHistory(rows => [value, ...rows.filter(row => row.id !== value.id)]);
  };

  useEffect(() => {
    setCount(value => Math.min(Math.max(1, value), maximumCount));
  }, [maximumCount]);

  useEffect(() => {
    mounted.current = true;
    return () => {mounted.current = false; operations.current?.abort();};
  }, []);

  useEffect(() => {
    let valid = true;
    const controller = new AbortController();
    operations.current?.abort(); operations.current = null;
    setLoading(true); setOperation(null); setJob(null); setHistory([]); setError(''); setPollError(''); setStatus(''); setAppliedId(null);
    setSelection('all'); setStart(1); setStep(1); setCount(Math.max(1, Math.min(50, frameCount))); setFilter('all');
    request<BatchOverview>(endpoint, {signal: controller.signal}).then(value => {
      if (!valid || latest.current.scope !== scope) return;
      setHistory(value.jobs ?? []); setJob(value.latest_job);
    }).catch(cause => {
      if (valid && !controller.signal.aborted && latest.current.scope === scope) setError(`读取批量任务失败：${errorText(cause)}`);
    }).finally(() => {if (valid && latest.current.scope === scope) setLoading(false);});
    return () => {valid = false; controller.abort(); operations.current?.abort(); operations.current = null;};
  }, [endpoint, scope, reload]);

  useEffect(() => {
    if (!job || !active(job)) return;
    let valid = true;
    let timer: ReturnType<typeof setTimeout>;
    const controller = new AbortController();
    const poll = async () => {
      try {
        const value = await request<BatchJob>(`${endpoint}/${encodeURIComponent(job.id)}`, {signal: controller.signal});
        if (!valid || latest.current.scope !== scope) return;
        acceptJob(value); setPollError('');
        if (active(value)) timer = setTimeout(() => void poll(), 1000);
      } catch (cause) {
        if (!valid || controller.signal.aborted || latest.current.scope !== scope) return;
        setPollError(`进度暂时不可用，正在重试：${errorText(cause)}`);
        timer = setTimeout(() => void poll(), 2500);
      }
    };
    timer = setTimeout(() => void poll(), 500);
    return () => {valid = false; clearTimeout(timer); controller.abort();};
  }, [endpoint, scope, job?.id, job?.status]);

  useEffect(() => {
    const previous = document.activeElement;
    closeButton.current?.focus();
    const key = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault(); event.stopPropagation();
        if (!latest.current.operation) latest.current.props.onClose();
      }
      if (event.key !== 'Tab') return;
      const controls = [...panel.current?.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), select:not(:disabled), [tabindex="0"]') ?? []]
        .filter(element => element.getClientRects().length > 0);
      const first = controls[0], last = controls.at(-1);
      if (!first || !last) {event.preventDefault(); panel.current?.focus(); return;}
      if (event.shiftKey && (document.activeElement === first || !panel.current?.contains(document.activeElement))) {event.preventDefault(); last.focus();}
      else if (!event.shiftKey && (document.activeElement === last || !panel.current?.contains(document.activeElement))) {event.preventDefault(); first.focus();}
    };
    document.addEventListener('keydown', key, true);
    return () => {document.removeEventListener('keydown', key, true); if (previous instanceof HTMLElement && previous.isConnected) previous.focus();};
  }, []);

  const beginOperation = (name: NonNullable<typeof operation>) => {
    operations.current?.abort();
    const controller = new AbortController(); operations.current = controller;
    setOperation(name); setError(''); setStatus('');
    const originalScope = latest.current.scope;
    return {controller, current: () => mounted.current && !controller.signal.aborted
      && operations.current === controller && latest.current.scope === originalScope};
  };
  const finishOperation = (controller: AbortController, current: () => boolean) => {
    if (current()) {operations.current = null; setOperation(null);}
  };
  const failOperation = (message: string) => {setError(message); latest.current.props.onError(message);};

  const startJob = async (retry = false) => {
    if (locked || running || anotherRunning || !props.modelAvailable) return;
    if (retry ? !job || !failedFrames.length : !selectedFrames.length || !props.cameraId || !props.label.trim()) return;
    const parameters: BatchParameters = {camera_id: props.cameraId, label: props.label.trim(), score_threshold: props.scoreThreshold,
      weight_variant: props.weightVariant, ...(selection === 'all' ? {} : {frame_ids: selectedFrames.map(frame => frame.id)})};
    const {controller, current} = beginOperation('start');
    try {
      const value = await request<BatchJob>(retry && job ? `${endpoint}/${encodeURIComponent(job.id)}/retry` : endpoint, {method: 'POST', signal: controller.signal,
        headers: {'Content-Type': 'application/json'}, body: retry ? undefined : JSON.stringify(parameters)});
      if (!current()) return;
      acceptJob(value); setFilter('all'); setAppliedId(null);
      setStatus('批量任务已启动，关闭面板后会继续运行。完成后应用候选，再使用车道线融合。');
    } catch (cause) {if (current()) failOperation(`启动批量预标注失败：${errorText(cause)}`);}
    finally {finishOperation(controller, current);}
  };
  const cancel = async () => {
    if (locked || !job || !running || job.cancel_requested) return;
    const {controller, current} = beginOperation('cancel');
    try {
      const value = await request<BatchJob>(`${endpoint}/${encodeURIComponent(job.id)}/cancel`, {method: 'POST', signal: controller.signal});
      if (!current()) return;
      acceptJob(value); setStatus('已请求取消；正在推理的帧结束后停止，已完成的候选会保留。');
    } catch (cause) {if (current()) failOperation(`取消请求失败：${errorText(cause)}`);}
    finally {finishOperation(controller, current);}
  };
  const loadJob = async (id: string) => {
    if (locked || id === job?.id) return;
    const {controller, current} = beginOperation('load');
    try {
      const value = await request<BatchJob>(`${endpoint}/${encodeURIComponent(id)}`, {signal: controller.signal});
      if (current()) {acceptJob(value); setFilter('all');}
    } catch (cause) {if (current()) failOperation(`读取批量任务失败：${errorText(cause)}`);}
    finally {finishOperation(controller, current);}
  };
  const apply = async () => {
    if (locked || !job || running || !job.vector_count || appliedId === job.id) return;
    const id = job.id;
    const {controller, current} = beginOperation('apply');
    try {
      const proposal = await request<BatchProposal>(`${endpoint}/${encodeURIComponent(id)}/proposal`, {signal: controller.signal});
      if (!current()) return;
      if (proposal.job_id !== id || proposal.clip_id !== props.clip.id || proposal.camera_id !== job.camera_id || proposal.coordinate_frame !== 'world') {
        throw new Error('批量候选与当前任务不对应，请刷新任务状态。');
      }
      const applied = await latest.current.props.onApply(proposal);
      if (!current()) return;
      if (applied) {setAppliedId(id); setStatus('批量候选已加入场景草稿。关闭面板后可融合重复车道线，审核后保存。');}
    } catch (cause) {if (current()) failOperation(`批量候选未应用：${errorText(cause)}`);}
    finally {finishOperation(controller, current);}
  };
  const close = () => {if (!operation) props.onClose();};

  return <div className="modal-backdrop latr-batch-backdrop" onClick={close}>
    <div className="latr-batch-panel" role="dialog" aria-modal="true" aria-label="LATR批量预标注" tabIndex={-1}
      ref={panel} onClick={event => event.stopPropagation()}>
      <div className="latr-batch-heading"><div><span className="section-eyebrow">LATR / CLIP</span><h2>LATR 批量预标注</h2>
        <p>{props.clip.name} · 当前相机：{cameraName(props.cameraId)} · {frameCount} 帧</p></div>
        <button className="icon-button" ref={closeButton} aria-label="关闭LATR批量预标注" disabled={Boolean(operation)} onClick={close}>×</button></div>
      <div className="latr-batch-body">
        <div className="latr-batch-selection"><label className="field-label">图像范围<select aria-label="批量预标注范围" value={selection}
          disabled={locked || running || anotherRunning} onChange={event => setSelection(event.target.value as 'all' | 'range')}>
          <option value="all">当前相机 · 全部图像</option><option value="range">指定图像数量</option></select></label>
          {selection === 'range' && <div className="latr-batch-range">
            <label className="field-label">起始帧<input type="number" min="1" max={frameCount} step="1" aria-label="批量起始帧" value={start}
              disabled={locked || running || anotherRunning} onChange={event => {if (Number.isFinite(event.target.valueAsNumber)) setStart(Math.min(frameCount, Math.max(1, Math.floor(event.target.valueAsNumber))));}}/></label>
            <label className="field-label">图像数量<input type="number" min="1" max={maximumCount} step="1" aria-label="批量图像数量" value={requestedCount}
              disabled={locked || running || anotherRunning} onChange={event => {if (Number.isFinite(event.target.valueAsNumber)) setCount(Math.min(maximumCount, Math.max(1, Math.floor(event.target.valueAsNumber))));}}/></label>
            <label className="field-label">每隔几帧取一张<input type="number" min="1" max={frameCount} step="1" aria-label="批量抽帧间隔" value={step}
              disabled={locked || running || anotherRunning} onChange={event => {if (Number.isFinite(event.target.valueAsNumber)) setStep(Math.min(frameCount, Math.max(1, Math.floor(event.target.valueAsNumber))));}}/></label>
          </div>}
          <p className="latr-batch-help" data-testid="latr-batch-selection">已选择 {selectedFrames.length} 帧图像；{selectedFrames.length - missingCount} 张可用{missingCount ? `，${missingCount} 帧缺少当前相机图像，将跳过` : ''}。
            {selection === 'range' && selectedFrames.length > 0 && <> 首帧 {selectedFrames[0].id}，末帧 {selectedFrames.at(-1)!.id}。</>}</p>
          <p className="latr-batch-help">权重 {props.weightVariant.toUpperCase()} · 类别 {props.label} · 置信度 ≥ {props.scoreThreshold}。每帧候选统一转换到世界坐标，批量阶段保留独立折线，之后可预览融合。</p>
          <button className="button primary" aria-label="启动LATR批量预标注" disabled={locked || running || anotherRunning || !props.modelAvailable || !props.cameraId || !props.label.trim() || !selectedFrames.length}
            onClick={() => void startJob()}>{operation === 'start' ? '正在启动…' : `批量预标注 ${selectedFrames.length} 张图像`}</button>
          {!props.modelAvailable && <p className="latr-batch-help">请先配置并检查 LATR 推理服务；已有批量候选仍可查看。</p>}
        </div>
        {loading && <p className="latr-batch-status" role="status">正在读取批量任务…</p>}
        {history.length > 1 && <label className="field-label latr-batch-history">历史任务<select aria-label="查看批量任务" value={job?.id ?? ''} disabled={locked || running}
          onChange={event => void loadJob(event.target.value)}>{history.map(value => <option key={value.id} value={value.id}>
            {cameraName(value.camera_id)} · {jobLabels[value.status]} · {value.processed}/{value.total} · {value.id.slice(0, 8)}
          </option>)}</select></label>}
        {job && <section className="latr-batch-job" data-testid="latr-batch-job" data-job-id={job.id} data-status={job.status} data-camera-id={job.camera_id}>
          <div className="latr-batch-job-heading"><h3>{jobLabels[job.status]}</h3><span>{cameraName(job.camera_id)} · {job.parameters.weight_variant.toUpperCase()} · {job.parameters.label}</span>
            {running && <button className="button ghost" aria-label="取消批量预标注" disabled={locked || job.cancel_requested} onClick={() => void cancel()}>{job.cancel_requested ? '正在取消…' : '取消后续帧'}</button>}</div>
          <div className="latr-batch-progress" role="progressbar" aria-label="批量预标注进度" aria-valuemin={0} aria-valuemax={job.total} aria-valuenow={job.processed}><span style={{width: `${progress}%`}}/></div>
          <div className="latr-batch-counts"><span>已处理 <strong>{job.processed} / {job.total}</strong></span><span>成功 <strong>{job.succeeded}</strong></span>
            <span>失败 <strong>{job.failed}</strong></span><span>跳过 <strong>{job.skipped}</strong></span><span>车道线候选 <strong>{job.vector_count}</strong></span></div>
          {job.error && <p className="latr-batch-error" role="alert">{job.error}</p>}
          <div className="latr-batch-results-heading"><label>帧结果<select aria-label="批量帧结果筛选" value={filter} onChange={event => setFilter(event.target.value as typeof filter)}>
            <option value="all">全部帧</option><option value="succeeded">成功帧</option><option value="attention">失败 / 跳过</option></select></label><span>{filteredItems.length} 帧</span></div>
          <ol className="latr-batch-results" aria-label="批量预标注逐帧结果">{filteredItems.map(item => <li key={item.frame_id} className={`latr-batch-item ${item.status}`}
            data-frame-id={item.frame_id} data-status={item.status}><div><strong>{item.frame_id}</strong><span>{itemLabels[item.status]}</span>
              {item.status === 'succeeded' && <small>{item.vector_count} 条候选</small>}</div>{item.error && <p>{item.error}</p>}</li>)}</ol>
          <div className="latr-batch-apply"><div><strong>预标注结果尚未保存为标注</strong><p>应用成功帧的候选后，使用「融合重复车道线」整合整段 clip，审核并保存。</p></div>
            <button className="button primary" aria-label="应用批量候选" disabled={locked || running || !job.vector_count || appliedId === job.id} onClick={() => void apply()}>
              {operation === 'apply' ? '读取候选中…' : appliedId === job.id ? '已加入场景草稿' : '应用批量候选'}</button></div>
          {!running && failedFrames.length > 0 && <button className="button subtle" aria-label="重试失败帧" disabled={locked || anotherRunning || !props.modelAvailable} onClick={() => void startJob(true)}>重试失败 / 未完成帧（{failedFrames.length}）</button>}
          {job.status === 'cancelled' && <p className="latr-batch-help">已取消后续处理；成功完成的 {job.succeeded} 帧候选仍可应用。</p>}
        </section>}
        {error && <div className="latr-batch-error" role="alert">{error}<button className="button ghost" disabled={Boolean(operation)} onClick={() => setReload(value => value + 1)}>刷新任务状态</button></div>}
        {pollError && <p className="latr-batch-error" role="alert">{pollError}</p>}
        {status && <p className="latr-batch-status" role="status">{status}</p>}
      </div>
      <footer className="latr-batch-footer"><span>关闭面板不会停止后台任务。成功候选保留，应用与保存由你确认。</span>
        <button className="button subtle" disabled={Boolean(operation)} onClick={close}>关闭</button></footer>
    </div>
  </div>;
}
