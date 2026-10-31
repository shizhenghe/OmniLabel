import {useEffect, useId, useRef, useState} from 'react';
import {download, jsonRequest, request} from '../api';
import type {Clip} from '../types';
import './AutomaticCalibrationPanel.css';

type EngineId = 'koide' | 'direct_nid' | 'pjlab';
type JobStatus = 'queued' | 'running' | 'completed' | 'cancelled' | 'failed';
type ItemStatus = 'pending' | 'running' | 'accepted' | 'rejected' | 'failed' | 'skipped' | 'saved' | 'conflict';
interface CalibrationEngine {id: EngineId; name: string; available: boolean; detail: string;}
interface CalibrationItem {
  frame_id: string;
  camera_id: string;
  status: ItemStatus;
  reason?: string;
  metrics?: Record<string, unknown>;
  matrix?: number[][];
}
interface CalibrationJob {
  id: string;
  clip_id: string;
  engine: EngineId;
  status: JobStatus;
  created_at: string;
  updated_at: string;
  total: number;
  processed: number;
  accepted: number;
  rejected: number;
  failed: number;
  skipped: number;
  saved: number;
  items: CalibrationItem[];
  cancel_requested?: boolean;
  error?: string;
}
interface CalibrationOverview {
  engines: CalibrationEngine[];
  latest_job: CalibrationJob | null;
  total_pairs: number;
  saved_pairs: number;
}
export interface AutomaticCalibrationPanelProps {
  clip: Clip;
  frameId?: string;
  cameraId?: string;
  disabled?: boolean;
  onSaved: () => Promise<void> | void;
  onInspect?: (frameId: string, cameraId: string) => Promise<void> | void;
  onClose: () => void;
}

const errorText = (error: unknown) => error instanceof Error ? error.message : String(error);
const jobLabels: Record<JobStatus, string> = {queued: '等待执行', running: '正在配准', completed: '任务完成', cancelled: '已取消', failed: '任务失败'};
const itemLabels: Record<ItemStatus, string> = {pending: '等待', running: '配准中', accepted: '通过待保存', rejected: '未通过', failed: '失败', skipped: '已跳过', saved: '已保存', conflict: '版本冲突'};
const isRunning = (job: CalibrationJob | null) => job?.status === 'queued' || job?.status === 'running';
const engineName = (id: EngineId) => id === 'koide' ? 'Koide' : id === 'pjlab' ? 'PJLab CRLF' : '内置 CPU NID';
const metricValue = (value: unknown) => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(4) : null;
function itemMetric(item: CalibrationItem): string {
  const source = item.metrics?.input_source === 'synchronized_sensor_scan' ? '同步单帧点云' : '';
  const lineBefore = metricValue(item.metrics?.line_overlap_before), lineAfter = metricValue(item.metrics?.line_overlap_after);
  if (lineBefore !== null && lineAfter !== null) return `${source ? source + ' · ' : ''}线特征匹配 ${lineBefore} → ${lineAfter}（需人工复核）`;
  const before = item.metrics?.holdout_before as {nid?: number} | undefined;
  const after = item.metrics?.holdout_after as {nid?: number} | undefined;
  const first = metricValue(before?.nid), last = metricValue(after?.nid);
  if (first !== null && last !== null) return `${source ? source + ' · ' : ''}验证 NID ${first} → ${last}`;
  const rmse = metricValue(item.metrics?.rmse_px);
  if (rmse !== null) return `${source ? source + ' · ' : ''}对应点 RMSE ${rmse} px`;
  return source;
}

/** A server-side job keeps running if this panel is closed; reopening restores it. */
export function AutomaticCalibrationPanel({clip, frameId, cameraId, disabled, onSaved, onInspect, onClose}: AutomaticCalibrationPanelProps) {
  const [overview, setOverview] = useState<CalibrationOverview | null>(null);
  const [job, setJob] = useState<CalibrationJob | null>(null);
  const [engine, setEngine] = useState<EngineId>('koide');
  const [calibrationScope, setCalibrationScope] = useState<'pair' | 'camera' | 'all'>('pair');
  const [overwrite, setOverwrite] = useState(false);
  const [loading, setLoading] = useState(true);
  const [operation, setOperation] = useState<'start' | 'cancel' | 'save' | 'export' | 'inspect' | 'mask' | null>(null);
  const [error, setError] = useState('');
  const [pollError, setPollError] = useState('');
  const [status, setStatus] = useState('');
  const [filter, setFilter] = useState<'all' | 'attention' | 'passed'>('all');
  const [page, setPage] = useState(0);
  const [reload, setReload] = useState(0);
  const panel = useRef<HTMLDivElement>(null);
  const closeButton = useRef<HTMLButtonElement>(null);
  const latest = useRef({operation, onClose});
  latest.current = {operation, onClose};
  const titleId = useId();
  const scope = useRef(clip.id);
  scope.current = clip.id;
  const endpoint = `/clips/${encodeURIComponent(clip.id)}/automatic-calibration`;
  const running = isRunning(job);
  const selectedEngine = overview?.engines.find(value => value.id === engine);
  const selectedCamera = clip.cameras.find(value => value.id === cameraId);
  const selectionCount = calibrationScope === 'pair' ? 1 : calibrationScope === 'camera' ? clip.frames.length : overview?.total_pairs ?? 0;
  const selectionLabel = calibrationScope === 'pair' ? '当前帧 · 当前相机'
    : calibrationScope === 'camera' ? '全部帧 · 当前相机' : '全部帧 · 全部相机';
  const locked = Boolean(disabled || loading || operation);
  const pendingSave = job?.items.filter(item => item.status === 'accepted').length ?? 0;
  const progress = job?.total ? Math.min(100, Math.round(job.processed / job.total * 100)) : 0;
  const filteredItems = (job?.items ?? []).filter(item => filter === 'all' || (filter === 'passed'
    ? item.status === 'accepted' || item.status === 'saved'
    : ['rejected', 'failed', 'skipped', 'conflict'].includes(item.status)));
  const pageSize = 40;
  const pageCount = Math.max(1, Math.ceil(filteredItems.length / pageSize));
  const currentPage = Math.min(page, pageCount - 1);
  const visibleItems = filteredItems.slice(currentPage * pageSize, (currentPage + 1) * pageSize);

  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    setLoading(true); setOverview(null); setJob(null); setError(''); setPollError(''); setStatus(''); setPage(0);
    request<CalibrationOverview>(endpoint, {signal: controller.signal}).then(value => {
      if (!active) return;
      setOverview(value); setJob(value.latest_job);
      setEngine(value.engines.find(item => item.id === 'koide' && item.available)?.id
        ?? value.engines.find(item => item.available)?.id ?? 'koide');
    }).catch(cause => {if (active) setError(`读取自动标定状态失败：${errorText(cause)}`);})
      .finally(() => {if (active) setLoading(false);});
    return () => {active = false; controller.abort();};
  }, [endpoint, reload]);

  useEffect(() => {
    if (!job || !isRunning(job)) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const controller = new AbortController();
    const poll = async () => {
      try {
        const value = await request<CalibrationJob>(`${endpoint}/jobs/${encodeURIComponent(job.id)}`, {signal: controller.signal});
        if (!active) return;
        setJob(value); setPollError('');
        if (isRunning(value)) timer = setTimeout(() => void poll(), 1200);
      } catch (cause) {
        if (!active) return;
        setPollError(`进度暂时无法读取，正在重试：${errorText(cause)}`);
        timer = setTimeout(() => void poll(), 3000);
      }
    };
    timer = setTimeout(() => void poll(), 800);
    return () => {active = false; clearTimeout(timer); controller.abort();};
  }, [endpoint, job?.id, job?.status]);

  useEffect(() => {
    const previous = document.activeElement;
    closeButton.current?.focus();
    const keyboard = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault(); event.stopPropagation();
        if (!latest.current.operation) latest.current.onClose();
      }
      if (event.key !== 'Tab') return;
      const controls = Array.from(panel.current?.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), select:not(:disabled), a[href], [tabindex="0"]') ?? [])
        .filter(value => value.getClientRects().length > 0);
      const first = controls[0], last = controls.at(-1);
      if (!first || !last) {event.preventDefault(); panel.current?.focus(); return;}
      if (event.shiftKey && (document.activeElement === first || !panel.current?.contains(document.activeElement))) {event.preventDefault(); last.focus();}
      else if (!event.shiftKey && (document.activeElement === last || !panel.current?.contains(document.activeElement))) {event.preventDefault(); first.focus();}
    };
    document.addEventListener('keydown', keyboard, true);
    return () => {document.removeEventListener('keydown', keyboard, true); if (previous instanceof HTMLElement && previous.isConnected) previous.focus();};
  }, []);

  const start = async () => {
    if (locked || running || !selectedEngine?.available) return;
    if (calibrationScope !== 'all' && !cameraId) {setError('请先选择一路相机。'); return;}
    if (calibrationScope === 'pair' && !frameId) {setError('请先选择一帧。'); return;}
    const startScope = clip.id;
    setOperation('start'); setError(''); setStatus(''); setPollError('');
    try {
      const selection = calibrationScope === 'pair' ? {frame_ids: [frameId], camera_ids: [cameraId]}
        : calibrationScope === 'camera' ? {camera_ids: [cameraId]} : {};
      const value = await jsonRequest<CalibrationJob>(`${endpoint}/jobs`, {engine, overwrite_saved: overwrite, ...selection});
      if (scope.current !== startScope) return;
      setJob(value); setFilter('all'); setPage(0);
      setStatus('后台任务已启动。关闭此窗口后任务继续，重新打开可查看进度。');
    } catch (cause) {if (scope.current === startScope) setError(`启动失败：${errorText(cause)}`);}
    finally {if (scope.current === startScope) setOperation(null);}
  };
  const cancel = async () => {
    if (locked || !job || !running || job.cancel_requested) return;
    const startScope = clip.id;
    setOperation('cancel'); setError('');
    try {
      const value = await request<CalibrationJob>(`${endpoint}/jobs/${encodeURIComponent(job.id)}/cancel`, {method: 'POST'});
      if (scope.current !== startScope) return;
      setJob(value); setStatus('已请求取消；等待当前组合结束，已完成的候选会保留。');
    } catch (cause) {if (scope.current === startScope) setError(`取消失败：${errorText(cause)}`);}
    finally {if (scope.current === startScope) setOperation(null);}
  };
  const save = async () => {
    if (locked || running || !job || !pendingSave) return;
    const startScope = clip.id;
    setOperation('save'); setError(''); setStatus('');
    try {
      const value = await request<CalibrationJob>(`${endpoint}/jobs/${encodeURIComponent(job.id)}/apply`, {method: 'POST'});
      if (scope.current !== startScope) return;
      setJob(value);
      setStatus(`本任务已有 ${value.saved} 组参数保存到工作区。版本冲突的项目已保留原参数，请查看逐项原因。`);
      try {await onSaved();}
      catch (cause) {setError(`参数已保存，但投影刷新失败：${errorText(cause)}。请重新加载场景以显示最新参数。`);}
      try {
        const updated = await request<CalibrationOverview>(endpoint);
        if (scope.current === startScope) setOverview(updated);
      } catch { /* The saved job remains authoritative if only the summary refresh fails. */ }
    } catch (cause) {if (scope.current === startScope) setError(`保存失败：${errorText(cause)}`);}
    finally {if (scope.current === startScope) setOperation(null);}
  };
  const exportParameters = async () => {
    if (locked) return;
    const startScope = clip.id;
    setOperation('export'); setError('');
    try {
      const data = await request<Record<string, unknown>>(`${endpoint}/export`);
      if (scope.current !== startScope) return;
      download(`${clip.id.replace(/[^a-zA-Z0-9_.-]/g, '_')}-calibration.json`, JSON.stringify(data, null, 2));
      setStatus('已导出当前保存的全 clip 标定参数。未保存候选不会包含在导出文件中。');
    } catch (cause) {if (scope.current === startScope) setError(`导出失败：${errorText(cause)}`);}
    finally {if (scope.current === startScope) setOperation(null);}
  };
  const close = () => {if (!operation) onClose();};
  const uploadMask = async (file: File) => {
    if (!frameId || !cameraId || locked || running) return;
    if (file.size > 16 * 1024 * 1024) {setError('掩码文件不能超过 16 MiB。'); return;}
    const startScope = clip.id;
    setOperation('mask'); setError(''); setStatus('');
    try {
      await request(`${endpoint}/masks/${encodeURIComponent(frameId)}/${encodeURIComponent(cameraId)}`, {
        method: 'PUT', headers: {'Content-Type': 'image/png'}, body: file,
      });
      if (scope.current === startScope) setStatus(`已保存 ${frameId} / ${cameraId} 的 PJLab 线特征掩码。未配置自动提取时，其他帧需分别上传对应掩码。`);
    } catch (cause) {if (scope.current === startScope) setError(`上传掩码失败：${errorText(cause)}`);}
    finally {if (scope.current === startScope) setOperation(null);}
  };
  const inspect = async (item: CalibrationItem) => {
    if (!onInspect || locked || running) return;
    setOperation('inspect'); setError('');
    try {await onInspect(item.frame_id, item.camera_id);}
    catch (cause) {setError(`打开逐帧校准失败：${errorText(cause)}`);}
    finally {setOperation(null);}
  };

  return <div className="modal-backdrop automatic-calibration-backdrop" onClick={close}>
    <div className="automatic-calibration-panel" role="dialog" aria-modal="true" aria-labelledby={titleId} tabIndex={-1} ref={panel} onClick={event => event.stopPropagation()}>
      <div className="automatic-calibration-heading">
        <div><span className="section-eyebrow">IMAGE / LIDAR REGISTRATION</span><h2 id={titleId}>全 clip 自动标定</h2><p>{clip.name} · {clip.frames.length} 帧 × {clip.cameras.length} 路相机</p></div>
        <button ref={closeButton} className="icon-button" aria-label="关闭自动标定" disabled={Boolean(operation)} onClick={close}>×</button>
      </div>
      <div className="automatic-calibration-body">
        <p className="automatic-calibration-intro">逐帧、逐相机独立求解并保存外参，适用于震动引起的逐帧偏差，不会用一组结果覆盖整个 clip。保存后同步用于点云和三维标注投影。若数据包含同步单帧扫描，求解器优先使用它。自动结果需通过质量检查；请复核不同距离的静态物体与道路标线。</p>
        {engine === 'pjlab' && <section className="automatic-calibration-help"><p>PJLab CRLF 需要道路标线/杆体的线特征掩码（原图尺寸 PNG、黑色背景），不可上传原图代替。可配置自动线特征提取，或选中对应帧相机后上传；批量任务会逐项检查掩码。</p>
          <label>当前帧 {frameId} · 相机 {cameraId} 的线特征掩码<input type="file" accept="image/png" aria-label="上传当前帧PJLab线特征掩码" disabled={locked || running || !frameId || !cameraId} onChange={event => {const file = event.target.files?.[0]; event.target.value = ''; if (file) void uploadMask(file);}}/></label>
        </section>}
        {loading ? <p className="automatic-calibration-status" role="status">正在检查配准引擎与已有参数…</p> : overview && <>
          <div className="automatic-calibration-summary"><span>图像 / 点云组合<strong>{overview.total_pairs.toLocaleString()}</strong></span><span>已有参数<strong>{overview.saved_pairs.toLocaleString()}</strong></span><span>本次处理<strong>{selectionCount.toLocaleString()} 组</strong></span></div>
          <label className="automatic-calibration-scope">处理范围<select aria-label="自动标定处理范围" value={calibrationScope} disabled={locked || running} onChange={event => setCalibrationScope(event.target.value as typeof calibrationScope)}><option value="pair" disabled={!frameId || !cameraId}>当前帧 + 当前相机（推荐）</option><option value="camera" disabled={!cameraId}>全部帧 + 当前相机</option><option value="all">全部帧 + 全部相机</option></select><small>{selectionLabel}{cameraId ? ` · ${selectedCamera?.name ?? cameraId}` : ''}{frameId && calibrationScope === 'pair' ? ` · ${frameId}` : ''}</small></label>
          <fieldset className="automatic-calibration-engines" disabled={locked || running}><legend>配准引擎</legend>{overview.engines.map(value => <label key={value.id} className={`automatic-calibration-engine ${engine === value.id ? 'selected' : ''} ${value.available ? '' : 'unavailable'}`}>
            <span><input type="radio" name={titleId + '-engine'} value={value.id} checked={engine === value.id} disabled={!value.available} onChange={() => setEngine(value.id)}/><strong>{value.id === 'direct_nid' ? '内置 CPU NID' : value.name || 'Koide'}</strong><em>{value.available ? '可用' : '未就绪'}</em></span>
            <p>{value.detail || (value.id === 'direct_nid' ? '使用点云反射强度与图像灰度进行局部外参优化。' : '连接 Koide 的实际标定运行环境后可用。')}</p>
          </label>)}</fieldset>
          <p className="automatic-calibration-help">PJLab 使用线特征掩码进行原生标定；Koide 需单独部署，CPU NID 可直接运行。评分通过不代表已测得像素级误差。</p>
          <label className="automatic-calibration-overwrite"><input type="checkbox" checked={overwrite} disabled={locked || running} onChange={event => setOverwrite(event.target.checked)}/>重新计算已有保存参数的组合</label>
          <p className="automatic-calibration-help">{overwrite ? '重新计算只生成候选；点击保存后才替换已保存参数。' : '默认跳过已有保存参数的组合，保留之前的校准。'}</p>
          {calibrationScope === 'all' && engine === 'koide' && <p className="automatic-calibration-warning">Koide 全量任务会逐组运行，当前片段共有 {overview.total_pairs.toLocaleString()} 组；建议先用当前帧和当前相机验证。</p>}
          <div className="automatic-calibration-actions"><button className="button primary" disabled={locked || running || !selectedEngine?.available || !selectionCount} onClick={() => void start()}>{operation === 'start' ? '正在启动…' : running ? `${engineName(job!.engine)} 运行中…` : `启动自动标定（${selectionCount}）`}</button><button className="button subtle" disabled={locked} onClick={() => void exportParameters()}>{operation === 'export' ? '正在导出…' : '导出已保存参数 JSON'}</button></div>
          {pendingSave > 0 && !running && <p className="automatic-calibration-help">上一次任务有待保存候选。需要保留这些结果时，请先保存再启动新任务。</p>}
        </>}
        {job && <section className="automatic-calibration-job" aria-label="自动标定任务进度">
          <div className="automatic-calibration-job-heading"><h3>{job.cancel_requested && running ? '正在取消' : jobLabels[job.status]}</h3><span>{engineName(job.engine)} · {job.processed.toLocaleString()} / {job.total.toLocaleString()}</span>{running && <button className="button ghost" disabled={locked || job.cancel_requested} onClick={() => void cancel()}>{operation === 'cancel' || job.cancel_requested ? '正在取消…' : '取消任务'}</button>}</div>
          <div className="automatic-calibration-progress" role="progressbar" aria-label="图像点云配准进度" aria-valuenow={progress} aria-valuemin={0} aria-valuemax={100}><span style={{width: `${progress}%`}}/></div>
          <div className="automatic-calibration-counts" role="status"><span>通过 <strong>{job.accepted}</strong></span><span>已保存 <strong>{job.saved}</strong></span><span>未通过 <strong>{job.rejected}</strong></span><span>失败 <strong>{job.failed}</strong></span><span>跳过 <strong>{job.skipped}</strong></span></div>
          <div className="automatic-calibration-results-heading"><label>逐项结果<select aria-label="筛选自动标定结果" value={filter} onChange={event => {setFilter(event.target.value as typeof filter); setPage(0);}}><option value="all">全部（{job.items.length}）</option><option value="attention">未通过 / 失败 / 跳过</option><option value="passed">通过 / 已保存</option></select></label><span>帧 / 相机</span></div>
          <ol className="automatic-calibration-results" aria-label="逐帧逐相机标定结果">{visibleItems.map(item => <li key={`${item.frame_id}/${item.camera_id}`} className={`automatic-calibration-item ${item.status}`}>
            <div><strong>{item.frame_id}</strong><span>{clip.cameras.find(value => value.id === item.camera_id)?.name ?? item.camera_id}</span><em>{itemLabels[item.status]}</em></div>
            <p>{item.reason || (item.status === 'accepted' ? '质量检查通过，等待保存。' : item.status === 'saved' ? '已保存并用于投影。' : item.status === 'pending' ? '等待处理。' : item.status === 'running' ? '正在求解与验证。' : '未返回详细原因。')}</p>
            {itemMetric(item) && <small>{itemMetric(item)}</small>}
            {onInspect && <button className="automatic-calibration-inspect" disabled={locked || running} onClick={() => void inspect(item)}>查看该帧 / 手动校准</button>}
          </li>)}</ol>
          {!filteredItems.length && <p className="automatic-calibration-help">当前筛选下没有结果。</p>}
          {pageCount > 1 && <div className="automatic-calibration-pagination"><button className="button ghost" disabled={currentPage === 0} onClick={() => setPage(currentPage - 1)}>上一页</button><span>{currentPage + 1} / {pageCount} 页 · {filteredItems.length} 项</span><button className="button ghost" disabled={currentPage >= pageCount - 1} onClick={() => setPage(currentPage + 1)}>下一页</button></div>}
          <div className="automatic-calibration-save"><div><strong>{pendingSave} 组通过结果待保存</strong><p>仅保存通过质量检查的组合；失败、跳过和冲突项目保留已有参数。</p></div><button className="button primary" disabled={locked || running || !pendingSave} onClick={() => void save()}>{operation === 'save' ? '正在保存…' : `保存通过结果（${pendingSave}）`}</button></div>
          {job.error && <p className="automatic-calibration-error" role="alert">{job.error}</p>}
        </section>}
        {error && <div className="automatic-calibration-error" role="alert">{error}{!overview && !loading && <button className="button ghost" disabled={Boolean(operation)} onClick={() => setReload(value => value + 1)}>重新读取</button>}</div>}
        {pollError && <p className="automatic-calibration-error" role="status">{pollError}</p>}
        {status && <p className="automatic-calibration-status" role="status">{status}</p>}
      </div>
      <div className="automatic-calibration-footer"><span>{running ? '任务在后台执行，关闭窗口后仍会继续。' : '参数保存到工作区，可随场景导出，也可单独导出 JSON。'}</span><button className="button subtle" disabled={Boolean(operation)} onClick={close}>关闭</button></div>
    </div>
  </div>;
}
