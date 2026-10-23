import {useCallback, useEffect, useMemo, useRef, useState} from 'react';
import {CameraView} from './components/CameraView';
import {PointCloudView} from './components/PointCloudView';
import {Icon, type IconName} from './components/Icon';
import {request, jsonRequest, download} from './api';
import type {Annotations, Box, Clip, ClipSummary, Frame, ModelInfo, ProjectionPoint, Vec3, Voxel} from './types';

const EMPTY: Annotations = {boxes: [], vectors: [], image_polygons: [], point_labels: {}, revision: 0};
const tasks: {id: string; name: string; short: string; icon: IconName; subtitle: string}[] = [
  {id: 'detection', name: '3D 目标检测', short: '3D', icon: 'cube', subtitle: '三维目标与跨帧轨迹'},
  {id: 'fusion', name: '4D 点云地图', short: '4D', icon: 'layers', subtitle: '多帧位姿融合 · 静态场景重建'},
  {id: 'lane', name: '车道线标注', short: 'LANE', icon: 'lane', subtitle: '在俯视图中绘制三维车道线'},
  {id: 'vector', name: '矢量地图', short: 'MAP', icon: 'map', subtitle: '道路边界、停止线与面要素'},
  {id: 'image', name: '图像语义分割', short: '2D', icon: 'image', subtitle: '原始分辨率多边形语义标注'},
  {id: 'semantic', name: '点云语义分割', short: 'SEG', icon: 'points', subtitle: '逐点语义标注与地面辅助分割'},
  {id: 'occupancy', name: 'OCC 真值转换', short: 'OCC', icon: 'grid', subtitle: '稀疏体素 · 占据 / 自由 / 未知'},
];
const objectClasses = [{id: 'vehicle', name: '车辆', color: '#37ddbf'}, {id: 'car', name: '汽车', color: '#37ddbf'}, {id: 'truck', name: '卡车', color: '#fbad5c'}, {id: 'pedestrian', name: '行人', color: '#f476aa'}, {id: 'cyclist', name: '骑行者', color: '#aa96ff'}];
const semanticClasses = [{id: 'road', name: '路面', color: '#65acff'}, {id: 'vehicle', name: '车辆', color: '#37ddbf'}, {id: 'building', name: '建筑', color: '#c39dff'}, {id: 'vegetation', name: '植被', color: '#a3d465'}, {id: 'pedestrian', name: '行人', color: '#f476aa'}, {id: 'unlabeled', name: '未分类', color: '#98a8b8'}];
const vectorClasses = [{id: 'lane', name: '车道线', color: '#f7d16a'}, {id: 'road_boundary', name: '道路边界', color: '#37ddbf'}, {id: 'stop_line', name: '停止线', color: '#f476aa'}, {id: 'crosswalk', name: '人行横道', color: '#aa96ff'}];
type Tool = 'select' | 'box' | 'vector' | 'brush';
type Notice = {message: string; error?: boolean};
type JobView = {id: string; kind: string; status: string; error?: string; result?: Record<string, unknown>};
const uid = (prefix: string) => `${prefix}-${crypto.randomUUID().slice(0, 8)}`;

function worldToLidar(points: number[][], frame: Frame) {
  const a = frame.ego_to_world, b = frame.lidar_to_ego;
  const m = a.map((row, i) => row.map((_, j) => row.reduce((s, value, k) => s + value * b[k][j], 0)));
  return points.map(p => {
    const v = [p[0] - m[0][3], p[1] - m[1][3], p[2] - m[2][3]];
    return [m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2], m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2], m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2], ...p.slice(3)];
  });
}

export default function App() {
  const [clips, setClips] = useState<ClipSummary[]>([]);
  const [clip, setClip] = useState<Clip | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [annotations, setAnnotations] = useState<Annotations>(EMPTY);
  const [dirty, setDirty] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [task, setTask] = useState('detection');
  const [tool, setTool] = useState<Tool>('select');
  const [view, setView] = useState<'perspective' | 'top'>('perspective');
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [classId, setClassId] = useState('vehicle');
  const [draftVector, setDraftVector] = useState<Vec3[]>([]);
  const [draftPolygon, setDraftPolygon] = useState<number[][]>([]);
  const [activeCamera, setActiveCamera] = useState('');
  const [expandedCamera, setExpandedCamera] = useState<string | null>(null);
  const [projection, setProjection] = useState(true);
  const [projections, setProjections] = useState<Record<string, ProjectionPoint[]>>({});
  const [showBoxes, setShowBoxes] = useState(true);
  const [showVectors, setShowVectors] = useState(true);
  const [playing, setPlaying] = useState(false);
  const [brushRadius, setBrushRadius] = useState(1.5);
  const [voxelSize, setVoxelSize] = useState(0.5);
  const [showFree, setShowFree] = useState(false);
  const [notice, setNotice] = useState<Notice | null>(null);
  const [job, setJob] = useState<JobView | null>(null);
  const [fusionPoints, setFusionPoints] = useState<number[][] | undefined>();
  const [fusionLabels, setFusionLabels] = useState<Record<string, string>>({});
  const [groundZ, setGroundZ] = useState(-1.7);
  const [workerModel, setWorkerModel] = useState('sam2');
  const [workerPrompt, setWorkerPrompt] = useState('');
  const [proposal, setProposal] = useState<{annotations: Annotations; frameId: string; clipId: string; revision: number; model: string} | null>(null);
  const [occupancy, setOccupancy] = useState<{frameId: string; voxels: Voxel[]; size: number} | null>(null);
  const [modelList, setModelList] = useState<ModelInfo[] | null>(null);
  const [modal, setModal] = useState<'models' | 'help' | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const requestGeneration = useRef(0);
  const context = useRef('');
  const dirtyRef = useRef(false);
  const editGeneration = useRef(0);
  const runningJob = useRef(false);
  const frame = clip?.frames[frameIndex] ?? null;
  const selectedBox = annotations.boxes.find(b => b.id === selectedId);
  const selectedVector = annotations.vectors.find(v => v.id === selectedId);
  const selectedPolygon = annotations.image_polygons.find(p => p.id === selectedId);
  const taskInfo = tasks.find(t => t.id === task)!;
  const classes = task === 'lane' || task === 'vector' ? vectorClasses : task === 'image' || task === 'semantic' || task === 'occupancy' ? semanticClasses : objectClasses;
  const busy = loading || saving;
  const jobBusy = job?.status === 'running' || job?.status === 'queued';
  const currentKey = clip && frame ? `${clip.id}/${frame.id}` : '';
  const overlayPoints = useMemo(() => fusionPoints && frame ? worldToLidar(fusionPoints, frame) : undefined, [fusionPoints, frame]);
  context.current = currentKey;
  dirtyRef.current = dirty;

  const flash = useCallback((message: string, error = false) => setNotice({message, error}), []);
  useEffect(() => { if (!notice) return; const t = setTimeout(() => setNotice(null), notice.error ? 9000 : 4000); return () => clearTimeout(t); }, [notice]);

  const refreshClips = useCallback(async () => {
    const rows = await request<ClipSummary[]>('/clips');
    setClips(rows);
    return rows;
  }, []);

  const loadClip = useCallback(async (id: string) => {
    const generation = ++requestGeneration.current;
    setPlaying(false); setLoading(true);
    try {
      const next = await request<Clip>(`/clips/${encodeURIComponent(id)}`);
      const first = next.frames[0];
      const saved = first ? await request<Annotations>(`/clips/${encodeURIComponent(id)}/frames/${encodeURIComponent(first.id)}/annotations`) : EMPTY;
      if (generation !== requestGeneration.current) return;
      setClip(next); setFrameIndex(0); setAnnotations(saved); setDirty(false);
      setSelectedId(null); setDraftVector([]); setDraftPolygon([]); setActiveCamera(next.cameras[0]?.id ?? ''); setExpandedCamera(null); setOccupancy(null); setFusionPoints(undefined); setFusionLabels({}); setProposal(null); setJob(null);
    } catch (e) { if (generation === requestGeneration.current) flash(`读取 clip 失败：${String(e instanceof Error ? e.message : e)}`, true); }
    finally { if (generation === requestGeneration.current) setLoading(false); }
  }, [flash]);

  useEffect(() => {
    let active = true;
    refreshClips().then(rows => { if (active && rows[0]) return loadClip(rows[0].id); if (active) setLoading(false); }).catch(e => { if (active) { flash(`连接服务失败：${e.message}`, true); setLoading(false); } });
    return () => {active = false;};
  }, [refreshClips, loadClip, flash]);

  useEffect(() => {
    const guard = (event: BeforeUnloadEvent) => {if (dirtyRef.current) {event.preventDefault(); event.returnValue = '';}};
    window.addEventListener('beforeunload', guard); return () => window.removeEventListener('beforeunload', guard);
  }, []);

  useEffect(() => {
    if (!clip || !frame || !projection) {setProjections({}); return;}
    let active = true;
    setProjections({});
    Promise.all(clip.cameras.map(camera => jsonRequest<{points: ProjectionPoint[]}>(`/clips/${encodeURIComponent(clip.id)}/project`, {frame_id: frame.id, camera_id: camera.id, stride: 2}).then(result => [camera.id, result.points] as const)))
      .then(results => {if (active) setProjections(Object.fromEntries(results));})
      .catch(e => {if (active) flash(`点云投影失败：${e.message}`, true);});
    return () => {active = false;};
  }, [clip, frame, projection, flash]);

  const confirmDiscard = () => {
    if (!dirty && !draftVector.length && !draftPolygon.length) return true;
    return window.confirm('当前帧有未保存的标注或未完成的绘制。放弃这些修改并继续？');
  };

  const moveFrame = useCallback(async (index: number, playback = false) => {
    if (!clip || busy || index === frameIndex || index < 0 || index >= clip.frames.length) return;
    if (!playback && !confirmDiscard()) return;
    if (playback && dirtyRef.current) {setPlaying(false); return;}
    const generation = ++requestGeneration.current;
    setLoading(true);
    try {
      const next = await request<Annotations>(`/clips/${encodeURIComponent(clip.id)}/frames/${encodeURIComponent(clip.frames[index].id)}/annotations`);
      if (generation !== requestGeneration.current) return;
      setFrameIndex(index); setAnnotations(next); setDirty(false); setSelectedId(null); setDraftVector([]); setDraftPolygon([]);
    } catch (e) {flash(`切换帧失败：${e instanceof Error ? e.message : e}`, true); setPlaying(false);}
    finally {if (generation === requestGeneration.current) setLoading(false);}
  }, [clip, busy, frameIndex, dirty, draftVector.length, draftPolygon.length, flash]);

  useEffect(() => {
    if (!playing || !clip || busy) return;
    if (frameIndex >= clip.frames.length - 1) {setPlaying(false); return;}
    const timer = setTimeout(() => void moveFrame(frameIndex + 1, true), Math.max(1000 / clip.fps, 180));
    return () => clearTimeout(timer);
  }, [playing, clip, frameIndex, busy, moveFrame]);

  const edit = (change: (a: Annotations) => Annotations) => {
    if (busy || jobBusy) return;
    setAnnotations(a => change(a)); setDirty(true); dirtyRef.current = true; editGeneration.current++; setPlaying(false);
  };

  const save = async (): Promise<boolean> => {
    if (!clip || !frame || saving || loading) return false;
    if (!dirty) return true;
    const key = currentKey, generation = editGeneration.current;
    setSaving(true); setPlaying(false);
    try {
      const saved = await jsonRequest<Annotations>(`/clips/${encodeURIComponent(clip.id)}/frames/${encodeURIComponent(frame.id)}/annotations`, annotations, 'PUT');
      if (context.current === key && generation === editGeneration.current) {setAnnotations(saved); setDirty(false); dirtyRef.current = false;}
      flash('标注已保存到工作区'); return true;
    } catch (e) {flash(`保存失败：${e instanceof Error ? e.message : e}。当前修改仍保留，可在右下角下载当前帧草稿后重新载入。`, true); return false;}
    finally {setSaving(false);}
  };

  const switchTask = (next: string) => {
    if ((draftVector.length || draftPolygon.length) && !window.confirm('切换任务将放弃尚未完成的绘制，是否继续？')) return;
    setTask(next); setTool('select'); setDraftVector([]); setDraftPolygon([]); setPlaying(false);
    setClassId(next === 'lane' || next === 'vector' ? 'lane' : ['image', 'semantic', 'occupancy'].includes(next) ? 'road' : 'vehicle');
    if (next === 'lane' || next === 'vector') setView('top');
  };

  const createBox = (center: Vec3) => {
    const id = uid('box');
    edit(a => ({...a, boxes: [...a.boxes, {id, track_id: uid('track'), label: classId, center: [center[0], center[1], center[2] + 0.8], size: [4.2, 1.8, 1.6], yaw: 0}]}));
    setSelectedId(id); setTool('select');
  };

  const updateBox = (patch: Partial<Box>) => edit(a => ({...a, boxes: a.boxes.map(box => box.id === selectedId ? {...box, ...patch} : box)}));
  const deleteSelected = () => {edit(a => ({...a, boxes: a.boxes.filter(b => b.id !== selectedId), vectors: a.vectors.filter(v => v.id !== selectedId), image_polygons: a.image_polygons.filter(p => p.id !== selectedId)})); setSelectedId(null);};

  const chooseCamera = (id: string) => {
    if (activeCamera === id) return true;
    if (draftPolygon.length && !window.confirm('放弃当前相机未完成的多边形？')) return false;
    setActiveCamera(id); setDraftPolygon([]); return true;
  };

  const finishVector = () => {
    const closed = classId === 'crosswalk';
    if (draftVector.length < (closed ? 3 : 2)) {flash(closed ? '面要素至少需要 3 个顶点' : '折线至少需要 2 个顶点', true); return;}
    const id = uid('vector');
    edit(a => ({...a, vectors: [...a.vectors, {id, label: classId, points: draftVector, closed}]})); setDraftVector([]); setSelectedId(id); setTool('select');
  };

  const finishPolygon = () => {
    if (draftPolygon.length < 3) {flash('多边形至少需要 3 个顶点', true); return;}
    const id = uid('mask');
    edit(a => ({...a, image_polygons: [...a.image_polygons, {id, camera_id: activeCamera, label: classId, points: draftPolygon}]})); setDraftPolygon([]); setSelectedId(id);
  };

  const importFile = async (file?: File) => {
    if (!file || !confirmDiscard()) return;
    setPlaying(false); setLoading(true);
    const generation = ++requestGeneration.current;
    try {
      const body = new FormData(); body.append('file', file);
      const next = await request<Clip>('/clips/import', {method: 'POST', body});
      if (generation !== requestGeneration.current) return;
      await refreshClips(); await loadClip(next.id); flash(`已导入 ${next.name} · ${next.frames.length} 帧`);
    } catch (e) {flash(`导入失败：${e instanceof Error ? e.message : e}`, true); setLoading(false);}
  };

  const exportClip = async () => {
    if (draftPolygon.length || draftVector.length) {flash('请先完成或取消绘制，再导出 clip', true); return;}
    if (!clip || !(await save())) return;
    try {const response = await fetch(`/api/clips/${encodeURIComponent(clip.id)}/export.zip`); if (!response.ok) throw new Error(`HTTP ${response.status}`); download(`${clip.id}-dataset.zip`, await response.blob(), 'application/zip'); flash('已导出可迁移 ZIP：图像、点云、标定和已保存标注');}
    catch (e) {flash(`导出失败：${e instanceof Error ? e.message : e}`, true);}
  };

  const exportMask = async () => {
    if (!clip || !frame || !activeCamera || !(await save())) return;
    try {
      const path = `/clips/${encodeURIComponent(clip.id)}/frames/${encodeURIComponent(frame.id)}/masks/${encodeURIComponent(activeCamera)}`;
      const response = await fetch(`/api${path}.zip`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      download(`${frame.id}-${activeCamera}-mask.zip`, await response.blob(), 'application/zip');
      flash('已导出同版本 16 位类别 ID 掩码与标签映射 ZIP');
    } catch (e) {flash(`掩码导出失败：${e instanceof Error ? e.message : e}`, true);}
  };

  const runJob = async (kind: string) => {
    if (!clip || !frame || runningJob.current || busy) return;
    if (kind === 'propagate' && !clip.frames[frameIndex + 1]) {flash('当前已是最后一帧，没有下一帧可传播', true); return;}
    if (!(await save())) return;
    runningJob.current = true; setPlaying(false); setJob({id: '', kind, status: 'queued'});
    const key = currentKey, clipId = clip.id, frameId = frame.id;
    try {
      let next = await jsonRequest<JobView>(`/clips/${encodeURIComponent(clip.id)}/jobs`, {kind, frame_id: frame.id, voxel_size: voxelSize, ...(kind === 'prelabel' ? {model_id: workerModel, prompt: workerPrompt} : {}), ...(kind === 'propagate' && clip.frames[frameIndex + 1] ? {target_frame_id: clip.frames[frameIndex + 1].id} : {})});
      setJob(next);
      let iterations = 0;
      while (next.status === 'queued' || next.status === 'running') {
        if (++iterations > 480) throw new Error('任务等待超时，可通过任务 ID 查询后台状态');
        await new Promise(resolve => setTimeout(resolve, 400));
        next = await request<JobView>(`/jobs/${encodeURIComponent(next.id)}`); setJob(next);
      }
      if (next.status === 'failed') throw new Error(next.error || '计算任务失败');
      if (context.current !== key) {flash('后台任务已完成，请返回原 clip 查看结果'); return;}
      if (kind === 'map') {setFusionPoints(next.result?.points as number[][]); setFusionLabels((next.result?.point_labels ?? {}) as Record<string, string>);}
      else if (kind === 'occupancy') setOccupancy({frameId, voxels: (next.result?.voxels ?? []) as Voxel[], size: Number(next.result?.voxel_size ?? voxelSize)});
      else if (kind === 'prelabel') setProposal({annotations: next.result?.proposal as Annotations, frameId, clipId, revision: Number((next.result?.proposal as Annotations)?.revision), model: String(next.result?.model_id ?? workerModel)});
      else if (!dirtyRef.current) {
        const beforeRefresh = editGeneration.current;
        const saved = await request<Annotations>(`/clips/${encodeURIComponent(clipId)}/frames/${encodeURIComponent(frameId)}/annotations`);
        if (context.current === key && !dirtyRef.current && beforeRefresh === editGeneration.current) setAnnotations(saved);
      }
      flash(kind === 'prelabel' ? '候选标注已生成，等待人工审核与应用' : kind === 'propagate' ? '静态轨迹已按位姿传播至下一帧，请人工核验运动目标' : '计算完成，结果已载入');
    } catch (e) {const error = e instanceof Error ? e.message : String(e); setJob(j => j ? {...j, status: 'failed', error} : {id: '', kind, status: 'failed', error}); flash(`任务失败：${error}`, true);}
    finally {runningJob.current = false;}
  };

  const openModels = async () => {setModal('models'); try {setModelList((await request<{models: ModelInfo[]}>('/models')).models);} catch (e) {flash(`模型目录读取失败：${e instanceof Error ? e.message : e}`, true);}};

  const applyProposal = () => {
    if (!proposal || !clip || !frame || proposal.clipId !== clip.id || proposal.frameId !== frame.id || proposal.revision !== annotations.revision || dirty) {flash('候选标注已过期或当前帧有修改，请重新生成候选', true); return;}
    if (!window.confirm('应用候选将替换当前帧标注。请在应用后人工审核并保存，是否继续？')) return;
    edit(() => ({...proposal.annotations, revision: annotations.revision})); setProposal(null); setModal(null); flash('候选已应用为未保存修改，请人工核验后保存');
  };

  const cameraCards = (expanded = false) => clip?.cameras.filter(c => !expanded || c.id === expandedCamera).map(camera => <div className={`camera-card ${activeCamera === camera.id ? 'camera-active' : ''}`} key={camera.id}>
    <CameraView camera={camera} frame={frame} annotations={annotations} projections={projections[camera.id] ?? []} selected={activeCamera === camera.id} tool={task === 'image' && !busy && !jobBusy ? 'polygon' : 'select'} onSelect={() => chooseCamera(camera.id)} onAddPolygonPoint={point => {if (busy || jobBusy) return; if (activeCamera !== camera.id) {setActiveCamera(camera.id); setDraftPolygon([point]);} else setDraftPolygon(p => [...p, point]); setPlaying(false);}} onFinishPolygon={finishPolygon} draftPolygon={activeCamera === camera.id ? draftPolygon : []} showProjection={projection} showBoxes={showBoxes} selectedId={selectedId}/>
    <button className="camera-expand icon-button" aria-label={`放大 ${camera.name}`} title={`放大 ${camera.name}`} onClick={() => {if (!chooseCamera(camera.id)) return; setExpandedCamera(expanded ? null : camera.id);}}><Icon name={expanded ? 'close' : 'expand'} size={13}/></button>
  </div>);

  return <div className="app-shell">
    <header className="topbar">
      <div className="brand"><span className="brand-mark"><Icon name="cube" size={26}/></span><span>Omni<span className="brand-light">Label</span><small>MULTIMODAL ANNOTATION</small></span><span className="version">LAB</span></div>
      <div className="workspace-breadcrumb"><span>工作空间</span><Icon name="chevron" size={12}/><strong>多任务标注</strong></div>
      <div className="topbar-actions"><span className={`save-status ${dirty ? 'is-dirty' : ''}`}><span className="status-dot"/>{saving ? '正在保存…' : dirty ? '有未保存修改' : '已同步至工作区'}</span><button className="button ghost" onClick={() => setModal('help')} title="数据规范与操作说明"><Icon name="help" size={16}/><span>使用说明</span></button><button className="button primary" onClick={() => void save()} disabled={busy || !dirty} aria-label="保存标注"><Icon name="save" size={15}/>{saving ? '保存中' : '保存标注'}</button><span className="avatar">OL</span></div>
    </header>

    <aside className="sidebar">
      <div className="nav-label">标注任务 <span>WORKSPACE</span></div>
      <nav aria-label="标注任务">{tasks.map(t => <button key={t.id} className={`task-button ${task === t.id ? 'active' : ''}`} onClick={() => switchTask(t.id)}><Icon name={t.icon} size={19}/><span>{t.name}</span><small>{t.short}</small>{task === t.id && <i/>}</button>)}</nav>
      <div className="sidebar-divider"/><div className="nav-label">自动化工具</div>
      <button className="task-button model-link" onClick={() => void openModels()}><Icon name="model" size={19}/><span>模型与预标注</span><span className="small-badge">接入</span></button>
      <div className="sidebar-bottom"><div className="sensor-orbit"><Icon name="cube" size={25}/><span/><span/><span/></div><strong>统一时空 · 多模态协同</strong><p>7 路环视 / LiDAR / 标定参数<br/>一个 clip，完整的真值工作流</p><div className="local-indicator"><span className="status-dot"/>本地工作区 · 数据持久化</div></div>
    </aside>

    <main className="main-workspace">
      <section className="dataset-bar">
        <div className="dataset-select"><span className="dataset-icon"><Icon name="layers" size={20}/></span><div><label htmlFor="clip-select">当前数据 CLIP</label><select id="clip-select" aria-label="选择clip" value={clip?.id ?? ''} disabled={busy || jobBusy} onChange={e => {if (confirmDiscard()) void loadClip(e.target.value);}}>{!clip && <option value="">{loading ? '正在加载…' : '暂无数据，请导入 clip'}</option>}{clips.map(c => <option value={c.id} key={c.id}>{c.name}</option>)}</select></div><span className="demo-badge">{clip?.source === 'synthetic' || clip?.id === 'demo' ? '合成示例' : 'CLIP'}</span></div>
        <div className="dataset-metrics"><span><strong>{clip?.frames.length ?? '—'}</strong> 帧</span><span><strong>{clip?.cameras.length ?? '—'}</strong> 相机</span><span><strong>{frame?.points.length.toLocaleString() ?? '—'}</strong> 点 / 帧</span><span className="metric-fps">{clip?.fps ?? '—'} FPS</span></div>
        <div className="dataset-actions"><input ref={fileInput} type="file" accept=".json,.zip" hidden onChange={e => {void importFile(e.target.files?.[0]); e.target.value = '';}}/><button className="button ghost" aria-label="导入clip" disabled={busy || jobBusy} onClick={() => fileInput.current?.click()}><Icon name="upload" size={15}/>导入</button><button className="button subtle" aria-label="导出clip" disabled={!clip || busy || jobBusy} onClick={() => void exportClip()}><Icon name="download" size={15}/>导出标注</button></div>
      </section>

      <div className="workspace-grid">
        <section className="view-workspace">
          <div className="section-header"><div><span className="section-eyebrow">{taskInfo.short} ANNOTATION</span><h1>{taskInfo.name}<span>{taskInfo.subtitle}</span></h1></div><span className="coordinate-tag"><span/>LiDAR · 米 · 右手坐标系</span></div>
          <div className="scene-panel">
            <div className="scene-toolbar"><div className="segmented"><button className={view === 'perspective' ? 'active' : ''} onClick={() => setView('perspective')}>透视 3D</button><button className={view === 'top' ? 'active' : ''} onClick={() => setView('top')}>俯视 BEV</button></div><span className="toolbar-divider"/><div className="tool-buttons"><button className={`icon-button ${tool === 'select' ? 'active' : ''}`} title="选择目标" aria-label="选择工具" onClick={() => setTool('select')}><Icon name="select" size={17}/></button>{task === 'detection' && <button className={`icon-button ${tool === 'box' ? 'active' : ''}`} title="点击地面创建 3D 框" aria-label="新建3D框" onClick={() => {setTool('box'); setPlaying(false);}}><Icon name="plus" size={19}/></button>}{['lane', 'vector'].includes(task) && <button className={`icon-button ${tool === 'vector' ? 'active' : ''}`} title="点击地面绘制矢量顶点" aria-label="绘制矢量" onClick={() => {setTool('vector'); setView('top'); setPlaying(false);}}><Icon name="pen" size={17}/></button>}{task === 'semantic' && <button className={`icon-button ${tool === 'brush' ? 'active' : ''}`} title="点击点云标注笔刷半径内的点" aria-label="点云语义笔刷" onClick={() => {setTool('brush'); setPlaying(false);}}><Icon name="brush" size={17}/></button>}</div><div className="scene-toolbar-right"><button className={`toggle-button ${showBoxes ? 'on' : ''}`} onClick={() => setShowBoxes(!showBoxes)}><Icon name="cube" size={14}/>目标框</button><button className={`toggle-button ${showVectors ? 'on' : ''}`} onClick={() => setShowVectors(!showVectors)}><Icon name="lane" size={14}/>矢量线</button></div></div>
            <div className={`scene-canvas ${busy || jobBusy ? 'scene-disabled' : ''}`}>
              <PointCloudView frame={frame} annotations={annotations} selectedId={selectedId} onSelect={setSelectedId} onCreateBox={createBox} onAddVectorPoint={point => {if (!busy && !jobBusy) setDraftVector(p => [...p, point]);}} onPaint={indices => edit(a => ({...a, point_labels: {...a.point_labels, ...Object.fromEntries(indices.map(i => [String(i), classId]))}}))} tool={busy || jobBusy ? 'select' : tool} view={view} overlayPoints={task === 'fusion' ? overlayPoints : undefined} overlayPointLabels={fusionLabels} groundZ={groundZ} showFree={showFree} voxels={task === 'occupancy' && occupancy?.frameId === frame?.id ? occupancy?.voxels : undefined} voxelSize={occupancy?.size ?? voxelSize} showBoxes={showBoxes} showVectors={showVectors} semantic={task === 'semantic' || task === 'occupancy' || task === 'fusion'} draftVector={draftVector} brushRadius={brushRadius}/>
              <div className="scene-top-info"><span className="live-pill"><i/>FRAME {String(frameIndex + 1).padStart(4, '0')}</span><span>{task === 'fusion' && fusionPoints ? `融合结果 · ${fusionPoints.length.toLocaleString()} 点` : `${frame?.points.length.toLocaleString() ?? 0} POINTS`}</span></div>
              <div className="axis-legend"><b className="axis-x">X</b><b className="axis-y">Y</b><b className="axis-z">Z</b><span>ego / lidar</span></div>
              {!clip && !loading && <div className="empty-scene"><Icon name="layers" size={42}/><h2>从一个 clip 开始</h2><p>导入 JSON 清单或包含图像、点云的 ZIP 数据包</p><button className="button primary" onClick={() => fileInput.current?.click()}>导入数据</button></div>}
              {loading && <div className="loading-overlay"><span className="spinner"/>正在加载多模态数据…</div>}
              {draftVector.length > 0 && <div className="drawing-bar"><span>{draftVector.length} 个矢量顶点</span><button onClick={finishVector}>完成绘制</button><button onClick={() => setDraftVector(p => p.slice(0, -1))}>撤回一点</button><button onClick={() => setDraftVector([])}>取消</button></div>}
            </div>
            <div className="scene-footer"><span><span className="keycap">左键</span>{tool === 'box' ? '地面创建目标框' : tool === 'vector' ? '添加顶点' : tool === 'brush' ? '点云语义涂刷' : '旋转 / 选择'}<span className="keycap">右键</span>平移<span className="keycap">滚轮</span>缩放</span><span>外参已用于点云投影 <Icon name="check" size={12}/></span></div>
          </div>

          <div className="camera-section-header"><h2><Icon name="image" size={15}/>环视相机 <span>{clip?.cameras.length ?? 0} CHANNELS</span></h2><button className={`toggle-button ${projection ? 'on' : ''}`} onClick={() => setProjection(!projection)} aria-label="切换点云投影"><Icon name="points" size={14}/><span>点云投影</span><i className="toggle-switch"/></button></div>
          <div className="camera-grid">{cameraCards()}{!clip && <div className="camera-placeholder">导入 clip 后显示同步环视图像</div>}</div>
          {task === 'image' && <div className="image-draw-footer"><span>选择相机后点击图像添加多边形顶点，使用原始图像坐标。{draftPolygon.length ? ` 已添加 ${draftPolygon.length} 点` : ''}</span><button className="button subtle" onClick={finishPolygon} disabled={draftPolygon.length < 3}>完成多边形</button><button className="button ghost" onClick={() => setDraftPolygon([])}>取消</button><button className="button subtle" aria-label="导出语义掩码" disabled={!frame || busy || jobBusy || draftPolygon.length > 0} onClick={() => void exportMask()}><Icon name="download" size={13}/>导出掩码</button></div>}
        </section>

        <aside className="inspector">
          <div className="inspector-heading"><h2>标注属性</h2><span className="count-badge">{annotations.boxes.length + annotations.vectors.length + annotations.image_polygons.length}</span><button className="icon-button" onClick={() => setModal('help')} aria-label="标注帮助"><Icon name="settings" size={15}/></button></div>
          <div className="inspector-body">
            <section className="inspector-section"><div className="mini-heading"><span>类别标签</span><small>CLASS LABEL</small></div><div className="class-list">{classes.map(c => <button key={c.id} className={`class-option ${classId === c.id ? 'selected' : ''}`} onClick={() => {setClassId(c.id); if (selectedBox && task === 'detection') updateBox({label: c.id});}}><i style={{background: c.color}}/><span>{c.name}</span><small>{c.id}</small>{classId === c.id && <Icon name="check" size={13}/>}</button>)}</div></section>

            {selectedBox ? <section className="inspector-section"><div className="mini-heading"><span>目标属性</span><span className="text-cyan">3D BOX</span></div><label className="field-label">轨迹 ID<input aria-label="轨迹ID" value={selectedBox.track_id} disabled={busy || jobBusy} onChange={e => updateBox({track_id: e.target.value})}/></label><div className="coordinate-fields">{(['center', 'size'] as const).map(key => <div key={key}><label>{key === 'center' ? '中心位置 · m' : '尺寸 · m'}</label><div className="numeric-triplet">{(key === 'center' ? ['X', 'Y', 'Z'] : ['长', '宽', '高']).map((axis, i) => <label key={axis}><span>{axis}</span><input type="number" step="0.1" min={key === 'size' ? '0.01' : undefined} aria-label={`${key === 'center' ? '位置' : '尺寸'}${axis}`} value={selectedBox[key][i]} disabled={busy || jobBusy} onChange={e => {if (!Number.isFinite(e.target.valueAsNumber) || (key === 'size' && e.target.valueAsNumber <= 0)) return; const values = [...selectedBox[key]] as Vec3; values[i] = e.target.valueAsNumber; updateBox({[key]: values});}}/></label>)}</div></div>)}</div><label className="field-label">航向角 · 度<input type="number" step="1" aria-label="航向角" value={Number((selectedBox.yaw * 180 / Math.PI).toFixed(2))} disabled={busy || jobBusy} onChange={e => {if (Number.isFinite(e.target.valueAsNumber)) updateBox({yaw: e.target.valueAsNumber * Math.PI / 180});}}/></label><button className="button danger full-width" disabled={busy || jobBusy} onClick={deleteSelected}><Icon name="trash" size={14}/>删除选中目标</button></section> : selectedVector || selectedPolygon ? <section className="inspector-section"><div className="mini-heading"><span>选中要素</span></div><label className="field-label">要素类别<input aria-label="要素类别" value={selectedVector?.label ?? selectedPolygon?.label ?? ''} disabled={busy || jobBusy} onChange={e => edit(a => ({...a, vectors: a.vectors.map(v => v.id === selectedId ? {...v, label: e.target.value} : v), image_polygons: a.image_polygons.map(p => p.id === selectedId ? {...p, label: e.target.value} : p)}))}/></label>{selectedVector && <label className="closed-field"><input type="checkbox" checked={selectedVector.closed} disabled={busy || jobBusy || selectedVector.points.length < 3} onChange={e => edit(a => ({...a, vectors: a.vectors.map(v => v.id === selectedId ? {...v, closed: e.target.checked} : v)}))}/>闭合面要素</label>}<div className="vertex-list">{(selectedVector?.points ?? selectedPolygon?.points ?? []).map((point, pointIndex) => <div className="vertex-row" key={pointIndex}><span>{String(pointIndex + 1).padStart(2, '0')}</span>{point.map((value, axis) => <label key={axis}><small>{['X','Y','Z'][axis]}</small><input type="number" step="0.1" aria-label={`顶点${pointIndex + 1}${['X','Y','Z'][axis]}`} value={value} disabled={busy || jobBusy} onChange={e => {if (!Number.isFinite(e.target.valueAsNumber)) return; const n = e.target.valueAsNumber; edit(a => ({...a, vectors: a.vectors.map(v => v.id === selectedId ? {...v, points: v.points.map((p,i) => i === pointIndex ? p.map((x,j) => j === axis ? n : x) as Vec3 : p)} : v), image_polygons: a.image_polygons.map(poly => poly.id === selectedId ? {...poly, points: poly.points.map((p,i) => i === pointIndex ? p.map((x,j) => j === axis ? n : x) : p)} : poly)}));}}/></label>)}</div>)}</div><button className="button danger full-width" disabled={busy || jobBusy} onClick={deleteSelected}><Icon name="trash" size={14}/>删除选中要素</button></section> : <section className="inspector-section selection-empty"><Icon name={taskInfo.icon} size={29}/><p>{task === 'detection' ? '选择一个目标以编辑属性' : task === 'image' ? '点击图像绘制语义区域' : ['lane', 'vector'].includes(task) ? '启用绘制工具添加顶点' : task === 'semantic' ? '启用笔刷标注点云语义' : '在下方运行几何处理任务'}</p>{task === 'detection' && <button className="button subtle full-width" disabled={!frame || busy} onClick={() => setTool('box')}><Icon name="plus" size={15}/>新建 3D 目标框</button>}</section>}

            <section className="inspector-section object-section"><div className="mini-heading"><span>当前帧标注</span><small>{annotations.boxes.length} 目标 / {annotations.vectors.length} 矢量</small></div><div className="object-list">{annotations.boxes.map((box, i) => <button key={box.id} className={`object-row ${selectedId === box.id ? 'selected' : ''}`} onClick={() => {setSelectedId(box.id); setClassId(box.label);}}><Icon name="cube" size={14}/><span>{box.label}<small>{box.track_id}</small></span><em>{String(i + 1).padStart(2, '0')}</em></button>)}{annotations.vectors.map(v => <button key={v.id} className={`object-row ${selectedId === v.id ? 'selected' : ''}`} onClick={() => setSelectedId(v.id)}><Icon name="lane" size={14}/><span>{v.label}<small>{v.points.length} 个顶点</small></span></button>)}{annotations.image_polygons.map(p => <button key={p.id} className={`object-row ${selectedId === p.id ? 'selected' : ''}`} onClick={() => {setSelectedId(p.id); setActiveCamera(p.camera_id);}}><Icon name="image" size={14}/><span>{p.label}<small>{p.camera_id} · {p.points.length} 点</small></span></button>)}{!annotations.boxes.length && !annotations.vectors.length && !annotations.image_polygons.length && <p className="empty-caption">此帧暂无目标或矢量标注</p>}</div>{Object.keys(annotations.point_labels).length > 0 && <p className="point-label-count"><Icon name="points" size={13}/>{Object.keys(annotations.point_labels).length.toLocaleString()} 个点已标注语义</p>}</section>

            <section className="inspector-section automation-section"><div className="mini-heading"><span>几何处理</span><span className="cpu-tag">CPU</span></div>{['detection', 'lane', 'vector'].includes(task) && <label className="field-label">地面参考高度 · 米<input type="number" aria-label="地面参考高度" step="0.1" value={groundZ} onChange={e => {if (Number.isFinite(e.target.valueAsNumber)) setGroundZ(e.target.valueAsNumber);}}/></label>}{task === 'semantic' && <label className="field-label">笔刷半径 · 米 <span>{brushRadius.toFixed(1)}</span><input type="range" min="0.3" max="6" step="0.1" value={brushRadius} onChange={e => setBrushRadius(Number(e.target.value))}/></label>}{['fusion', 'occupancy'].includes(task) && <label className="field-label">体素分辨率 · 米<select aria-label="体素分辨率" value={voxelSize} onChange={e => setVoxelSize(Number(e.target.value))}><option value="0.2">0.2 m · 精细</option><option value="0.5">0.5 m · 标准</option><option value="1">1.0 m · 快速</option></select></label>}{['detection', 'fusion', 'occupancy', 'semantic'].includes(task) && <button className="button subtle full-width" disabled={!clip || busy || jobBusy || (task === 'detection' && frameIndex >= clip.frames.length - 1)} onClick={() => void runJob(task === 'fusion' ? 'map' : task === 'occupancy' ? 'occupancy' : task === 'detection' ? 'propagate' : 'ground-segmentation')}><Icon name={task === 'occupancy' ? 'grid' : task === 'fusion' ? 'layers' : task === 'detection' ? 'next' : 'points'} size={15}/>{jobBusy ? '正在计算…' : task === 'fusion' ? '融合 clip 静态地图' : task === 'occupancy' ? '转换当前帧 OCC' : task === 'detection' ? '位姿传播至下一帧' : '运行地面分割'}</button>}{task === 'occupancy' && <div className="occupancy-legend"><label className="closed-field"><input type="checkbox" aria-label="显示自由空间" checked={showFree} onChange={e => setShowFree(e.target.checked)}/>显示已观测自由空间</label>{occupancy?.frameId === frame?.id && <p><span>占据 {occupancy?.voxels.filter(v => v.state === 'occupied').length.toLocaleString()}</span><span>自由 {occupancy?.voxels.filter(v => v.state === 'free').length.toLocaleString()}</span></p>}<small>未观测体素保持未知，不在稀疏结果中。</small></div>}<p className="micro-copy">{task === 'fusion' ? '应用每帧自车位姿；剔除已标注动态框内点。' : task === 'occupancy' ? '射线可见区域标记自由空间，未观测区域保持未知。' : task === 'detection' ? '按世界坐标传播静态框；动态目标需人工修正。' : task === 'semantic' ? '基于高度的 CPU 几何辅助结果，需人工审核。' : task === 'image' ? '分割区域按原始像素坐标保存，可导出类别 ID 掩码。' : '参考高度用于无点命中时落地；顶点高度可在属性中逐点调整。'}</p>{job && <div className={`job-status ${job.status}`}><span className="status-dot"/><strong>{job.status === 'completed' ? '处理完成' : job.status === 'failed' ? '处理失败' : '后台计算中'}</strong><small>{job.error ?? job.kind}</small>{job.status === 'completed' && job.result && ['map', 'occupancy'].includes(job.kind) && <button title="导出计算结果" aria-label="导出计算结果" onClick={() => download(`${clip?.id}-${job.kind}.json`, JSON.stringify(job.result))}><Icon name="download" size={14}/></button>}</div>}</section>
          </div>
          <div className="inspector-bottom"><Icon name="check" size={13}/><span>标注版本 {annotations.revision} · {dirty ? '待保存' : '已保存'}</span>{dirty && <button className="draft-download" aria-label="下载当前帧草稿" title="下载当前帧草稿备份，不写入服务端" onClick={() => download(`${clip?.id}-${frame?.id}-draft.json`, JSON.stringify({clip_id: clip?.id, frame_id: frame?.id, annotations}, null, 2))}><Icon name="download" size={12}/>草稿</button>}</div>
        </aside>
      </div>

      <section className="timeline" aria-label="clip时间轴"><div className="timeline-top"><div className="playback-controls"><button className="icon-button" aria-label="上一帧" disabled={!clip || frameIndex === 0 || busy || jobBusy} onClick={() => void moveFrame(frameIndex - 1)}><Icon name="prev" size={16}/></button><button className={`play-button ${playing ? 'playing' : ''}`} aria-label={playing ? '暂停' : '播放'} disabled={!clip || busy || jobBusy} onClick={() => {if (dirty || draftVector.length || draftPolygon.length) {flash('请先保存标注并完成绘制，再播放 clip', true); return;} if (frameIndex === (clip?.frames.length ?? 0) - 1) void moveFrame(0); setPlaying(!playing);}}><Icon name={playing ? 'pause' : 'play'} size={15}/></button><button className="icon-button" aria-label="下一帧" disabled={!clip || frameIndex >= clip.frames.length - 1 || busy || jobBusy} onClick={() => void moveFrame(frameIndex + 1)}><Icon name="next" size={16}/></button><span className="frame-position"><strong>{String(frameIndex + 1).padStart(4, '0')}</strong><span>/ {String(clip?.frames.length ?? 0).padStart(4, '0')}</span></span></div><span className="timeline-title">CLIP TIMELINE <span>同步帧序列</span></span><span className="timestamp">t = {frame?.timestamp.toFixed(3) ?? '0.000'} s <span className="timebase">{clip?.fps ?? 0} Hz</span></span></div><div className="timeline-track"><div className="timeline-ruler">{[0, 1, 2, 3, 4].map(i => <span key={i}>{(clip?.frames.length ? (clip.frames[clip.frames.length - 1].timestamp - clip.frames[0].timestamp) * i / 4 : 0).toFixed(1)}s</span>)}</div><div className="frame-cells">{clip?.frames.map((f, i) => <button key={f.id} className={`${frameIndex === i ? 'current' : ''} ${i < frameIndex ? 'past' : ''}`} aria-label={`跳转到第${i + 1}帧`} disabled={busy || jobBusy} title={`${f.id} · ${f.timestamp.toFixed(3)}s`} onClick={() => void moveFrame(i)}><i/>{frameIndex === i && <span>{String(i + 1).padStart(2, '0')}</span>}</button>)}</div></div></section>
    </main>

    {notice && <div className={`toast ${notice.error ? 'error' : ''}`} role={notice.error ? 'alert' : 'status'}><Icon name={notice.error ? 'help' : 'check'} size={18}/><span>{notice.message}</span><button className="icon-button" onClick={() => setNotice(null)} aria-label="关闭提示"><Icon name="close" size={15}/></button></div>}
    {expandedCamera && <div className="modal-backdrop" onClick={() => setExpandedCamera(null)}><div className="expanded-camera" onClick={e => e.stopPropagation()}>{cameraCards(true)}{task === 'image' && <div className="image-draw-footer"><span>{draftPolygon.length} 个顶点 · 点击图像绘制</span><button className="button primary" onClick={finishPolygon} disabled={draftPolygon.length < 3}>完成多边形</button><button className="button ghost" onClick={() => setDraftPolygon([])}>取消</button></div>}</div></div>}
    {modal && <div className="modal-backdrop" onClick={() => setModal(null)}><div className="modal" role="dialog" aria-modal="true" aria-label={modal === 'models' ? '模型与预标注' : '使用说明'} onClick={e => e.stopPropagation()}><div className="modal-heading"><div><span className="section-eyebrow">{modal === 'models' ? 'MODEL ECOSYSTEM' : 'GETTING STARTED'}</span><h2>{modal === 'models' ? '模型与预标注' : '多模态标注工作流'}</h2></div><button className="icon-button" onClick={() => setModal(null)} aria-label="关闭对话框"><Icon name="close"/></button></div>{modal === 'models' ? <><p className="modal-intro">开源模型生成候选标注，经人工校验后才能作为真值。平台内置 CPU 几何处理；模型权重未捆绑，可配置外部推理服务。服务已配置不代表模型健康已验证。</p><div className="model-list">{modelList?.map(model => <div className="model-card" key={model.id}><div><Icon name="model" size={19}/><h3>{model.name}</h3><span className={model.available ? 'available-tag' : 'unavailable-tag'}>{model.available ? model.id === 'external-worker' ? '已配置' : '可用' : '未部署'}</span></div><p>{model.description}</p><div className="model-tasks">{model.tasks.map(t => <span key={t}>{t}</span>)}</div><a href={model.url} target="_blank" rel="noreferrer">查看开源项目 ↗</a></div>)}{!modelList && <p className="muted">正在读取模型目录…</p>}</div><section className="worker-panel"><div className="mini-heading"><span>外部模型候选生成</span><small>人工审核后应用</small></div><div className="worker-fields"><label className="field-label">模型 ID<input aria-label="推理模型ID" value={workerModel} onChange={e => setWorkerModel(e.target.value)}/></label><label className="field-label">文本提示 · 可选<input aria-label="推理提示词" value={workerPrompt} placeholder="例如：道路、车辆、行人" onChange={e => setWorkerPrompt(e.target.value)}/></label></div><button className="button subtle" aria-label="生成候选标注" disabled={!clip || busy || jobBusy || !workerModel.trim() || !modelList?.some(m => m.id === 'external-worker' && m.available)} onClick={() => void runJob('prelabel')}><Icon name="model" size={14}/>{jobBusy ? '正在生成…' : '生成候选标注'}</button>{!modelList?.some(m => m.id === 'external-worker' && m.available) && <p className="micro-copy">需要先在服务端配置 OMNILABEL_INFERENCE_URL；当前未连接推理服务。</p>}{proposal && <div className="proposal-review"><strong>{proposal.model} · 候选标注待审核</strong><p>{proposal.annotations.boxes.length} 个目标 / {proposal.annotations.vectors.length} 个矢量 / {proposal.annotations.image_polygons.length} 个图像区域。来源帧：{proposal.frameId}</p><button className="button primary" disabled={busy || jobBusy} onClick={applyProposal}>应用候选标注</button><button className="button ghost" onClick={() => setProposal(null)}>丢弃候选</button></div>}</section><p className="micro-copy">模型权重、许可证、GPU 与推理服务需要单独配置。预测数据可通过 clip 导入接口进入人工校验流程。</p></> : <div className="help-content"><p>数据以 clip 为单位组织，每帧包含同步相机图像、LiDAR 点云、自车位姿和标定矩阵。示例 clip 为合成场景，用于验证编辑与几何流程。</p><ol><li><strong>导入数据：</strong>选择 JSON 清单，或含 manifest.json、图像及点云文件的 ZIP。支持内联点、float32 XYZI .bin、.npy 和 ASCII .pcd。</li><li><strong>3D 标注：</strong>点击「＋」后点击地面放置目标框，在右侧编辑位置、尺寸、航向和轨迹 ID。跨帧传播只补偿自车运动。</li><li><strong>地图与分割：</strong>俯视图绘制折线；点击图像添加多边形顶点；点云笔刷按实际点索引保存语义标签。</li><li><strong>保存与计算：</strong>保存当前帧后进行融合或 OCC 转换。修改未保存时切帧会提示；导出包含所有已保存标注。</li></ol><div className="help-coordinate"><strong>坐标约定</strong><p>LiDAR / ego：X 向前，Y 向左，Z 向上，单位米；相机：X 向右，Y 向下，Z 向前。外参 camera_from_ego 将 ego 坐标变换到相机，投影支持 Brown–Conrady 畸变。</p></div><p className="micro-copy">详细数据格式、算法边界与开源参考见仓库 docs/。本原型适合本地单用户工作区；团队权限与生产部署需独立配置。</p></div>}</div></div>}
  </div>;
}
