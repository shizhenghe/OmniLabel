import {useEffect, useRef, useState} from 'react';
import {request} from '../api';
import type {Clip, ImagePolygon} from '../types';
import './SAM2LineMaskPanel.css';

type PointPrompt = {point: [number, number]; label: number};
type InferenceJob = {id: string; status: string; error?: string; result?: {
  proposal: {image_polygons: ImagePolygon[]}; sam2_candidate_ids?: string[]; warnings?: string[];
}};
interface Props {
  clip: Clip;
  frameId: string;
  cameraId: string;
  disabled: boolean;
  onBusy: (busy: boolean) => void;
}
const errorText = (error: unknown) => error instanceof Error ? error.message : String(error);

/** Converts reviewed SAM2 regions to a calibration mask without editing annotations. */
export function SAM2LineMaskPanel({clip, frameId, cameraId, disabled, onBusy}: Props) {
  const [available, setAvailable] = useState(false);
  const [readiness, setReadiness] = useState('正在检查 SAM2 服务…');
  const [mode, setMode] = useState<'auto' | 'points'>('auto');
  const [negative, setNegative] = useState(false);
  const [points, setPoints] = useState<PointPrompt[]>([]);
  const [candidates, setCandidates] = useState<ImagePolygon[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [jobId, setJobId] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [status, setStatus] = useState('');
  const [warnings, setWarnings] = useState<string[]>([]);
  const controller = useRef<AbortController | null>(null);
  const camera = clip.cameras.find(item => item.id === cameraId)!;
  const image = clip.frames.find(item => item.id === frameId)?.images[cameraId];
  const locked = disabled || busy;
  const base = `/clips/${encodeURIComponent(clip.id)}`;

  useEffect(() => {
    const abort = new AbortController();
    request<{models: {id: string; available: boolean; reason?: string}[]}>('/models', {signal: abort.signal})
      .then(value => {
        if (abort.signal.aborted) return;
        const model = value.models.find(item => item.id === 'sam2');
        setAvailable(Boolean(model?.available));
        setReadiness(model?.available ? 'SAM2 推理就绪' : model?.reason || 'SAM2 未就绪，请在模型预标注中配置并检查服务。');
      }).catch(cause => {if (!abort.signal.aborted) setReadiness(`SAM2 状态读取失败：${errorText(cause)}`);});
    return () => abort.abort();
  }, []);
  useEffect(() => () => {controller.current?.abort(); onBusy(false);}, [onBusy]);

  const toggle = (id: string) => {
    if (locked) return;
    setStatus('');
    setSelected(old => {const next = new Set(old); if (next.has(id)) next.delete(id); else next.add(id); return next;});
  };
  const generate = async () => {
    if (locked || !available || !image || (mode === 'points' && !points.some(p => p.label === 1))) return;
    const abort = new AbortController(); controller.current = abort;
    setBusy(true); onBusy(true); setError(''); setStatus('正在调用当前帧 SAM2…');
    setCandidates([]); setSelected(new Set()); setJobId(''); setWarnings([]);
    try {
      const prompts = mode === 'points' ? points : [];
      let job = await request<InferenceJob>(`${base}/jobs`, {method: 'POST', signal: abort.signal,
        headers: {'Content-Type': 'application/json'}, body: JSON.stringify({kind: 'prelabel', frame_id: frameId,
          model_id: 'sam2', scene_mode: false, prompt: 'lane', options: {camera_id: cameraId, label: 'lane',
            points: prompts.map(p => p.point), point_labels: prompts.map(p => p.label)}})});
      for (let attempts = 0; ['queued', 'running'].includes(job.status); attempts++) {
        if (attempts >= 1200) throw new Error(`推理等待超时，后台任务 ${job.id} 仍可查询。`);
        await new Promise(resolve => setTimeout(resolve, 500));
        if (abort.signal.aborted) return;
        job = await request<InferenceJob>(`/jobs/${encodeURIComponent(job.id)}`, {signal: abort.signal});
      }
      if (abort.signal.aborted) return;
      if (job.status !== 'completed' || !job.result) throw new Error(job.error || 'SAM2 未返回有效候选。');
      const ids = new Set(job.result.sam2_candidate_ids ?? []);
      const regions = job.result.proposal.image_polygons.filter(p => p.camera_id === cameraId && ids.has(p.id));
      setCandidates(regions); setJobId(job.id); setWarnings(job.result.warnings ?? []);
      setStatus(regions.length ? `已生成 ${regions.length} 个候选。请勾选标线或杆体区域，再保存线特征掩码。`
        : '未生成新的区域，请改用正负提示点，或检查 SAM2 服务是否支持返回分割多边形。');
    } catch (cause) {if (!abort.signal.aborted) {setError(`生成失败：${errorText(cause)}`); setStatus('');}}
    finally {if (controller.current === abort && !abort.signal.aborted) {setBusy(false); onBusy(false);}}
  };
  const save = async () => {
    if (locked || !jobId || !selected.size) return;
    const abort = new AbortController(); controller.current = abort;
    setBusy(true); onBusy(true); setError(''); setStatus('');
    try {
      await request(`${base}/automatic-calibration/masks/${encodeURIComponent(frameId)}/${encodeURIComponent(cameraId)}/from-sam2`, {
        method: 'POST', signal: abort.signal, headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({job_id: jobId, polygon_ids: [...selected]}),
      });
      if (!abort.signal.aborted) setStatus(`已保存 ${frameId} / ${cameraId} 的原图尺寸线特征掩码，可直接运行 PJLab 标定。其他帧需分别生成对应掩码。`);
    } catch (cause) {if (!abort.signal.aborted) setError(`掩码未保存：${errorText(cause)}`);}
    finally {if (controller.current === abort && !abort.signal.aborted) {setBusy(false); onBusy(false);}}
  };
  return <section className="sam2-line-mask" aria-label="SAM2线特征掩码">
    <h3>SAM2 生成线特征掩码</h3>
    <p>{readiness} · {frameId} / {cameraId}</p>
    <p>SAM2 不按文字识别车道线。请只选道路标线或杆体区域，不选整片路面、车辆、天空。候选不会写入语义标注。</p>
    <div className="sam2-mask-tools">
      <label>生成方式<select aria-label="线特征SAM2生成方式" value={mode} disabled={locked} onChange={e => {setMode(e.target.value as typeof mode); setStatus('');}}><option value="auto">整图自动候选</option><option value="points">正负提示点分割</option></select></label>
      {mode === 'points' && <><button className="button subtle" disabled={locked} aria-pressed={!negative} onClick={() => setNegative(false)}>正点</button><button className="button subtle" disabled={locked} aria-pressed={negative} onClick={() => setNegative(true)}>负点</button><button className="button ghost" disabled={locked} onClick={() => setPoints([])}>清除线特征提示</button></>}
      <button className="button primary" aria-label="SAM2生成当前帧线特征候选" disabled={locked || !available || !image || (mode === 'points' && !points.some(p => p.label === 1))} onClick={() => void generate()}>{busy ? '正在处理…' : 'SAM2 生成当前帧候选'}</button>
    </div>
    {mode === 'points' && <p>在下图标线上添加正点，在需要排除的区域添加负点。共 {points.length} 个提示点。</p>}
    {image && <svg className="sam2-mask-image" data-testid="sam2-mask-image" role="img" aria-label="线特征候选与提示图像" viewBox={`0 0 ${camera.width} ${camera.height}`} onClick={event => {
      if (locked || mode !== 'points' || points.length >= 256) return;
      const rect = event.currentTarget.getBoundingClientRect();
      const point: [number, number] = [
        Math.max(0, Math.min(camera.width - 1, (event.clientX - rect.left) / rect.width * camera.width)),
        Math.max(0, Math.min(camera.height - 1, (event.clientY - rect.top) / rect.height * camera.height)),
      ];
      setPoints(old => [...old, {point, label: negative ? 0 : 1}]); setStatus('');
    }}>
      <image href={image} width={camera.width} height={camera.height}/>
      {candidates.map((p, i) => <g key={p.id} onClick={event => {if (mode === 'auto') {event.stopPropagation(); toggle(p.id);}}}>
        <polygon points={p.points.map(point => point.join(',')).join(' ')} fill={selected.has(p.id) ? '#00e5bb77' : '#ffcc4433'} stroke={selected.has(p.id) ? '#00e5bb' : '#ffcc44'} strokeWidth="2" vectorEffect="non-scaling-stroke"/>
        <text x={p.points[0][0]} y={p.points[0][1]} fill="white" fontSize={Math.max(12, camera.width / 65)}>{i+1}</text>
      </g>)}
      {mode === 'points' && points.map((p, i) => <circle key={i} cx={p.point[0]} cy={p.point[1]} r={Math.max(3, camera.width / 180)} fill={p.label ? '#00e5bb' : '#ff5577'} stroke="white"/>)}
    </svg>}
    {candidates.length > 0 && <>
      <div className="sam2-mask-candidates">{candidates.map((p, i) => <label key={p.id}><input type="checkbox" aria-label={`线特征候选 ${i+1}`} checked={selected.has(p.id)} disabled={locked} onChange={() => toggle(p.id)}/>区域 {i+1} · {p.points.length} 个边界点</label>)}</div>
      <p>已选 {selected.size} 个区域 · 黑色为背景、白色为线特征 · {camera.width} × {camera.height}</p>
      <svg className="sam2-mask-preview" data-testid="sam2-line-mask-preview" role="img" aria-label="线特征黑白掩码预览" viewBox={`0 0 ${camera.width} ${camera.height}`}>
        <rect width={camera.width} height={camera.height} fill="black"/>
        {candidates.filter(p => selected.has(p.id)).map(p => <polygon key={p.id} points={p.points.map(point => point.join(',')).join(' ')} fill="white"/>)}
      </svg>
      <button className="button primary" aria-label="保存SAM2线特征掩码" disabled={locked || !selected.size} onClick={() => void save()}>保存选定区域为线特征掩码</button>
    </>}
    {warnings.map((warning, i) => <p className="sam2-mask-warning" key={i}>{warning}</p>)}
    {status && <p role="status">{status}</p>}{error && <p role="alert">{error}</p>}
  </section>;
}
