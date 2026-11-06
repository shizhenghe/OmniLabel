import {useEffect, useRef, useState} from 'react';
import {request} from '../api';

type Check = {key: string; title: string; status: 'ready' | 'warning' | 'missing'; detail: string; action: string};
interface InputReport {
  clip_id: string; frame_id: string; camera_id: string;
  ready_to_run: boolean; checks: Check[]; note: string;
}
interface Props {clipId: string; frameId: string; cameraId: string; disabled: boolean; refreshKey: number;}
const labels = {ready: '已具备', warning: '需核对', missing: '待处理'};
export function CalibrationInputChecklist({clipId, frameId, cameraId, disabled, refreshKey}: Props) {
  const [report, setReport] = useState<InputReport | null>(null);
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState('');
  const controller = useRef<AbortController | null>(null);
  useEffect(() => {controller.current?.abort(); setChecking(false); setReport(null); setError('');}, [refreshKey]);
  useEffect(() => () => controller.current?.abort(), []);
  const inspect = async () => {
    if (checking || disabled) return;
    const abort = new AbortController(); controller.current = abort;
    setChecking(true); setError(''); setReport(null);
    try {
      const value = await request<InputReport>(`/clips/${encodeURIComponent(clipId)}/automatic-calibration/inputs/${encodeURIComponent(frameId)}/${encodeURIComponent(cameraId)}`, {signal: abort.signal});
      if (!abort.signal.aborted) setReport(value);
    } catch (cause) {if (!abort.signal.aborted) setError(`检查失败：${cause instanceof Error ? cause.message : String(cause)}`);}
    finally {if (!abort.signal.aborted) setChecking(false);}
  };
  return <section className="calibration-input-checklist" aria-label="PJLab标定输入检查">
    <h3>当前帧能否开始标定？</h3>
    <p>{frameId} / {cameraId} · 检查实际原图、标定扫描、强度、内参、畸变、初值和掩码。不会修改数据或启动求解。</p>
    <button className="button subtle" aria-label="检查当前帧标定输入" disabled={disabled || checking} onClick={() => void inspect()}>{checking ? '正在读取并检查完整扫描…' : '检查当前帧标定输入'}</button>
    {checking && <p role="status">检查完整点云，无点数截断；大扫描可能需要一些时间。</p>}
    {report && <div data-testid="calibration-input-report">
      <p role="status">{report.ready_to_run ? '输入可开始试运行，仍需核对下列提示并复核求解效果。' : '请先处理以下「待处理」项目，再启动 PJLab。'}</p>
      <ul className="calibration-input-checks">{report.checks.map(item => <li key={item.key} data-testid={`calibration-input-${item.key}`} data-status={item.status}>
        <strong><span>{labels[item.status]}</span> · {item.title}</strong><p>{item.detail}</p>{item.action && <p className="calibration-input-action">{item.action}</p>}
      </li>)}</ul><p>{report.note}</p>
    </div>}
    {error && <p role="alert">{error}</p>}
    <details><summary>第一次使用：按这个顺序操作</summary><ol>
      <li>选真实 clip 的当前帧与前视相机，检查输入；缺少同步扫描或实测强度时先补齐数据。</li>
      <li>在下方调用 SAM2，勾选道路标线/杆体区域，保存线特征掩码，再检查一次输入。</li>
      <li>处理范围先选「当前帧 + 当前相机」，选择 PJLab，启动自动标定。</li>
      <li>打开「查看本次结果 / 手动校准」，比较近远处投影；通过项直接应用，未通过项复核后可人工确认应用。</li>
      <li>确认这一帧有效后，再逐帧准备掩码并扩大处理范围。当前帧掩码不能代替其他图像的掩码。</li>
    </ol></details>
  </section>;
}
