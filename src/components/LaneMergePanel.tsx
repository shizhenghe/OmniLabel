import {useEffect, useRef, useState} from 'react';
import {request} from '../api';
import type {Vector} from '../types';
import './LaneMergePanel.css';

export interface LaneMergeResult {
  vectors: Vector[];
  report: {
    input_count: number;
    output_count: number;
    merged_count: number;
    groups: {kept_id: string; source_ids: string[]}[];
  };
}

interface Props {
  clipId: string;
  scopeKey: string;
  vectors: Vector[];
  sceneRevision?: number;
  disabled: boolean;
  onPreview: (result: LaneMergeResult | null) => void;
  onApply: (result: LaneMergeResult) => void;
  onClose: () => void;
  onError: (message: string) => void;
}

/** Preview is disposable: a source edit, changed parameter, or clip/frame switch
 * aborts computation and clears the overlay without writing any annotations. */
export function LaneMergePanel(props: Props) {
  const [maxDistance, setMaxDistance] = useState(.25);
  const [maxAngle, setMaxAngle] = useState(15);
  const [minOverlap, setMinOverlap] = useState(1);
  const [pending, setPending] = useState(false);
  const [result, setResult] = useState<LaneMergeResult | null>(null);
  const [error, setError] = useState('');
  const controller = useRef<AbortController | null>(null);
  const latest = useRef({props, maxDistance, maxAngle, minOverlap});
  latest.current = {props, maxDistance, maxAngle, minOverlap};

  useEffect(() => {
    controller.current?.abort();
    controller.current = null;
    setPending(false); setResult(null); setError('');
    latest.current.props.onPreview(null);
    return () => {
      controller.current?.abort();
      controller.current = null;
      latest.current.props.onPreview(null);
    };
  }, [props.clipId, props.scopeKey, props.vectors, props.sceneRevision, maxDistance, maxAngle, minOverlap]);

  const preview = async () => {
    if (latest.current.props.disabled || pending) return;
    const snapshot = latest.current;
    const requestController = new AbortController();
    controller.current?.abort();
    controller.current = requestController;
    setPending(true); setResult(null); setError('');
    snapshot.props.onPreview(null);
    const current = () => {
      const value = latest.current;
      return !requestController.signal.aborted && controller.current === requestController
        && value.props.clipId === snapshot.props.clipId && value.props.scopeKey === snapshot.props.scopeKey
        && value.props.vectors === snapshot.props.vectors && value.props.sceneRevision === snapshot.props.sceneRevision
        && value.maxDistance === snapshot.maxDistance && value.maxAngle === snapshot.maxAngle
        && value.minOverlap === snapshot.minOverlap;
    };
    try {
      const merged = await request<LaneMergeResult>(`/clips/${encodeURIComponent(snapshot.props.clipId)}/lanes/merge-preview`, {
        method: 'POST', headers: {'Content-Type': 'application/json'}, signal: requestController.signal,
        body: JSON.stringify({vectors: snapshot.props.vectors, max_distance: snapshot.maxDistance,
          max_angle_degrees: snapshot.maxAngle, min_overlap: snapshot.minOverlap,
          ...(snapshot.props.sceneRevision === undefined ? {} : {scene_revision: snapshot.props.sceneRevision})}),
      });
      if (!current()) return;
      setResult(merged);
      latest.current.props.onPreview(merged);
    } catch (failure) {
      if (!current()) return;
      const message = `车道线融合预览失败：${failure instanceof Error ? failure.message : String(failure)}`;
      setError(message); latest.current.props.onError(message);
    } finally {
      if (controller.current === requestController) {controller.current = null; setPending(false);}
    }
  };

  const close = () => {
    controller.current?.abort(); controller.current = null;
    setResult(null); setPending(false);
    latest.current.props.onPreview(null);
    latest.current.props.onClose();
  };

  return <section className="lane-merge-panel" aria-label="车道线融合">
    <div className="lane-merge-heading"><h3>重复车道线融合</h3><button type="button" className="icon-button"
      aria-label="关闭车道线融合" onClick={close}>×</button></div>
    <div className="lane-merge-parameters">
      <label className="field-label">最大间距 · 米<input type="number" aria-label="融合最大间距" min="0.05" max="1" step="0.05"
        disabled={props.disabled || pending} value={maxDistance} onChange={event => {
          if (Number.isFinite(event.target.valueAsNumber)) setMaxDistance(Math.min(1, Math.max(.05, event.target.valueAsNumber)));
        }}/></label>
      <label className="field-label">最大方向差 · 度<input type="number" aria-label="融合最大方向差" min="1" max="30" step="1"
        disabled={props.disabled || pending} value={maxAngle} onChange={event => {
          if (Number.isFinite(event.target.valueAsNumber)) setMaxAngle(Math.min(30, Math.max(1, event.target.valueAsNumber)));
        }}/></label>
      <label className="field-label">最小重合长度 · 米<input type="number" aria-label="融合最小重合长度" min="0.2" max="10" step="0.1"
        disabled={props.disabled || pending} value={minOverlap} onChange={event => {
          if (Number.isFinite(event.target.valueAsNumber)) setMinOverlap(Math.min(10, Math.max(.2, event.target.valueAsNumber)));
        }}/></label>
    </div>
    <div className="lane-merge-actions"><button className="button subtle" type="button" disabled={props.disabled || pending || props.vectors.length < 2}
      onClick={() => void preview()}>{pending ? '正在计算…' : '预览融合'}</button>
      <button className="button primary" type="button" disabled={props.disabled || pending || !result || result.report.merged_count === 0}
        onClick={() => {if (result && !latest.current.props.disabled) latest.current.props.onApply(result);}}>应用融合</button>
      <button className="button ghost" type="button" onClick={close}>取消</button></div>
    {result && <div className="lane-merge-result" role="status" data-testid="lane-merge-result">
      <strong>{result.report.input_count} → {result.report.output_count} 条矢量</strong>
      <p>{result.report.merged_count > 0
        ? `已消除 ${result.report.merged_count} 条重复线；点云与图像正在显示融合预览。`
        : '未找到满足当前阈值的重复车道线。'}</p>
      {result.report.groups.length > 0 && <details><summary>{result.report.groups.length} 组融合记录</summary>
        <ul>{result.report.groups.map(group => <li key={group.kept_id}><code>{group.kept_id}</code><span>{group.source_ids.join(' / ')}</span></li>)}</ul>
      </details>}
    </div>}
    {error && <p className="lane-merge-error" role="alert">{error}</p>}
  </section>;
}
