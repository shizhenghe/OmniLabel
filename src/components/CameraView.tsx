import { useEffect, useMemo, useRef, useState } from 'react';
import type { Annotations, Camera, Frame, ProjectionPoint } from '../types';
import { labelColor } from './PointCloudView';
import { projectAnnotationResults, projectionSignature, type Pixel, type PixelSegment } from '../geometry/cameraProjection';

export { projectLidarPoint } from '../geometry/cameraProjection';

export interface CameraViewProps {
  camera: Camera;
  frame: Frame | null;
  annotations: Annotations;
  projections?: ProjectionPoint[];
  selected?: boolean;
  tool?: 'select' | 'polygon';
  onAddPolygonPoint?: (point: [number, number]) => void;
  onSelect?: () => void | boolean;
  onFinishPolygon?: () => void;
  draftPolygon?: number[][];
  showProjection?: boolean;
  /** Annotation results are separate from optional raw point-cloud overlays. */
  showAnnotationProjection?: boolean;
  showBoxes?: boolean;
  showVectors?: boolean;
  modelPromptMode?: 'point' | 'box' | null;
  onModelPoint?: (point: [number, number], label?: 0 | 1) => void;
  onModelBox?: (box: [number, number, number, number]) => void;
  modelPoints?: number[][];
  modelPointLabels?: number[];
  modelBox?: number[];
  selectedId?: string | null;
  className?: string;
}

export function CameraView(props: CameraViewProps) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const host = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const fit = useRef({ x: 0, y: 0, width: 0, height: 0 });
  const [imageState, setImageState] = useState<'loading' | 'ready' | 'missing' | 'error'>('loading');
  const [size, setSize] = useState({ width: 0, height: 0 });
  const modelDrag = useRef<{ start: Pixel; pointerId: number } | null>(null);
  const [draftModelBox, setDraftModelBox] = useState<number[] | null>(null);
  const source = props.frame?.images[props.camera.id];
  const results = useMemo(() => props.frame ? projectAnnotationResults(props.annotations, props.camera, props.frame)
    : { boxes: [], vectors: [], imagePolygons: [] }, [props.annotations, props.camera, props.frame]);
  const visibleResults = useMemo(() => ({
    boxes: props.showAnnotationProjection !== false && props.showBoxes !== false ? results.boxes : [],
    vectors: props.showAnnotationProjection !== false && props.showVectors !== false ? results.vectors : [],
    imagePolygons: props.showAnnotationProjection !== false ? results.imagePolygons : [],
  }), [results, props.showAnnotationProjection, props.showBoxes, props.showVectors]);
  const signature = useMemo(() => projectionSignature(visibleResults), [visibleResults]);

  useEffect(() => { modelDrag.current = null; setDraftModelBox(null); }, [props.modelPromptMode, props.frame?.id, props.camera.id]);

  useEffect(() => {
    if (!host.current) return;
    const element = host.current;
    const observer = new ResizeObserver(() => {
      const rect = element.getBoundingClientRect();
      setSize({ width: rect.width, height: rect.height });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    imageRef.current = null;
    if (!source) { setImageState('missing'); return; }
    let cancelled = false;
    setImageState('loading');
    const image = new Image();
    image.onload = () => {
      if (cancelled) return;
      imageRef.current = image;
      setImageState('ready');
    };
    image.onerror = () => { if (!cancelled) setImageState('error'); };
    image.src = source;
    return () => { cancelled = true; image.onload = null; image.onerror = null; };
  }, [source]);

  useEffect(() => {
    if (!canvas.current || !size.width || !size.height) return;
    const element = canvas.current;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    element.width = Math.round(size.width * ratio);
    element.height = Math.round(size.height * ratio);
    const ctx = element.getContext('2d');
    if (!ctx) return;
    ctx.scale(ratio, ratio);
    ctx.fillStyle = '#0b1420';
    ctx.fillRect(0, 0, size.width, size.height);
    const { width, height } = props.camera;
    const scale = Math.min(size.width / width, size.height / height);
    const displayWidth = width * scale;
    const displayHeight = height * scale;
    const offsetX = (size.width - displayWidth) / 2;
    const offsetY = (size.height - displayHeight) / 2;
    fit.current = { x: offsetX, y: offsetY, width: displayWidth, height: displayHeight };
    ctx.save();
    ctx.translate(offsetX, offsetY);
    ctx.beginPath();
    ctx.rect(0, 0, displayWidth, displayHeight);
    ctx.clip();
    const xy = (point: number[]) => [point[0] * scale, point[1] * scale];
    if (imageState === 'ready' && imageRef.current) {
      ctx.drawImage(imageRef.current, 0, 0, displayWidth, displayHeight);
    } else {
      ctx.fillStyle = '#121d2a';
      ctx.fillRect(0, 0, displayWidth, displayHeight);
      ctx.fillStyle = '#6e8296';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.font = '12px system-ui, sans-serif';
      ctx.fillText(imageState === 'loading' ? '加载图像…' : imageState === 'error' ? '图像加载失败' : '此帧未提供图像', displayWidth / 2, displayHeight / 2);
    }
    if (props.showProjection === true) {
      for (const point of props.projections ?? []) {
        const [x, y] = xy([point.u, point.v]);
        // Current annotations are authoritative, including erased labels; a
        // projection response may have been computed before unsaved edits.
        const label = props.annotations.point_labels[String(point.index)];
        ctx.fillStyle = label ? labelColor(label) : `hsl(${Math.max(0, Math.min(210, point.depth * 3.4))}, 90%, 64%)`;
        ctx.fillRect(x - 0.85, y - 0.85, 1.7, 1.7);
      }
    }
    const strokeSegments = (segments: PixelSegment[]) => {
      ctx.beginPath();
      segments.forEach(([a, b]) => { ctx.moveTo(...xy(a) as Pixel); ctx.lineTo(...xy(b) as Pixel); });
      ctx.stroke();
    };
    visibleResults.vectors.forEach(vector => {
      const selected = vector.id === props.selectedId;
      const color = selected ? '#ffffff' : labelColor(vector.label);
      if (vector.polygon.length >= 3) {
        ctx.fillStyle = color;
        ctx.globalAlpha = selected ? 0.2 : 0.12;
        ctx.beginPath();
        vector.polygon.forEach((point, i) => { const p = xy(point) as Pixel; if (!i) ctx.moveTo(...p); else ctx.lineTo(...p); });
        ctx.closePath(); ctx.fill('evenodd'); ctx.globalAlpha = 1;
      }
      ctx.strokeStyle = color; ctx.lineWidth = selected ? 2.2 : 1.6;
      strokeSegments(vector.segments);
    });
    visibleResults.boxes.forEach(box => {
        const color = box.id === props.selectedId ? '#ffffff' : labelColor(box.label);
        ctx.strokeStyle = color;
        ctx.lineWidth = box.id === props.selectedId ? 1.8 : 1.1;
        strokeSegments(box.segments);
        if (box.anchor) {
          ctx.fillStyle = color;
          ctx.font = '10px ui-monospace, monospace';
          ctx.textAlign = 'left';
          ctx.fillText(box.trackId, box.anchor[0] * scale + 2, Math.max(12, box.anchor[1] * scale - 5));
        }
    });
    visibleResults.imagePolygons.forEach((polygon) => {
      if (!polygon.points.length) return;
      ctx.fillStyle = `${labelColor(polygon.label)}44`;
      // hsl colours need a separate globalAlpha instead of hexadecimal alpha.
      ctx.globalAlpha = 0.24;
      ctx.fillStyle = labelColor(polygon.label);
      ctx.beginPath();
      polygon.points.forEach((point, index) => {
        const [x, y] = xy(point);
        if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.lineWidth = polygon.id === props.selectedId ? 2 : 1.2;
      ctx.strokeStyle = polygon.id === props.selectedId ? '#ffffff' : labelColor(polygon.label);
      ctx.stroke();
    });
    if (props.draftPolygon?.length) {
      ctx.strokeStyle = '#f7d778';
      ctx.fillStyle = '#f7d778';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      props.draftPolygon.forEach((point, index) => {
        const [x, y] = xy(point);
        if (!index) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
      props.draftPolygon.forEach((point) => {
        const [x, y] = xy(point);
        ctx.beginPath(); ctx.arc(x, y, 2.5, 0, Math.PI * 2); ctx.fill();
      });
    }
    const promptBox = draftModelBox ?? props.modelBox;
    if (promptBox?.length === 4) {
      const a = xy(promptBox.slice(0, 2)), b = xy(promptBox.slice(2));
      ctx.strokeStyle = '#ffe28b'; ctx.fillStyle = '#ffe28b'; ctx.lineWidth = 1.8; ctx.setLineDash([5, 3]);
      ctx.strokeRect(a[0], a[1], b[0] - a[0], b[1] - a[1]);
      ctx.globalAlpha = 0.08; ctx.fillRect(a[0], a[1], b[0] - a[0], b[1] - a[1]); ctx.globalAlpha = 1; ctx.setLineDash([]);
    }
    (props.modelPoints ?? []).forEach((point, index) => {
      const [x, y] = xy(point), positive = (props.modelPointLabels?.[index] ?? 1) === 1;
      ctx.fillStyle = positive ? '#56efac' : '#ff7684'; ctx.strokeStyle = '#101925'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, 5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      ctx.strokeStyle = '#0b1420'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x - 2.5, y); ctx.lineTo(x + 2.5, y);
      if (positive) { ctx.moveTo(x, y - 2.5); ctx.lineTo(x, y + 2.5); }
      ctx.stroke();
    });
    ctx.restore();
  }, [props, imageState, size, visibleResults, draftModelBox]);

  const originalPixel = (clientX: number, clientY: number, clamp = false): Pixel | null => {
    if (!host.current) return null;
    const rect = host.current.getBoundingClientRect();
    const { x, y, width, height } = fit.current;
    if (!width || !height) return null;
    const px = clientX - rect.left - x, py = clientY - rect.top - y;
    if (!clamp && (px < 0 || py < 0 || px > width || py > height)) return null;
    return [Math.max(0, Math.min(width, px)) / width * props.camera.width, Math.max(0, Math.min(height, py)) / height * props.camera.height];
  };

  return <div
    ref={host}
    className={props.className}
    data-testid={`camera-${props.camera.id}`}
    data-image-state={imageState}
    data-projected-box-count={visibleResults.boxes.length}
    data-projected-vector-count={visibleResults.vectors.length}
    data-projected-polygon-count={visibleResults.imagePolygons.length}
    data-projected-point-count={props.showProjection ? props.projections?.length ?? 0 : 0}
    data-raw-projection-signature={props.showProjection && props.projections?.length ? `${props.projections.length}:${props.projections[0].index}:${props.projections[0].u.toFixed(4)},${props.projections[0].v.toFixed(4)}` : ''}
    data-projection-signature={signature}
    data-model-point-count={props.modelPoints?.length ?? 0}
    data-model-box={props.modelBox?.join(',') ?? ''}
    style={{ position: 'relative', width: '100%', height: '100%', minHeight: 92, overflow: 'hidden', background: '#0b1420', outline: props.selected ? '1px solid #60dbc3' : undefined, outlineOffset: -1, cursor: props.tool === 'polygon' || props.modelPromptMode ? 'crosshair' : props.onSelect ? 'pointer' : 'default' }}
    onPointerDown={event => {
      if (props.modelPromptMode !== 'box' || event.button !== 0 || imageState !== 'ready') return;
      if (props.onSelect?.() === false) return;
      const point = originalPixel(event.clientX, event.clientY);
      if (!point) return;
      event.preventDefault();
      modelDrag.current = { start: point, pointerId: event.pointerId };
      event.currentTarget.setPointerCapture(event.pointerId);
      setDraftModelBox([...point, ...point]);
    }}
    onPointerMove={event => {
      if (!modelDrag.current || event.pointerId !== modelDrag.current.pointerId) return;
      const point = originalPixel(event.clientX, event.clientY, true);
      if (!point) return;
      const start = modelDrag.current.start;
      setDraftModelBox([Math.min(start[0], point[0]), Math.min(start[1], point[1]), Math.max(start[0], point[0]), Math.max(start[1], point[1])]);
    }}
    onPointerUp={event => {
      const drag = modelDrag.current;
      if (!drag || event.pointerId !== drag.pointerId) return;
      const point = originalPixel(event.clientX, event.clientY, true);
      modelDrag.current = null;
      setDraftModelBox(null);
      if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
      if (!point) return;
      const box: [number, number, number, number] = [Math.min(drag.start[0], point[0]), Math.min(drag.start[1], point[1]), Math.max(drag.start[0], point[0]), Math.max(drag.start[1], point[1])];
      if (box[2] - box[0] >= 1 && box[3] - box[1] >= 1) props.onModelBox?.(box);
    }}
    onPointerCancel={() => { modelDrag.current = null; setDraftModelBox(null); }}
    onClick={(event) => {
      if (props.onSelect?.() === false) return;
      if (props.modelPromptMode) {
        if (props.modelPromptMode === 'point' && imageState === 'ready') {
          const point = originalPixel(event.clientX, event.clientY);
          if (point) props.onModelPoint?.(point, event.shiftKey ? 0 : 1);
        }
        return;
      }
      if (props.tool !== 'polygon' || event.detail > 1) return;
      const point = originalPixel(event.clientX, event.clientY);
      if (point) props.onAddPolygonPoint?.(point);
    }}
    onDoubleClick={(event) => { event.preventDefault(); if (!props.modelPromptMode && props.tool === 'polygon' && props.selected !== false) props.onFinishPolygon?.(); }}
  >
    <canvas ref={canvas} aria-label={`${props.camera.name} 标注投影`} style={{ width: '100%', height: '100%', display: 'block' }} />
    <div style={{ position: 'absolute', top: 7, left: 8, display: 'flex', alignItems: 'center', gap: 5, background: '#0a121ec9', padding: '4px 6px', borderRadius: 3, pointerEvents: 'none', font: '9px ui-monospace, monospace', color: props.selected ? '#65dfc8' : '#d2dce7', letterSpacing: '.055em' }}>
      <span style={{ width: 4, height: 4, borderRadius: '50%', background: imageState === 'ready' ? '#63d4b3' : '#e8ac70' }} />
      {props.camera.name || props.camera.id}
    </div>
    <div style={{ position: 'absolute', bottom: 6, right: 7, color: '#bbcedb', font: '8px ui-monospace, monospace', background: '#0a121e99', padding: '2px 4px', pointerEvents: 'none' }}>{props.camera.width} × {props.camera.height}</div>
  </div>;
}

export default CameraView;
