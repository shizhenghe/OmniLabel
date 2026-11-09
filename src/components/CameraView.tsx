import { useEffect, useMemo, useRef, useState } from 'react';
import type { Annotations, Camera, Frame, ProjectionPoint, ProjectionRaster } from '../types';
import { labelColor } from './PointCloudView';
import { cameraFromLidar, projectAnnotationResults, projectLidarPoint, projectionSignature, type Pixel, type PixelSegment } from '../geometry/cameraProjection';
import type {LaneCalibrationOverlay} from './LaneCalibrationPanel';

export { projectLidarPoint } from '../geometry/cameraProjection';

export interface CameraViewProps {
  camera: Camera;
  frame: Frame | null;
  /** A decoded image supplied with a complete review frame; null means unavailable. */
  preparedImage?: HTMLImageElement | null;
  annotations: Annotations;
  projections?: ProjectionPoint[];
  projectionRaster?: ProjectionRaster;
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
  calibrationPicking?: boolean;
  onCalibrationPixel?: (pixel: Pixel) => void;
  calibrationAnchors?: {id: string; point: number[]; pixel: Pixel; label: string; selected?: boolean; enabled?: boolean}[];
  laneCalibrationLines?: LaneCalibrationOverlay[];
  labelDraftPoints?: boolean;
  zoomable?: boolean;
}

export function CameraView(props: CameraViewProps) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const host = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const loadedSource = useRef<string | undefined>(undefined);
  const fit = useRef({ x: 0, y: 0, width: 0, height: 0 });
  const [internalImageState, setImageState] = useState<'loading' | 'ready' | 'missing' | 'error'>('loading');
  const [size, setSize] = useState({ width: 0, height: 0 });
  const [viewport, setViewport] = useState({zoom: 1, x: 0, y: 0});
  const pan = useRef<{id: number; x: number; y: number; moved: boolean} | null>(null);
  const suppressClick = useRef(false);
  const modelDrag = useRef<{ start: Pixel; pointerId: number } | null>(null);
  const [draftModelBox, setDraftModelBox] = useState<number[] | null>(null);
  const source = props.frame?.images[props.camera.id];
  const displayImage = props.preparedImage !== undefined ? props.preparedImage : loadedSource.current === source ? imageRef.current : null;
  const imageState = props.preparedImage !== undefined ? props.preparedImage ? 'ready' : source ? 'error' : 'missing'
    : source && loadedSource.current !== source ? 'loading' : internalImageState;
  const projectionTransform = useMemo(() => props.frame ? cameraFromLidar(props.camera, props.frame) : null,
    [props.camera, props.frame?.lidar_to_ego, props.frame?.projection_camera_from_lidar?.[props.camera.id], props.frame?.annotation_camera_from_lidar?.[props.camera.id]]);
  const results = useMemo(() => props.frame ? projectAnnotationResults(props.annotations, props.camera, props.frame)
    : { boxes: [], vectors: [], imagePolygons: [] }, [props.annotations, props.camera, props.frame?.id, projectionTransform]);
  const visibleResults = useMemo(() => ({
    boxes: props.showAnnotationProjection !== false && props.showBoxes !== false ? results.boxes : [],
    vectors: props.showAnnotationProjection !== false && props.showVectors !== false ? results.vectors : [],
    imagePolygons: props.showAnnotationProjection !== false ? results.imagePolygons : [],
  }), [results, props.showAnnotationProjection, props.showBoxes, props.showVectors]);
  const signature = useMemo(() => projectionSignature(visibleResults), [visibleResults]);
  const boundViewport = (value: typeof viewport) => {
    const scale = Math.min(size.width / props.camera.width, size.height / props.camera.height) * value.zoom;
    const dx = Math.max(0, (props.camera.width * scale-size.width)/2);
    const dy = Math.max(0, (props.camera.height * scale-size.height)/2);
    return {...value, x: Math.max(-dx, Math.min(dx, value.x)), y: Math.max(-dy, Math.min(dy, value.y))};
  };
  const zoomAt = (factor: number, clientX?: number, clientY?: number) => {
    if (!props.zoomable || !size.width || !size.height) return;
    const rect = canvas.current?.getBoundingClientRect();
    const x = rect && clientX !== undefined ? (clientX-rect.left)*size.width/rect.width : size.width/2;
    const y = rect && clientY !== undefined ? (clientY-rect.top)*size.height/rect.height : size.height/2;
    setViewport(previous => {
      const zoom = Math.max(1, Math.min(16, previous.zoom*factor));
      const ratio = zoom/previous.zoom;
      return boundViewport({zoom, x: (x-size.width/2)*(1-ratio)+previous.x*ratio,
        y: (y-size.height/2)*(1-ratio)+previous.y*ratio});
    });
  };
  useEffect(() => {setViewport({zoom: 1, x: 0, y: 0}); pan.current = null;}, [props.frame?.id, props.camera.id, props.zoomable, source]);
  useEffect(() => {setViewport(value => boundViewport(value));}, [size.width, size.height]);
  useEffect(() => {
    const element = host.current;
    if (!element || !props.zoomable) return;
    const wheel = (event: WheelEvent) => {
      event.preventDefault();
      if (imageState === 'ready') zoomAt(Math.exp(-Math.max(-250, Math.min(250, event.deltaY))* .002), event.clientX, event.clientY);
    };
    element.addEventListener('wheel', wheel, {passive: false});
    return () => element.removeEventListener('wheel', wheel);
  }, [props.zoomable, size.width, size.height, props.camera.width, props.camera.height, imageState]);

  useEffect(() => { modelDrag.current = null; setDraftModelBox(null); }, [props.modelPromptMode, props.frame?.id, props.camera.id]);

  useEffect(() => {
    if (!host.current) return;
    const element = host.current;
    const observer = new ResizeObserver(() => {
      setSize({ width: element.clientWidth, height: element.clientHeight });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (props.preparedImage !== undefined) return;
    imageRef.current = null;
    loadedSource.current = source;
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
  }, [source, props.preparedImage]);

  useEffect(() => {
    if (!canvas.current || !size.width || !size.height) return;
    const element = canvas.current;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    if (element.width !== Math.round(size.width * ratio)) element.width = Math.round(size.width * ratio);
    if (element.height !== Math.round(size.height * ratio)) element.height = Math.round(size.height * ratio);
    const ctx = element.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.fillStyle = '#0b1420';
    ctx.fillRect(0, 0, size.width, size.height);
    const { width, height } = props.camera;
    const scale = Math.min(size.width / width, size.height / height) * viewport.zoom;
    const displayWidth = width * scale;
    const displayHeight = height * scale;
    const offsetX = (size.width - displayWidth) / 2 + viewport.x;
    const offsetY = (size.height - displayHeight) / 2 + viewport.y;
    fit.current = { x: offsetX, y: offsetY, width: displayWidth, height: displayHeight };
    ctx.save();
    ctx.translate(offsetX, offsetY);
    ctx.beginPath();
    ctx.rect(0, 0, displayWidth, displayHeight);
    ctx.clip();
    const xy = (point: number[]) => [point[0] * scale, point[1] * scale];
    if (imageState === 'ready' && displayImage) {
      ctx.drawImage(displayImage, 0, 0, displayWidth, displayHeight);
    } else {
      ctx.fillStyle = '#121d2a';
      ctx.fillRect(0, 0, displayWidth, displayHeight);
      ctx.fillStyle = '#6e8296';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.font = '12px system-ui, sans-serif';
      ctx.fillText(imageState === 'loading' ? '加载图像…' : imageState === 'error' ? '图像加载失败' : '此帧未提供图像', displayWidth / 2, displayHeight / 2);
    }
    if (props.showProjection === true && props.projectionRaster) {
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(props.projectionRaster.bitmap, 0, 0, displayWidth, displayHeight);
      ctx.imageSmoothingEnabled = true;
    } else if (props.showProjection === true) {
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
    for (const line of props.laneCalibrationLines ?? []) {
      ctx.strokeStyle = line.color; ctx.lineWidth = 2.2; ctx.beginPath();
      let connected = false;
      for (const point of line.points) {
        if (!point) {connected = false; continue;}
        const pixel = xy(point) as Pixel;
        if (connected) ctx.lineTo(...pixel); else ctx.moveTo(...pixel);
        connected = true;
      }
      ctx.stroke();
      if (line.vertices) line.points.forEach((point, index) => {
        if (!point) return;
        const [x, y] = xy(point);
        ctx.fillStyle = line.color; ctx.beginPath(); ctx.arc(x, y, 3.5, 0, Math.PI * 2); ctx.fill();
        ctx.font = '10px sans-serif'; ctx.fillText(String(index + 1), x + 5, y + 10);
      });
      const anchor = line.points.find(point => point !== null);
      if (anchor) {ctx.fillStyle = line.color; ctx.font = '10px sans-serif'; ctx.fillText(line.label, anchor[0] * scale + 4, anchor[1] * scale - 4);}
    }
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
      props.draftPolygon.forEach((point, index) => {
        const [x, y] = xy(point);
        ctx.beginPath(); ctx.arc(x, y, 2.5, 0, Math.PI * 2); ctx.fill();
        if (props.labelDraftPoints) {ctx.font = '11px sans-serif'; ctx.fillText(String(index + 1), x + 4, y - 4);}
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
    for (const anchor of props.calibrationAnchors ?? []) {
      const [x, y] = xy(anchor.pixel);
      const projected = props.frame ? projectLidarPoint(anchor.point, props.camera, props.frame) : null;
      const color = anchor.enabled === false ? '#8292a0' : anchor.selected ? '#ffffff' : '#ffcf70';
      if (projected && projected[0] >= 0 && projected[0] <= width && projected[1] >= 0 && projected[1] <= height) {
        const [px, py] = xy(projected);
        ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(px, py); ctx.stroke(); ctx.setLineDash([]);
        ctx.strokeRect(px - 3, py - 3, 6, 6);
      }
      ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 1.7;
      ctx.beginPath(); ctx.arc(x, y, anchor.selected ? 6 : 4, 0, Math.PI * 2); ctx.stroke();
      ctx.font = 'bold 11px system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillText(anchor.label, x + 7, y - 3);
    }
    ctx.restore();
  }, [imageState, displayImage, size, visibleResults, draftModelBox, viewport, projectionTransform, props.camera,
    props.annotations.point_labels, props.showProjection, props.projectionRaster, props.projections?.length ? props.projections : undefined, props.selectedId,
    props.draftPolygon?.length ? props.draftPolygon : undefined, props.labelDraftPoints,
    props.calibrationAnchors?.length ? props.calibrationAnchors : undefined,
    props.laneCalibrationLines?.length ? props.laneCalibrationLines : undefined,
    props.modelPoints?.length ? props.modelPoints : undefined, props.modelPointLabels?.length ? props.modelPointLabels : undefined, props.modelBox]);

  const originalPixel = (clientX: number, clientY: number, clamp = false): Pixel | null => {
    if (!canvas.current || !size.width || !size.height) return null;
    const rect = canvas.current.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    const { x, y, width, height } = fit.current;
    if (!width || !height) return null;
    const px = (clientX - rect.left) * size.width / rect.width - x;
    const py = (clientY - rect.top) * size.height / rect.height - y;
    if (!clamp && (px < 0 || py < 0 || px > width || py > height)) return null;
    return [Math.max(0, Math.min(width, px)) / width * props.camera.width, Math.max(0, Math.min(height, py)) / height * props.camera.height];
  };

  return <div
    ref={host}
    className={props.className}
    data-testid={`camera-${props.camera.id}`}
    data-image-state={imageState}
    data-image-zoom={viewport.zoom}
    data-image-offset={`${viewport.x},${viewport.y}`}
    tabIndex={props.zoomable ? 0 : undefined}
    onKeyDown={event => {
      if (!props.zoomable || imageState !== 'ready') return;
      if (event.key === '+' || event.key === '=') {event.preventDefault(); zoomAt(1.25);}
      else if (event.key === '-') {event.preventDefault(); zoomAt(1/1.25);}
      else if (event.key === '0') {event.preventDefault(); setViewport({zoom: 1, x: 0, y: 0});}
    }}
    data-active-frame-id={props.frame?.id ?? ''}
    data-rendered-image-frame-id={imageState === 'ready' && displayImage ? props.frame?.id ?? '' : ''}
    data-projected-box-count={visibleResults.boxes.length}
    data-projected-vector-count={visibleResults.vectors.length}
    data-projected-polygon-count={visibleResults.imagePolygons.length}
    data-projected-point-count={props.showProjection ? props.projectionRaster?.pointCount ?? props.projections?.length ?? 0 : 0}
    data-raw-projection-signature={props.showProjection && props.projections?.length ? `${props.projectionRaster?.pointCount ?? props.projections.length}:${props.projections[0].index}:${props.projections[0].u.toFixed(4)},${props.projections[0].v.toFixed(4)}` : ''}
    data-projection-source-count={props.projectionRaster?.sourceCount ?? props.frame?.points.length ?? 0}
    data-projection-pixel-count={props.projectionRaster?.pixelCount ?? ''}
    data-projection-mode={props.projectionRaster ? 'full-map-raster' : 'source-points'}
    data-projection-signature={signature}
    data-model-point-count={props.modelPoints?.length ?? 0}
    data-model-box={props.modelBox?.join(',') ?? ''}
    data-calibration-anchor-count={props.calibrationAnchors?.length ?? 0}
    data-calibration-picking={props.calibrationPicking ? 'true' : 'false'}
    data-lane-calibration-overlay-count={props.laneCalibrationLines?.length ?? 0}
    data-lane-calibration-draft-count={props.labelDraftPoints ? props.draftPolygon?.length ?? 0 : 0}
    style={{ position: 'relative', width: '100%', height: '100%', minHeight: 92, overflow: 'hidden', background: '#0b1420', outline: props.selected ? '1px solid #60dbc3' : undefined, outlineOffset: -1, cursor: props.calibrationPicking || props.tool === 'polygon' || props.modelPromptMode ? 'crosshair' : props.onSelect ? 'pointer' : 'default' }}
    onPointerDown={event => {
      if ((event.target as HTMLElement).closest('button')) return;
      if (props.zoomable) event.currentTarget.focus({preventScroll: true});
      if (props.zoomable && (event.button === 2 || event.button === 1)) {
        event.preventDefault(); suppressClick.current = false;
        pan.current = {id: event.pointerId, x: event.clientX, y: event.clientY, moved: false};
        event.currentTarget.setPointerCapture(event.pointerId); return;
      }
      suppressClick.current = false;
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
      if (pan.current?.id === event.pointerId) {
        const current = pan.current;
        const rect = canvas.current!.getBoundingClientRect();
        const dx = (event.clientX-current.x)*size.width/rect.width, dy = (event.clientY-current.y)*size.height/rect.height;
        current.x = event.clientX; current.y = event.clientY;
        if (dx || dy) current.moved = true;
        setViewport(value => boundViewport({...value, x: value.x+dx, y: value.y+dy}));
        return;
      }
      if (!modelDrag.current || event.pointerId !== modelDrag.current.pointerId) return;
      const point = originalPixel(event.clientX, event.clientY, true);
      if (!point) return;
      const start = modelDrag.current.start;
      setDraftModelBox([Math.min(start[0], point[0]), Math.min(start[1], point[1]), Math.max(start[0], point[0]), Math.max(start[1], point[1])]);
    }}
    onPointerUp={event => {
      if (pan.current?.id === event.pointerId) {
        suppressClick.current = pan.current.moved; pan.current = null;
        if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
        return;
      }
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
    onPointerCancel={() => { pan.current = null; modelDrag.current = null; setDraftModelBox(null); }}
    onLostPointerCapture={() => {pan.current = null;}}
    onContextMenu={event => {if (props.zoomable) event.preventDefault();}}
    onClick={(event) => {
      if (event.button !== 0 || suppressClick.current) {suppressClick.current = false; return;}
      if (props.onSelect?.() === false) return;
      if (props.calibrationPicking) {
        if (imageState === 'ready' && event.detail <= 1) {
          const point = originalPixel(event.clientX, event.clientY);
          if (point) props.onCalibrationPixel?.(point);
        }
        return;
      }
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
    {props.zoomable && <div role="group" aria-label="图像缩放" onPointerDown={event => event.stopPropagation()} onClick={event => event.stopPropagation()}
      style={{position: 'absolute', bottom: 30, left: '50%', transform: 'translateX(-50%)', display: 'flex', alignItems: 'center', gap: 8, padding: '5px 10px', background: '#0b192deb', borderRadius: 5, color: '#d2e6f1', fontSize: 12}}>
      <button type="button" className="button subtle" aria-label="缩小图像" disabled={viewport.zoom <= 1 || imageState !== 'ready'} onClick={() => zoomAt(1/1.25)}>−</button>
      <span aria-label="图像缩放比例">{Math.round(viewport.zoom*100)}%</span>
      <button type="button" className="button subtle" aria-label="放大图像" disabled={viewport.zoom >= 16 || imageState !== 'ready'} onClick={() => zoomAt(1.25)}>＋</button>
      <button type="button" className="button subtle" aria-label="复位图像缩放" onClick={() => setViewport({zoom: 1, x: 0, y: 0})}>适应窗口</button>
      <span style={{whiteSpace: 'nowrap'}}>滚轮缩放 · 右键拖动</span>
    </div>}
    <div style={{ position: 'absolute', top: 7, left: 8, display: 'flex', alignItems: 'center', gap: 5, background: '#0a121ec9', padding: '4px 6px', borderRadius: 3, pointerEvents: 'none', font: '9px ui-monospace, monospace', color: props.selected ? '#65dfc8' : '#d2dce7', letterSpacing: '.055em' }}>
      <span style={{ width: 4, height: 4, borderRadius: '50%', background: imageState === 'ready' ? '#63d4b3' : '#e8ac70' }} />
      {props.camera.name || props.camera.id}
    </div>
    <div style={{ position: 'absolute', bottom: 6, right: 7, color: '#bbcedb', font: '8px ui-monospace, monospace', background: '#0a121e99', padding: '2px 4px', pointerEvents: 'none' }}>{props.camera.width} × {props.camera.height}</div>
  </div>;
}

export default CameraView;
