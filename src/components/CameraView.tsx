import { useEffect, useRef, useState } from 'react';
import type { Annotations, Box, Camera, Frame, ProjectionPoint, Vec3 } from '../types';
import { labelColor } from './PointCloudView';

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
  showBoxes?: boolean;
  selectedId?: string | null;
  className?: string;
}

function transformPoint(matrix: number[][], point: number[]): Vec3 {
  return [0, 1, 2].map((row) => matrix[row][0] * point[0] + matrix[row][1] * point[1] + matrix[row][2] * point[2] + matrix[row][3]) as Vec3;
}

/** Shared calibrated pinhole/Brown-Conrady projection, optical +Z in front. */
export function projectLidarPoint(point: number[], camera: Camera, frame: Frame): [number, number, number] | null {
  return projectCameraPoint(transformPoint(camera.camera_from_ego, transformPoint(frame.lidar_to_ego, point)), camera);
}

function projectCameraPoint(point: number[], camera: Camera): [number, number, number] | null {
  if (point[2] <= 0.05) return null;
  const x = point[0] / point[2];
  const y = point[1] / point[2];
  const [k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0] = camera.distortion ?? [];
  const r2 = x * x + y * y;
  const radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3;
  const xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
  const yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;
  const k = camera.intrinsics;
  const u = k[0][0] * xd + k[0][1] * yd + k[0][2];
  const v = k[1][0] * xd + k[1][1] * yd + k[1][2];
  return Number.isFinite(u) && Number.isFinite(v) ? [u, v, point[2]] : null;
}

const EDGES = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];

function boxCorners(box: Box): Vec3[] {
  const [length, width, height] = box.size.map((size) => size / 2);
  return [[-length, -width, -height], [length, -width, -height], [length, width, -height], [-length, width, -height], [-length, -width, height], [length, -width, height], [length, width, height], [-length, width, height]].map(([x, y, z]) => [
    box.center[0] + x * Math.cos(box.yaw) - y * Math.sin(box.yaw),
    box.center[1] + x * Math.sin(box.yaw) + y * Math.cos(box.yaw),
    box.center[2] + z,
  ] as Vec3);
}

export function CameraView(props: CameraViewProps) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const host = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const fit = useRef({ x: 0, y: 0, width: 0, height: 0 });
  const [imageState, setImageState] = useState<'loading' | 'ready' | 'missing' | 'error'>('loading');
  const [size, setSize] = useState({ width: 0, height: 0 });
  const source = props.frame?.images[props.camera.id];

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
    if (props.showProjection !== false) {
      for (const point of props.projections ?? []) {
        const [x, y] = xy([point.u, point.v]);
        // Current annotations are authoritative, including erased labels; a
        // projection response may have been computed before unsaved edits.
        const label = props.annotations.point_labels[String(point.index)];
        ctx.fillStyle = label ? labelColor(label) : `hsl(${Math.max(0, Math.min(210, point.depth * 3.4))}, 90%, 64%)`;
        ctx.fillRect(x - 0.85, y - 0.85, 1.7, 1.7);
      }
    }
    if (props.frame && props.showBoxes !== false) {
      props.annotations.boxes.forEach((box) => {
        const points = boxCorners(box).map((point) => transformPoint(props.camera.camera_from_ego, transformPoint(props.frame!.lidar_to_ego, point)));
        const color = box.id === props.selectedId ? '#ffffff' : labelColor(box.label);
        ctx.strokeStyle = color;
        ctx.lineWidth = box.id === props.selectedId ? 1.8 : 1.1;
        EDGES.forEach(([from, to]) => {
          let start = [...points[from]];
          let end = [...points[to]];
          if (start[2] < 0.06 && end[2] < 0.06) return;
          if (start[2] < 0.06 || end[2] < 0.06) {
            if (end[2] < 0.06) [start, end] = [end, start];
            const t = (0.06 - start[2]) / (end[2] - start[2]);
            start = start.map((value, axis) => value + t * (end[axis] - value));
          }
          const a = projectCameraPoint(start, props.camera);
          const b = projectCameraPoint(end, props.camera);
          if (!a || !b) return;
          ctx.beginPath();
          ctx.moveTo(...xy(a) as [number, number]);
          ctx.lineTo(...xy(b) as [number, number]);
          ctx.stroke();
        });
        const center = projectLidarPoint([box.center[0], box.center[1], box.center[2] + box.size[2] / 2], props.camera, props.frame!);
        if (center && center[0] >= 0 && center[0] <= width && center[1] >= 0 && center[1] <= height) {
          ctx.fillStyle = color;
          ctx.font = '9px ui-monospace, monospace';
          ctx.textAlign = 'left';
          ctx.fillText(box.track_id, center[0] * scale, center[1] * scale - 5);
        }
      });
    }
    const polygons = props.annotations.image_polygons.filter((polygon) => polygon.camera_id === props.camera.id);
    polygons.forEach((polygon) => {
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
    ctx.restore();
  }, [props, imageState, size]);

  return <div
    ref={host}
    className={props.className}
    data-testid={`camera-${props.camera.id}`}
    style={{ position: 'relative', width: '100%', height: '100%', minHeight: 92, overflow: 'hidden', background: '#0b1420', outline: props.selected ? '1px solid #60dbc3' : undefined, outlineOffset: -1, cursor: props.tool === 'polygon' ? 'crosshair' : props.onSelect ? 'pointer' : 'default' }}
    onClick={(event) => {
      if (props.onSelect?.() === false) return;
      if (props.tool !== 'polygon' || event.detail > 1) return;
      const rect = event.currentTarget.getBoundingClientRect();
      const { x, y, width, height } = fit.current;
      const px = event.clientX - rect.left - x;
      const py = event.clientY - rect.top - y;
      if (px < 0 || py < 0 || px > width || py > height || !width || !height) return;
      props.onAddPolygonPoint?.([px / width * props.camera.width, py / height * props.camera.height]);
    }}
    onDoubleClick={(event) => { event.preventDefault(); if (props.tool === 'polygon' && props.selected !== false) props.onFinishPolygon?.(); }}
  >
    <canvas ref={canvas} style={{ width: '100%', height: '100%', display: 'block' }} />
    <div style={{ position: 'absolute', top: 7, left: 8, display: 'flex', alignItems: 'center', gap: 5, background: '#0a121ec9', padding: '4px 6px', borderRadius: 3, pointerEvents: 'none', font: '9px ui-monospace, monospace', color: props.selected ? '#65dfc8' : '#d2dce7', letterSpacing: '.055em' }}>
      <span style={{ width: 4, height: 4, borderRadius: '50%', background: imageState === 'ready' ? '#63d4b3' : '#e8ac70' }} />
      {props.camera.name || props.camera.id}
    </div>
    <div style={{ position: 'absolute', bottom: 6, right: 7, color: '#bbcedb', font: '8px ui-monospace, monospace', background: '#0a121e99', padding: '2px 4px', pointerEvents: 'none' }}>{props.camera.width} × {props.camera.height}</div>
  </div>;
}

export default CameraView;
