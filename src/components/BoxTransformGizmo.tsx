import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import type { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { Box, Vec3 } from '../types';

export type BoxTransformMode = 'translate' | 'scale' | 'rotate';

interface Props {
  box: Box;
  mode: BoxTransformMode;
  camera: THREE.Camera;
  canvas: HTMLCanvasElement;
  controls: OrbitControls;
  onUpdate: (id: string, patch: Partial<Box>) => void;
  onStart?: () => void;
  onEnd?: () => void;
}

interface Handle {
  id: string;
  action: BoxTransformMode;
  label: string;
  color: string;
  world: THREE.Vector3;
  axis: THREE.Vector3;
  dimension?: number;
}

interface ScreenHandle extends Handle { x: number; y: number; visible: boolean }
interface Layout { center: {x: number; y: number}; handles: ScreenHandle[]; width: number; height: number }
interface Drag {
  box: Box;
  handle: Handle;
  plane: THREE.Plane;
  start: THREE.Vector3;
  pointerId: number;
  element: HTMLButtonElement;
  orbitEnabled: boolean;
}

const AXES = ['x', 'y', 'z'];
const DIMENSIONS = ['长', '宽', '高'];
const COLORS = ['#ff8080', '#7de2aa', '#78adff'];

export function normalizeYaw(value: number) {
  return Math.atan2(Math.sin(value), Math.cos(value));
}

/** Resize a yaw-oriented face while leaving the opposite face fixed. */
export function resizeBoxFace(box: Box, dimension: number, sign: number, displacement: number): Partial<Box> {
  const size = [...box.size] as Vec3;
  size[dimension] = Math.max(0.05, Math.min(1000, size[dimension] + displacement));
  const halfDelta = (size[dimension] - box.size[dimension]) / 2;
  const axis = dimension === 0
    ? [Math.cos(box.yaw), Math.sin(box.yaw), 0]
    : dimension === 1 ? [-Math.sin(box.yaw), Math.cos(box.yaw), 0] : [0, 0, 1];
  return { size, center: box.center.map((value, index) => value + axis[index] * sign * halfDelta) as Vec3 };
}

function makeHandles(box: Box, mode: BoxTransformMode): Handle[] {
  const center = new THREE.Vector3(...box.center);
  if (mode === 'rotate') {
    const radius = Math.max(box.size[0], box.size[1]) * 0.7 + 1;
    return [{ id: 'rotate-z', action: 'rotate', label: '拖动旋转航向角', color: '#f7d16a',
      world: center.clone().add(new THREE.Vector3(Math.cos(box.yaw), Math.sin(box.yaw), 0).multiplyScalar(radius)),
      axis: new THREE.Vector3(0, 0, 1) }];
  }
  if (mode === 'translate') {
    const length = Math.max(...box.size) * 0.65 + 1.4;
    return [
      ...AXES.map((name, dimension) => {
        const axis = new THREE.Vector3().setComponent(dimension, 1);
        return { id: `translate-${name}`, action: 'translate' as const, label: `拖动沿 ${name.toUpperCase()} 平移`, color: COLORS[dimension],
          world: center.clone().addScaledVector(axis, length), axis, dimension };
      }),
      { id: 'translate-xy', action: 'translate', label: '拖动在水平面平移', color: '#ffffff', world: center, axis: new THREE.Vector3() },
    ];
  }
  const faces: Handle[] = AXES.flatMap((name, dimension) => [-1, 1].map(sign => {
    const axis = new THREE.Vector3().setComponent(dimension, sign);
    axis.applyAxisAngle(new THREE.Vector3(0, 0, 1), box.yaw);
    return { id: `scale-${name}${sign < 0 ? '-negative' : ''}`, action: 'scale' as const, label: `拖动调整 ${name.toUpperCase()}${sign < 0 ? '负' : '正'}侧尺寸`,
      color: COLORS[dimension], world: center.clone().addScaledVector(axis, box.size[dimension] / 2), axis, dimension };
  }));
  // The default resize tool also exposes a center grip, so moving and sizing
  // a selected box require no tool switch or property-panel input.
  return [...faces, {id: 'translate-xy', action: 'translate', label: '拖动在水平面平移',
    color: '#ffffff', world: center, axis: new THREE.Vector3()}];
}

function pointerRay(event: {clientX: number; clientY: number}, props: Props) {
  const rect = props.canvas.getBoundingClientRect();
  const ray = new THREE.Raycaster();
  ray.setFromCamera(new THREE.Vector2((event.clientX - rect.left) / rect.width * 2 - 1,
    -(event.clientY - rect.top) / rect.height * 2 + 1), props.camera);
  return ray.ray;
}

/** Actual pointer-captured 3D handles, projected into an accessible DOM overlay.
 *
 * Keeping a drag snapshot separate from the controlled box means live React
 * updates and mesh rebuilds cannot detach the active handle or reset its delta.
 */
export function BoxTransformGizmo(props: Props) {
  const latest = useRef(props);
  latest.current = props;
  const drag = useRef<Drag | null>(null);
  const [active, setActive] = useState<string | null>(null);
  const [layout, setLayout] = useState<Layout | null>(null);

  const endDrag = () => {
    const current = drag.current;
    if (!current) return;
    drag.current = null;
    latest.current.controls.enabled = current.orbitEnabled;
    if (current.element.hasPointerCapture(current.pointerId)) current.element.releasePointerCapture(current.pointerId);
    setActive(null);
    latest.current.onEnd?.();
  };

  useEffect(() => {
    let animation = 0;
    let previous = '';
    const render = () => {
      const p = latest.current;
      const rect = p.canvas.getBoundingClientRect();
      const centerWorld = new THREE.Vector3(...p.box.center);
      const center = centerWorld.clone().project(p.camera);
      const handles = makeHandles(p.box, p.mode).map(handle => {
        const pixel = handle.world.clone().project(p.camera);
        let x = (pixel.x + 1) * rect.width / 2;
        let y = (1 - pixel.y) * rect.height / 2;
        const length = handle.world.distanceTo(centerWorld);
        const centerX = (center.x + 1) * rect.width / 2, centerY = (1 - center.y) * rect.height / 2;
        const dx = x - centerX, dy = y - centerY;
        const screenLength = Math.hypot(dx, dy);
        // Keep handles usable while inspecting a large world map. The line
        // extends the projected face/axis; all drag deltas still use the true
        // world-space plane rather than a screen-pixel scale approximation.
        const minimum = handle.action === 'scale' ? 44 : 40;
        if (screenLength > 1e-4 && screenLength < minimum) {
          x = centerX + dx / screenLength * minimum;
          y = centerY + dy / screenLength * minimum;
        }
        return { ...handle, x, y,
          visible: pixel.z > -1 && pixel.z < 1 && (length < 0.001 || screenLength > 1e-4) };
      });
      // Narrow boxes can project a face to a fraction of one pixel at world-map
      // zoom. Keep that grip and separate overlapping labels in screen space;
      // the measured face positions and world-space drag planes stay unchanged.
      const centerGrip = handles.find(handle => handle.id === 'translate-xy');
      const placed: ScreenHandle[] = centerGrip?.visible ? [centerGrip] : [];
      const dimensions = (handle: ScreenHandle) => handle.action === 'scale' ? [34, 26] : [25, 25];
      handles.filter(handle => handle.visible && handle.id !== 'translate-xy').forEach(handle => {
        const dx = handle.x - (center.x + 1) * rect.width / 2;
        const dy = handle.y - (1 - center.y) * rect.height / 2;
        const length = Math.max(Math.hypot(dx, dy), 1e-4);
        const tangent = {x: -dy / length, y: dx / length};
        const size = dimensions(handle);
        const offsets = [0, 42, -42, 84, -84, 126, -126];
        const desired = {x: handle.x, y: handle.y};
        for (const offset of offsets) {
          const x = desired.x + tangent.x * offset, y = desired.y + tangent.y * offset;
          if (placed.some(other => {
            const otherSize = dimensions(other);
            return Math.abs(x - other.x) < (size[0] + otherSize[0]) / 2 + 5 &&
              Math.abs(y - other.y) < (size[1] + otherSize[1]) / 2 + 5;
          })) continue;
          handle.x = x; handle.y = y;
          break;
        }
        placed.push(handle);
      });
      const next = { center: {x: (center.x + 1) * rect.width / 2, y: (1 - center.y) * rect.height / 2},
        handles, width: rect.width, height: rect.height };
      // Orbit renders every frame; only ask React to update when screen positions
      // or box/mode values change. Rounded pixels prevent subpixel jitter churn.
      const key = JSON.stringify([p.box.id, p.box.center, p.box.size, p.box.yaw, p.mode, next.center.x.toFixed(1), next.center.y.toFixed(1),
        rect.width, rect.height, ...handles.map(h => [h.id, h.x.toFixed(1), h.y.toFixed(1), h.visible])]);
      if (key !== previous) { previous = key; setLayout(next); }
      animation = requestAnimationFrame(render);
    };
    render();
    return () => { cancelAnimationFrame(animation); endDrag(); };
  }, [props.box.id, props.mode]);

  const begin = (event: React.PointerEvent<HTMLButtonElement>, handle: Handle) => {
    if (event.button !== 0) return;
    event.preventDefault(); event.stopPropagation();
    const p = latest.current;
    // Finish any residual orbit damping before fixing the drag plane. Disabled
    // controls still run update() in the render loop, so leaving momentum here
    // would move the camera underneath an otherwise stationary drag.
    const damping = p.controls.enableDamping;
    p.controls.enableDamping = false;
    p.controls.update();
    p.controls.enableDamping = damping;
    const center = new THREE.Vector3(...p.box.center);
    let normal: THREE.Vector3;
    if (handle.action === 'rotate' || handle.id === 'translate-xy') normal = new THREE.Vector3(0, 0, 1);
    else {
      normal = p.camera.getWorldDirection(new THREE.Vector3());
      normal.addScaledVector(handle.axis, -normal.dot(handle.axis));
      if (normal.lengthSq() < 1e-6) {
        normal.copy(p.camera.up).addScaledVector(handle.axis, -p.camera.up.dot(handle.axis));
      }
      if (normal.lengthSq() < 1e-6) return;
      normal.normalize();
    }
    const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(normal, handle.action === 'rotate' ? center : handle.world);
    const start = pointerRay(event, p).intersectPlane(plane, new THREE.Vector3());
    if (!start) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = { box: {...p.box, center: [...p.box.center], size: [...p.box.size]}, handle, plane,
      start, pointerId: event.pointerId, element: event.currentTarget, orbitEnabled: p.controls.enabled };
    p.controls.enabled = false;
    setActive(handle.id);
    p.onStart?.();
  };

  const move = (event: React.PointerEvent<HTMLButtonElement>) => {
    const current = drag.current;
    if (!current || event.pointerId !== current.pointerId) return;
    event.preventDefault(); event.stopPropagation();
    const hit = pointerRay(event, latest.current).intersectPlane(current.plane, new THREE.Vector3());
    if (!hit) return;
    const delta = hit.clone().sub(current.start);
    let patch: Partial<Box>;
    if (current.handle.action === 'rotate') {
      const origin = new THREE.Vector3(...current.box.center);
      const first = current.start.clone().sub(origin), now = hit.clone().sub(origin);
      patch = {yaw: normalizeYaw(current.box.yaw + Math.atan2(now.y, now.x) - Math.atan2(first.y, first.x))};
    } else if (current.handle.action === 'scale') {
      const dimension = current.handle.dimension!;
      patch = resizeBoxFace(current.box, dimension, current.handle.id.endsWith('-negative') ? -1 : 1, delta.dot(current.handle.axis));
    } else {
      const displacement = current.handle.id === 'translate-xy' ? new THREE.Vector3(delta.x, delta.y, 0)
        : current.handle.axis.clone().multiplyScalar(delta.dot(current.handle.axis));
      patch = {center: current.box.center.map((value, index) => value + displacement.getComponent(index)) as Vec3};
    }
    latest.current.onUpdate(current.box.id, patch);
  };

  if (!layout) return null;
  return <div data-testid="box-transform-gizmo" data-box-id={props.box.id} data-mode={props.mode}
    style={{position: 'absolute', inset: 0, pointerEvents: 'none', overflow: 'hidden', zIndex: 6}}>
    <svg width={layout.width} height={layout.height} style={{position: 'absolute', inset: 0, overflow: 'visible'}} aria-hidden="true">
      {layout.handles.filter(h => h.visible && h.id !== 'translate-xy').map(handle => <line key={handle.id}
        x1={layout.center.x} y1={layout.center.y} x2={handle.x} y2={handle.y}
        stroke={handle.color} strokeWidth={active === handle.id ? 3 : 1.6} opacity={0.85} />)}
    </svg>
    {layout.handles.map(handle => <button key={handle.id} type="button" title={handle.label} aria-label={handle.label}
      data-testid={`box-handle-${handle.id}`} data-action={handle.action} data-world-x={handle.world.x} data-world-y={handle.world.y} data-world-z={handle.world.z}
      onPointerDown={event => begin(event, handle)} onPointerMove={move}
      onPointerUp={event => {event.stopPropagation(); endDrag();}}
      onPointerCancel={endDrag} onLostPointerCapture={() => {if (drag.current) endDrag();}}
      onClick={event => {event.preventDefault(); event.stopPropagation();}}
      style={{position: 'absolute', left: handle.x, top: handle.y, transform: 'translate(-50%,-50%)',
        width: handle.action === 'scale' ? 34 : 25, height: handle.action === 'scale' ? 26 : 25,
        display: handle.visible || active === handle.id ? 'grid' : 'none', placeItems: 'center', padding: 0,
        background: active === handle.id ? handle.color : '#0d1927e8', color: active === handle.id ? '#10212e' : handle.color,
        border: `2px solid ${handle.color}`, borderRadius: handle.action === 'scale' ? 3 : '50%',
        boxShadow: '0 0 0 2px #09111bcc', pointerEvents: 'auto', touchAction: 'none', cursor: active ? 'grabbing' : 'grab',
        font: '700 10px ui-monospace,monospace', userSelect: 'none'}}>
      {handle.id === 'translate-xy' ? '↔' : handle.action === 'rotate' ? '↻' : handle.action === 'scale'
        ? `${DIMENSIONS[handle.dimension!]}${handle.id.endsWith('-negative') ? '−' : '+'}` : AXES[handle.dimension!].toUpperCase()}
    </button>)}
  </div>;
}
