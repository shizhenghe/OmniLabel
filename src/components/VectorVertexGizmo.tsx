import {useEffect, useRef, useState} from 'react';
import * as THREE from 'three';
import type {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import type {Vec3, Vector} from '../types';

interface Props {
  vector: Vector;
  camera: THREE.Camera;
  renderOrigin: Vec3;
  canvas: HTMLCanvasElement;
  controls: OrbitControls;
  onUpdate: (id: string, index: number, point: Vec3) => void;
  onStart?: () => void;
  onEnd?: () => void;
}

interface ScreenVertex {index: number; x: number; y: number; visible: boolean; point: Vec3}
interface Drag {
  vectorId: string;
  index: number;
  point: Vec3;
  plane: THREE.Plane;
  start: THREE.Vector3;
  pointerId: number;
  element: HTMLButtonElement;
  orbitEnabled: boolean;
}

function pointerRay(event: {clientX: number; clientY: number}, props: Props) {
  const rect = props.canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return null;
  const raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(new THREE.Vector2((event.clientX - rect.left) / rect.width * 2 - 1,
    -(event.clientY - rect.top) / rect.height * 2 + 1), props.camera);
  // Match the floating-origin renderer while retaining double-precision world
  // coordinates in the document and in the pointer's intersection plane.
  raycaster.ray.origin.add(new THREE.Vector3(...props.renderOrigin));
  return raycaster.ray;
}

/** Projected handles for existing vectors. A drag snapshots its original vertex
 * and keeps its measured height, so editing a slope does not flatten the line
 * onto the global ground reference or pick a roof behind the cursor. */
export function VectorVertexGizmo(props: Props) {
  const latest = useRef(props);
  latest.current = props;
  const drag = useRef<Drag | null>(null);
  const [active, setActive] = useState<number | null>(null);
  const [vertices, setVertices] = useState<ScreenVertex[]>([]);

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
      const offset = new THREE.Vector3(...p.renderOrigin);
      const next = p.vector.points.map((point, index) => {
        const projected = new THREE.Vector3(...point).sub(offset).project(p.camera);
        const x = (projected.x + 1) * rect.width / 2;
        const y = (1 - projected.y) * rect.height / 2;
        return {index, point, x, y, visible: Number.isFinite(x) && Number.isFinite(y)
          && projected.z > -1 && projected.z < 1 && x >= 0 && x <= rect.width && y >= 0 && y <= rect.height};
      });
      const key = JSON.stringify([p.vector.id, rect.width, rect.height,
        ...next.map(vertex => [vertex.x.toFixed(1), vertex.y.toFixed(1), vertex.visible, vertex.point])]);
      if (previous !== key) {previous = key; setVertices(next);}
      animation = requestAnimationFrame(render);
    };
    render();
    return () => {cancelAnimationFrame(animation); endDrag();};
  }, [props.vector.id]);

  const begin = (event: React.PointerEvent<HTMLButtonElement>, index: number) => {
    if (event.button !== 0 || drag.current) return;
    event.preventDefault(); event.stopPropagation();
    const p = latest.current;
    const point = p.vector.points[index];
    if (!point?.every(Number.isFinite)) return;
    // Flush residual orbit damping before freezing the camera for this drag.
    const damping = p.controls.enableDamping;
    p.controls.enableDamping = false;
    p.controls.update();
    p.controls.enableDamping = damping;
    const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), -point[2]);
    const start = pointerRay(event, p)?.intersectPlane(plane, new THREE.Vector3());
    if (!start) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = {vectorId: p.vector.id, index, point: [...point], plane, start,
      pointerId: event.pointerId, element: event.currentTarget, orbitEnabled: p.controls.enabled};
    p.controls.enabled = false;
    setActive(index);
    p.onStart?.();
  };

  const move = (event: React.PointerEvent<HTMLButtonElement>) => {
    const current = drag.current;
    if (!current || event.pointerId !== current.pointerId) return;
    event.preventDefault(); event.stopPropagation();
    const hit = pointerRay(event, latest.current)?.intersectPlane(current.plane, new THREE.Vector3());
    if (!hit) return;
    const point: Vec3 = [current.point[0] + hit.x - current.start.x,
      current.point[1] + hit.y - current.start.y, current.point[2]];
    if (point.every(Number.isFinite)) latest.current.onUpdate(current.vectorId, current.index, point);
  };

  return <div data-testid="vector-vertex-gizmo" data-vector-id={props.vector.id}
    data-vertex-count={props.vector.points.length} data-dragging={active === null ? 'false' : 'true'}
    style={{position: 'absolute', inset: 0, pointerEvents: 'none', overflow: 'hidden', zIndex: 6}}>
    {vertices.map(vertex => <button key={vertex.index} type="button"
      aria-label={`拖动车道线顶点${vertex.index + 1}`} title={`拖动车道线顶点 ${vertex.index + 1}`}
      data-testid={`vector-vertex-${vertex.index}`} data-vector-id={props.vector.id} data-vertex-index={vertex.index}
      data-world-x={vertex.point[0]} data-world-y={vertex.point[1]} data-world-z={vertex.point[2]}
      onPointerDown={event => begin(event, vertex.index)} onPointerMove={move}
      onPointerUp={event => {event.stopPropagation(); endDrag();}}
      onPointerCancel={endDrag} onLostPointerCapture={() => {if (drag.current) endDrag();}}
      onClick={event => {event.preventDefault(); event.stopPropagation();}}
      style={{position: 'absolute', left: vertex.x, top: vertex.y, transform: 'translate(-50%,-50%)',
        width: 20, height: 20, display: vertex.visible || active === vertex.index ? 'grid' : 'none',
        placeItems: 'center', padding: 0, borderRadius: '50%', border: '2px solid #f7d16a',
        background: active === vertex.index ? '#f7d16a' : '#0d1927eb',
        color: active === vertex.index ? '#10212e' : '#f7d16a', boxShadow: '0 0 0 2px #09111bcc',
        pointerEvents: 'auto', touchAction: 'none', cursor: active === vertex.index ? 'grabbing' : 'grab',
        userSelect: 'none', font: '700 9px ui-monospace,monospace'}}>{vertex.index + 1}</button>)}
  </div>;
}
