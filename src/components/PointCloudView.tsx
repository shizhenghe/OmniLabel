import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { Annotations, Box, Frame, Vec3, Voxel } from '../types';

export interface PointCloudViewProps {
  frame: Frame | null;
  annotations: Annotations;
  selectedId?: string | null;
  onSelect?: (id: string | null) => void;
  onCreateBox?: (center: Vec3) => void;
  onAddVectorPoint?: (point: Vec3) => void;
  onPaint?: (indices: number[]) => void;
  tool?: 'select' | 'box' | 'vector' | 'brush';
  view?: 'perspective' | 'top';
  overlayPoints?: number[][];
  overlayPointLabels?: Record<string, string>;
  voxels?: Voxel[];
  /** Show measured free-space cells; unknown cells are deliberately absent. */
  showFree?: boolean;
  voxelSize?: number;
  showBoxes?: boolean;
  showVectors?: boolean;
  semantic?: boolean;
  draftVector?: Vec3[];
  brushRadius?: number;
  /** Fallback drawing plane in LiDAR metres; surface hits retain measured Z. */
  groundZ?: number;
  className?: string;
}

const COLORS: Record<string, string> = {
  car: '#37ddbf', truck: '#fbad5c', bus: '#fbad5c', vehicle: '#37ddbf',
  pedestrian: '#f476aa', cyclist: '#aa96ff', bicycle: '#aa96ff',
  road: '#65acff', ground: '#65acff', sidewalk: '#cd9fe8',
  vegetation: '#a3d465', building: '#c39dff', pole: '#f1d36a',
  lane: '#f7d16a', lane_line: '#f7d16a', lane_marking: '#f7d16a',
  road_boundary: '#37ddbf', stop_line: '#f476aa',
  curb: '#ef8d85', crosswalk: '#aa96ff', drivable_area: '#65acff',
  unknown: '#98a8b8', unlabeled: '#98a8b8', free: '#446f8d',
};

export function labelColor(label: string): string {
  if (COLORS[label]) return COLORS[label];
  let hash = 0;
  for (let i = 0; i < label.length; i += 1) hash = (hash * 31 + label.charCodeAt(i)) | 0;
  return `hsl(${Math.abs(hash) % 360}, 68%, 66%)`;
}

function disposeObject(object: THREE.Object3D) {
  object.traverse((node) => {
    if (node instanceof THREE.Sprite) {
      node.material.map?.dispose();
      node.material.dispose();
      return; // Sprite geometry is shared internally by Three.js.
    }
    if (node instanceof THREE.InstancedMesh) node.dispose();
    const drawable = node as THREE.Mesh;
    drawable.geometry?.dispose();
    const materials = Array.isArray(drawable.material) ? drawable.material : [drawable.material];
    materials.forEach((material) => material?.dispose());
  });
}

function makeLabel(text: string, color: string): THREE.Sprite {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 80;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = 'rgba(9, 20, 31, .85)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.strokeRect(2, 2, canvas.width - 4, canvas.height - 4);
  ctx.fillStyle = color;
  ctx.font = '500 32px monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);
  const texture = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false, transparent: true }));
  sprite.scale.set(5, 0.78, 1);
  return sprite;
}

function boxMesh(box: Box, selected: boolean) {
  const group = new THREE.Group();
  group.position.fromArray(box.center);
  group.rotation.z = box.yaw;
  const color = selected ? '#ffffff' : labelColor(box.label);
  const geometry = new THREE.BoxGeometry(...box.size);
  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geometry), new THREE.LineBasicMaterial({ color, linewidth: selected ? 2 : 1 }));
  geometry.dispose();
  group.add(edges);
  const volume = new THREE.Mesh(new THREE.BoxGeometry(...box.size), new THREE.MeshBasicMaterial({ color, opacity: selected ? 0.14 : 0.045, transparent: true, depthWrite: false }));
  volume.userData.boxId = box.id;
  group.add(volume);
  const length = box.size[0] * 0.65;
  const arrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, box.size[2] / 2 + 0.08), length, color, Math.min(0.8, length * 0.3), 0.35);
  group.add(arrow);
  const label = makeLabel(`${box.label.toUpperCase()}  ${box.track_id}`, color);
  label.position.z = box.size[2] / 2 + 0.95;
  group.add(label);
  return group;
}

interface Runtime {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  pointsGroup: THREE.Group;
  annotationsGroup: THREE.Group;
  overlaysGroup: THREE.Group;
  cloud: THREE.Points | null;
  grid: THREE.GridHelper;
}

export function PointCloudView(props: PointCloudViewProps) {
  const container = useRef<HTMLDivElement>(null);
  const runtime = useRef<Runtime | null>(null);
  const latest = useRef(props);
  latest.current = props;
  const [error, setError] = useState('');
  const [cursor, setCursor] = useState('');
  const [ready, setReady] = useState(false);
  const pointerStart = useRef<[number, number] | null>(null);

  useEffect(() => {
    const host = container.current;
    if (!host) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
    } catch {
      setError('无法初始化 WebGL。请启用浏览器硬件加速后重试。');
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor('#0b131f');
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    renderer.domElement.style.display = 'block';
    renderer.domElement.setAttribute('aria-label', '三维点云标注视图');
    host.prepend(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(47, 1, 0.05, 4000);
    camera.up.set(0, 0, 1);
    camera.position.set(-31, -36, 35);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(10, 0, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.09;
    controls.minDistance = 2;
    controls.maxDistance = 800;
    controls.maxPolarAngle = Math.PI * 0.94;
    const grid = new THREE.GridHelper(160, 80, '#2c414f', '#172735');
    grid.rotation.x = Math.PI / 2;
    grid.position.z = -1.75;
    scene.add(grid);
    const axes = new THREE.AxesHelper(4);
    axes.position.set(0, 0, -1.7);
    scene.add(axes);
    const pointsGroup = new THREE.Group();
    const annotationsGroup = new THREE.Group();
    const overlaysGroup = new THREE.Group();
    scene.add(pointsGroup, annotationsGroup, overlaysGroup);
    runtime.current = { renderer, scene, camera, controls, pointsGroup, annotationsGroup, overlaysGroup, cloud: null, grid };
    const resize = () => {
      const { width, height } = host.getBoundingClientRect();
      if (width <= 0 || height <= 0) return;
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();
    let animation = 0;
    const render = () => {
      animation = requestAnimationFrame(render);
      controls.update();
      renderer.render(scene, camera);
    };
    render();
    setReady(true);
    return () => {
      setReady(false);
      cancelAnimationFrame(animation);
      observer.disconnect();
      controls.dispose();
      disposeObject(scene);
      renderer.dispose();
      renderer.domElement.remove();
      runtime.current = null;
    };
  }, []);

  useEffect(() => {
    const rt = runtime.current;
    if (!rt) return;
    disposeObject(rt.pointsGroup);
    rt.pointsGroup.clear();
    rt.cloud = null;
    const points = props.frame?.points;
    if (!points?.length) return;
    const positions = new Float32Array(points.length * 3);
    const colors = new Float32Array(points.length * 3);
    const color = new THREE.Color();
    points.forEach((p, i) => {
      positions.set(p.slice(0, 3), i * 3);
      const label = props.annotations.point_labels[String(i)];
      if (props.semantic && label) color.set(labelColor(label));
      else if (props.semantic) color.set('#748a9e');
      else {
        const height = THREE.MathUtils.clamp((p[2] + 2.2) / 7, 0, 1);
        color.setHSL(0.55 - height * 0.5, 0.64, 0.46 + height * 0.17);
      }
      colors.set([color.r, color.g, color.b], i * 3);
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.computeBoundingSphere();
    const cloud = new THREE.Points(geometry, new THREE.PointsMaterial({ size: 0.095, vertexColors: true, sizeAttenuation: true }));
    rt.pointsGroup.add(cloud);
    rt.cloud = cloud;
  }, [ready, props.frame, props.annotations.point_labels, props.semantic]);

  useEffect(() => {
    const rt = runtime.current;
    if (!rt) return;
    disposeObject(rt.annotationsGroup);
    rt.annotationsGroup.clear();
    if (props.showBoxes !== false) props.annotations.boxes.forEach((box) => rt.annotationsGroup.add(boxMesh(box, props.selectedId === box.id)));
    if (props.showVectors !== false) {
      const vectors = [...props.annotations.vectors];
      if (props.draftVector?.length) vectors.push({ id: '__draft__', label: 'lane', points: props.draftVector, closed: false });
      vectors.forEach((vector) => {
        if (!vector.points.length) return;
        const coordinates = vector.points.map((p) => new THREE.Vector3(...p));
        if (vector.closed && coordinates.length > 2) coordinates.push(coordinates[0].clone());
        const color = vector.id === props.selectedId ? '#ffffff' : labelColor(vector.label);
        const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(coordinates), new THREE.LineBasicMaterial({ color, depthTest: false }));
        line.userData.vectorId = vector.id;
        rt.annotationsGroup.add(line);
        const dots = new THREE.Points(new THREE.BufferGeometry().setFromPoints(coordinates), new THREE.PointsMaterial({ color, size: 0.24, sizeAttenuation: true, depthTest: false }));
        rt.annotationsGroup.add(dots);
      });
    }
  }, [ready, props.annotations.boxes, props.annotations.vectors, props.selectedId, props.draftVector, props.showBoxes, props.showVectors]);

  useEffect(() => {
    const rt = runtime.current;
    if (!rt) return;
    disposeObject(rt.overlaysGroup);
    rt.overlaysGroup.clear();
    if (props.overlayPoints?.length) {
      const coordinates = props.overlayPoints.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
      const geometry = new THREE.BufferGeometry().setFromPoints(coordinates);
      if (props.semantic && props.overlayPointLabels) {
        const colors = new Float32Array(coordinates.length * 3);
        const color = new THREE.Color();
        coordinates.forEach((_, index) => {
          color.set(labelColor(props.overlayPointLabels![String(index)] ?? 'unlabeled'));
          colors.set([color.r, color.g, color.b], index * 3);
        });
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      }
      rt.overlaysGroup.add(new THREE.Points(geometry, new THREE.PointsMaterial({ color: props.semantic && props.overlayPointLabels ? '#ffffff' : '#e7bb72', vertexColors: Boolean(props.semantic && props.overlayPointLabels), size: 0.09, transparent: true, opacity: 0.7 })));
    }
    const voxels = props.voxels?.filter((voxel) => voxel.state === 'occupied');
    const freeVoxels = props.showFree ? props.voxels?.filter((voxel) => voxel.state === 'free') : undefined;
    if (freeVoxels?.length) {
      const geometry = new THREE.BufferGeometry().setFromPoints(freeVoxels.map((voxel) => new THREE.Vector3(...voxel.center)));
      const freeSpace = new THREE.Points(geometry, new THREE.PointsMaterial({
        color: '#6e9cb5', size: (props.voxelSize ?? 0.5) * 0.2,
        transparent: true, opacity: 0.26, depthWrite: false, sizeAttenuation: true,
      }));
      freeSpace.name = 'observed-free-space';
      rt.overlaysGroup.add(freeSpace);
    }
    if (voxels?.length) {
      const size = (props.voxelSize ?? 0.5) * 0.92;
      const geometry = new THREE.BoxGeometry(size, size, size);
      const material = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.66 });
      const mesh = new THREE.InstancedMesh(geometry, material, voxels.length);
      const transform = new THREE.Matrix4();
      const color = new THREE.Color();
      voxels.forEach((voxel, index) => {
        transform.makeTranslation(...voxel.center);
        mesh.setMatrixAt(index, transform);
        color.set(labelColor(voxel.label));
        mesh.setColorAt(index, color);
      });
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
      rt.overlaysGroup.add(mesh);
    }
  }, [ready, props.overlayPoints, props.overlayPointLabels, props.semantic, props.voxels, props.voxelSize, props.showFree]);

  useEffect(() => {
    if (runtime.current) runtime.current.grid.position.z = (props.groundZ ?? -1.7) - 0.05;
  }, [ready, props.groundZ]);

  useEffect(() => {
    const rt = runtime.current;
    if (!rt) return;
    const target = rt.controls.target;
    if (props.view === 'top') {
      rt.camera.position.set(target.x - 0.01, target.y, target.z + 68);
      rt.controls.enableRotate = false;
    } else {
      rt.camera.position.set(target.x - 41, target.y - 36, target.z + 35);
      rt.controls.enableRotate = true;
    }
    rt.controls.update();
  }, [ready, props.view]);

  const rayAt = (clientX: number, clientY: number) => {
    const rt = runtime.current;
    if (!rt) return null;
    const rect = rt.renderer.domElement.getBoundingClientRect();
    const raycaster = new THREE.Raycaster();
    raycaster.params.Points = { threshold: 0.4 };
    raycaster.params.Line = { threshold: 0.3 };
    raycaster.setFromCamera(new THREE.Vector2((clientX - rect.left) / rect.width * 2 - 1, -(clientY - rect.top) / rect.height * 2 + 1), rt.camera);
    return raycaster;
  };

  const onClick = (event: React.MouseEvent<HTMLDivElement>) => {
    const start = pointerStart.current;
    pointerStart.current = null;
    if (!start || Math.hypot(event.clientX - start[0], event.clientY - start[1]) > 5 || event.button !== 0) return;
    const rt = runtime.current;
    const raycaster = rayAt(event.clientX, event.clientY);
    if (!rt || !raycaster) return;
    const current = latest.current;
    if (current.tool === 'select' || !current.tool) {
      const hit = raycaster.intersectObjects(rt.annotationsGroup.children, true).find((item) => item.object.userData.boxId || item.object.userData.vectorId);
      current.onSelect?.(hit?.object.userData.boxId ?? hit?.object.userData.vectorId ?? null);
      return;
    }
    const cloudHit = rt.cloud ? raycaster.intersectObject(rt.cloud)[0] : undefined;
    const ground = raycaster.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 0, 1), -(current.groundZ ?? -1.7)), new THREE.Vector3());
    const point = cloudHit?.point ?? ground;
    if (!point) return;
    const coordinates: Vec3 = [point.x, point.y, point.z];
    if (current.tool === 'box') current.onCreateBox?.(coordinates);
    else if (current.tool === 'vector') current.onAddVectorPoint?.(coordinates);
    else if (current.tool === 'brush') {
      const radius = current.brushRadius ?? 1.5;
      const indices: number[] = [];
      current.frame?.points.forEach((p, index) => {
        if (Math.hypot(p[0] - point.x, p[1] - point.y, p[2] - point.z) <= radius) indices.push(index);
      });
      current.onPaint?.(indices);
    }
  };

  return <div
    ref={container}
    className={props.className}
    data-testid="point-cloud-view"
    style={{ position: 'relative', width: '100%', height: '100%', minHeight: 240, overflow: 'hidden', cursor: props.tool && props.tool !== 'select' ? 'crosshair' : 'grab' }}
    onPointerDown={(event) => { pointerStart.current = [event.clientX, event.clientY]; }}
    onClick={onClick}
    onPointerMove={(event) => {
      if (!runtime.current || event.buttons) return;
      const ray = rayAt(event.clientX, event.clientY);
      const point = ray?.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 0, 1), -(latest.current.groundZ ?? -1.7)), new THREE.Vector3());
      if (point) setCursor(`X ${point.x.toFixed(1)} · Y ${point.y.toFixed(1)} m`);
    }}
    onPointerLeave={() => setCursor('')}
  >
    {error && <div role="alert" style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', padding: 24, color: '#f3c2b2', background: '#0b131f', textAlign: 'center' }}>{error}</div>}
    <div style={{ position: 'absolute', right: 14, bottom: 12, pointerEvents: 'none', color: '#8294a9', font: '10px ui-monospace, monospace', display: 'flex', gap: 16 }}>
      {cursor && <span>{cursor}</span>}<span>X 前 · Y 左 · Z 上</span>
    </div>
    {!props.frame && !error && <div style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', color: '#8294a9', pointerEvents: 'none' }}>请选择一个 clip</div>}
  </div>;
}

export default PointCloudView;
