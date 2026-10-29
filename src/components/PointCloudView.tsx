import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { BoxTransformGizmo, type BoxTransformMode } from './BoxTransformGizmo';
import { VectorVertexGizmo } from './VectorVertexGizmo';
import type { Annotations, Box, Frame, PointBuffer, Vec3, Voxel } from '../types';
import { groundDrawingPoint, intensityBrightness, intensityRange, visiblePointIndices, type PointColorMode } from '../geometry/pointDisplay';
import {renderOriginForPoints, toRenderPoint} from '../geometry/renderCoordinates';

export interface PointCloudViewProps {
  frame: Frame | null;
  /** Flat XYZI rows loaded directly from NPY for very large clouds. */
  binaryPoints?: PointBuffer;
  cloudBounds?: {min: Vec3; max: Vec3};
  intensityRangeHint?: [number, number] | null;
  annotations: Annotations;
  selectedId?: string | null;
  onSelect?: (id: string | null) => void;
  onCreateBox?: (center: Vec3) => void;
  onAddVectorPoint?: (point: Vec3) => void;
  onUpdateVectorPoint?: (id: string, index: number, point: Vec3) => void;
  /** Independent from detection-box editing, disabled during review/calibration. */
  vectorEditable?: boolean;
  calibrationPicking?: boolean;
  onCalibrationPoint?: (point: Vec3) => void;
  calibrationPoints?: {id: string; point: Vec3; label: string; selected?: boolean; enabled?: boolean}[];
  onPaint?: (indices: number[]) => void;
  onUpdateBox?: (id: string, patch: Partial<Box>) => void;
  transformMode?: BoxTransformMode;
  editable?: boolean;
  onTransformStart?: () => void;
  onTransformEnd?: () => void;
  /** All frame.points and annotations are already in world coordinates. */
  sceneMode?: boolean;
  trajectory?: {frame_id: string; timestamp: number; position: Vec3}[];
  activeFrameId?: string;
  /** Stable raw source identity prevents refitting the camera on every live calibration step. */
  autoFitReference?: number[][] | PointBuffer;
  /** Review can follow ego translation while preserving the chosen view direction and zoom. */
  followPosition?: Vec3;
  /** Increment to frame the currently selected box once; dragging never refits. */
  focusSelection?: number;
  tool?: 'select' | 'box' | 'vector' | 'brush';
  view?: 'perspective' | 'top';
  overlayPoints?: number[][];
  overlayPointLabels?: Record<string, string>;
  voxels?: Voxel[];
  /** Show measured free-space cells; unknown cells are deliberately absent. */
  showFree?: boolean;
  voxelSize?: number;
  voxelDimensions?: Vec3;
  showBoxes?: boolean;
  showVectors?: boolean;
  semantic?: boolean;
  /** Explicit mode takes precedence over the legacy semantic switch. */
  colorMode?: PointColorMode;
  /** Display a height slice around groundZ without modifying the source cloud. */
  groundOnly?: boolean;
  groundBand?: number;
  /** Ground vectors use the drawing plane rather than the first roof/tree hit. */
  drawOnGround?: boolean;
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
  lane: '#f7d16a', lane_line: '#f7d16a', lane_marking: '#f7d16a', lane_solid: '#f7d16a', lane_dashed: '#f7d16a',
  road_boundary: '#37ddbf', stop_line: '#f476aa',
  curb: '#ef8d85', crosswalk: '#aa96ff', direction_arrow: '#aa96ff', drivable_area: '#65acff',
  unknown: '#ff3b5c', unlabeled: '#98a8b8', free: '#446f8d',
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

function boxMesh(box: Box, selected: boolean, origin: Vec3) {
  const group = new THREE.Group();
  group.position.fromArray(toRenderPoint(box.center, origin));
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
  trajectoryGroup: THREE.Group;
  calibrationGroup: THREE.Group;
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
  const origin = useMemo<Vec3>(() => {
    if (!props.binaryPoints) return renderOriginForPoints(props.frame?.points ?? []);
    for (let offset = 0; offset < props.binaryPoints.length; offset += 4) {
      const first = [props.binaryPoints[offset], props.binaryPoints[offset + 1], props.binaryPoints[offset + 2]] as Vec3;
      if (first.every(Number.isFinite)) return first.some(value => Math.abs(value) >= 10000) ? first : [0, 0, 0];
    }
    return [0, 0, 0];
  }, [props.autoFitReference ?? props.binaryPoints ?? props.frame?.points]);
  const previousOrigin = useRef<Vec3>([0, 0, 0]);
  const colorMode = props.colorMode ?? (props.semantic ? 'semantic' : 'height');
  const pointLabelCount = useMemo(() => Object.keys(props.annotations.point_labels).length, [props.annotations.point_labels]);
  const pointCount = props.binaryPoints ? props.binaryPoints.length / 4 : props.frame?.points.length ?? 0;
  const visibleIndices = useMemo<number[] | null>(() => {
    if (!props.binaryPoints) return visiblePointIndices(props.frame?.points ?? [], props.groundOnly, props.groundZ ?? -1.7, props.groundBand);
    if (!props.groundOnly) return null;
    const result: number[] = [];
    const groundZ = props.groundZ ?? -1.7;
    const band = Math.max(0, props.groundBand ?? .35);
    for (let offset = 0, index = 0; offset < props.binaryPoints.length; offset += 4, index++) {
      const x = props.binaryPoints[offset], y = props.binaryPoints[offset + 1], z = props.binaryPoints[offset + 2];
      if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z) && Math.abs(z - groundZ) <= band) result.push(index);
    }
    return result;
  }, [props.binaryPoints, props.frame?.points, props.groundOnly, props.groundZ, props.groundBand]);
  const visibleCount = visibleIndices?.length ?? pointCount;

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
    const trajectoryGroup = new THREE.Group();
    const calibrationGroup = new THREE.Group();
    scene.add(pointsGroup, annotationsGroup, overlaysGroup, trajectoryGroup, calibrationGroup);
    runtime.current = { renderer, scene, camera, controls, pointsGroup, annotationsGroup, overlaysGroup, trajectoryGroup, calibrationGroup, cloud: null, grid };
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
    const delta = new THREE.Vector3(...previousOrigin.current).sub(new THREE.Vector3(...origin));
    rt.camera.position.add(delta); rt.controls.target.add(delta);
    previousOrigin.current = origin;
  }, [ready, origin]);

  useEffect(() => {
    const rt = runtime.current;
    if (!rt) return;
    disposeObject(rt.pointsGroup);
    rt.pointsGroup.clear();
    rt.cloud = null;
    const points = props.frame?.points ?? [];
    const binary = props.binaryPoints;
    if (!pointCount) return;
    const positions = new Float32Array(visibleCount * 3);
    const colors = new Float32Array(visibleCount * 3);
    const color = new THREE.Color();
    let range = colorMode === 'intensity' && !binary ? intensityRange(points, visibleIndices ?? [])
      : colorMode === 'intensity' ? props.intensityRangeHint ?? null : null;
    if (colorMode === 'intensity' && binary && !range) {
      let minimum = Infinity, maximum = -Infinity;
      const samples: number[] = [];
      const sampleStep = Math.max(1, Math.floor(visibleCount / 8192));
      for (let displayIndex = 0; displayIndex < visibleCount; displayIndex++) {
        const sourceIndex = visibleIndices?.[displayIndex] ?? displayIndex;
        const value = binary[sourceIndex * 4 + 3];
        if (!Number.isFinite(value)) continue;
        minimum = Math.min(minimum, value); maximum = Math.max(maximum, value);
        if (displayIndex % sampleStep === 0) samples.push(value);
      }
      if (Number.isFinite(minimum) && maximum - minimum > Math.max(1e-9, Math.abs(maximum) * 1e-7)) {
        samples.sort((a, b) => a - b);
        const low = samples[Math.floor((samples.length - 1) * .02)] ?? minimum;
        const high = samples[Math.floor((samples.length - 1) * .995)] ?? maximum;
        range = high > low ? [low, high] : [minimum, maximum];
      }
    }
    for (let displayIndex = 0; displayIndex < visibleCount; displayIndex++) {
      const sourceIndex = visibleIndices?.[displayIndex] ?? displayIndex;
      const offset = sourceIndex * 4;
      const row = binary ? undefined : points[sourceIndex];
      const x = binary ? binary[offset] : row![0];
      const y = binary ? binary[offset + 1] : row![1];
      const z = binary ? binary[offset + 2] : row![2];
      const intensity = binary ? binary[offset + 3] : row![3];
      const target = displayIndex * 3;
      positions[target] = x - origin[0]; positions[target + 1] = y - origin[1]; positions[target + 2] = z - origin[2];
      const label = props.annotations.point_labels[String(sourceIndex)];
      if (colorMode === 'semantic' && label) color.set(labelColor(label));
      else if (colorMode === 'semantic') color.set('#748a9e');
      else if (colorMode === 'intensity') {
        const brightness = intensityBrightness(intensity, range);
        color.setRGB(brightness, brightness, brightness, THREE.SRGBColorSpace);
      }
      else {
        const height = THREE.MathUtils.clamp((z + 2.2) / 7, 0, 1);
        color.setHSL(0.55 - height * 0.5, 0.64, 0.46 + height * 0.17);
      }
      colors[target] = color.r; colors[target + 1] = color.g; colors[target + 2] = color.b;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.computeBoundingSphere();
    const cloud = new THREE.Points(geometry, new THREE.PointsMaterial({ size: colorMode === 'intensity' ? 2.5 : .095, vertexColors: true, sizeAttenuation: colorMode !== 'intensity' }));
    cloud.userData.sourceIndices = visibleIndices;
    rt.pointsGroup.add(cloud);
    rt.cloud = cloud;
  }, [ready, props.frame?.points, props.binaryPoints, props.intensityRangeHint, props.annotations.point_labels, colorMode, visibleIndices, visibleCount, pointCount, origin]);

  const fittedScene = useRef<number[][] | PointBuffer | null>(null);
  const fittedSpace = useRef<boolean | null>(null);
  useEffect(() => {
    const rt = runtime.current;
    const points = props.frame?.points ?? [];
    const binary = props.binaryPoints;
    if (!rt || !pointCount) return;
    const spaceChanged = fittedSpace.current !== Boolean(props.sceneMode);
    // Recenter once when switching world/LiDAR coordinates; subsequent source
    // frames keep the operator's view, just as live annotation edits do.
    if (!props.sceneMode && !spaceChanged) { fittedScene.current = null; return; }
    const fitReference = props.autoFitReference ?? binary ?? points;
    if (!spaceChanged && fittedScene.current === fitReference) return;
    fittedSpace.current = Boolean(props.sceneMode);
    fittedScene.current = fitReference ?? points;
    const bounds = new THREE.Box3();
    if (binary && props.cloudBounds) {
      bounds.expandByPoint(new THREE.Vector3(...toRenderPoint(props.cloudBounds.min, origin)));
      bounds.expandByPoint(new THREE.Vector3(...toRenderPoint(props.cloudBounds.max, origin)));
    } else if (binary) {
      for (let offset = 0; offset < binary.length; offset += 4) {
        bounds.expandByPoint(new THREE.Vector3(binary[offset] - origin[0], binary[offset + 1] - origin[1], binary[offset + 2] - origin[2]));
      }
    } else points.forEach(point => bounds.expandByPoint(new THREE.Vector3(...toRenderPoint(point, origin))));
    const center = bounds.getCenter(new THREE.Vector3());
    const radius = Math.max(bounds.getSize(new THREE.Vector3()).length() / 2, 12);
    rt.controls.target.copy(center);
    if (props.view === 'top') rt.camera.position.copy(center).add(new THREE.Vector3(-0.01, 0, radius * 2.4));
    else rt.camera.position.copy(center).add(new THREE.Vector3(-1, -1, 0.85).normalize().multiplyScalar(radius * 2.5));
    rt.controls.maxDistance = Math.max(radius * 20, 800);
    rt.camera.far = Math.max(radius * 40, 4000);
    rt.camera.updateProjectionMatrix();
    rt.controls.update();
  }, [ready, props.sceneMode, props.frame?.points, props.binaryPoints, props.cloudBounds, props.autoFitReference, props.view, pointCount, origin]);

  const followState = useRef<{reference: number[][] | PointBuffer | undefined; position: THREE.Vector3} | null>(null);
  useEffect(() => {
    const rt = runtime.current;
    if (!rt || !props.followPosition) {followState.current = null; return;}
    const position = new THREE.Vector3(...props.followPosition);
    const previous = followState.current;
    if (previous && previous.reference === props.autoFitReference) {
      const delta = position.clone().sub(previous.position);
      rt.camera.position.add(delta); rt.controls.target.add(delta);
    }
    followState.current = {reference: props.autoFitReference, position};
  }, [ready, props.followPosition?.[0], props.followPosition?.[1], props.followPosition?.[2], props.autoFitReference]);

  useEffect(() => {
    const rt = runtime.current;
    if (!rt) return;
    disposeObject(rt.trajectoryGroup);
    rt.trajectoryGroup.clear();
    if (!props.sceneMode || !props.trajectory?.length) return;
    const positions = props.trajectory.map(frame => new THREE.Vector3(...toRenderPoint(frame.position, origin)));
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(positions),
      new THREE.LineBasicMaterial({color: '#7fa8ef', transparent: true, opacity: 0.75, depthTest: false}));
    rt.trajectoryGroup.add(line);
    const trajectory = new THREE.Points(new THREE.BufferGeometry().setFromPoints(positions),
      new THREE.PointsMaterial({color: '#8ba8d6', size: 0.3, depthTest: false}));
    rt.trajectoryGroup.add(trajectory);
    const active = props.trajectory.find(frame => frame.frame_id === (props.activeFrameId ?? props.frame?.id));
    if (active) {
      const marker = new THREE.Mesh(new THREE.SphereGeometry(0.42, 12, 8), new THREE.MeshBasicMaterial({color: '#fff1a0', depthTest: false}));
      marker.position.fromArray(toRenderPoint(active.position, origin));
      rt.trajectoryGroup.add(marker);
      const label = makeLabel(`当前时刻 ${active.frame_id}`, '#fff1a0');
      label.position.fromArray(toRenderPoint(active.position, origin)).add(new THREE.Vector3(0, 0, 1.6));
      rt.trajectoryGroup.add(label);
    }
  }, [ready, props.sceneMode, props.trajectory, props.activeFrameId, props.frame?.id, origin]);

  useEffect(() => {
    const rt = runtime.current;
    if (!rt) return;
    disposeObject(rt.calibrationGroup);
    rt.calibrationGroup.clear();
    for (const anchor of props.calibrationPoints ?? []) {
      const color = anchor.enabled === false ? '#8292a0' : anchor.selected ? '#ffffff' : '#ffcf70';
      const point = new THREE.Vector3(...toRenderPoint(anchor.point, origin));
      rt.calibrationGroup.add(new THREE.Points(new THREE.BufferGeometry().setFromPoints([point]),
        new THREE.PointsMaterial({color, size: anchor.selected ? 13 : 10, sizeAttenuation: false, depthTest: false})));
      const label = makeLabel(anchor.label, color);
      label.scale.set(1.8, .65, 1); label.position.copy(point).add(new THREE.Vector3(0, 0, .35));
      rt.calibrationGroup.add(label);
    }
  }, [ready, props.calibrationPoints, origin]);

  useEffect(() => {
    const rt = runtime.current;
    if (!rt) return;
    disposeObject(rt.annotationsGroup);
    rt.annotationsGroup.clear();
    if (props.showBoxes !== false) props.annotations.boxes.forEach((box) => rt.annotationsGroup.add(boxMesh(box, props.selectedId === box.id, origin)));
    if (props.showVectors !== false) {
      const vectors = [...props.annotations.vectors];
      if (props.draftVector?.length) vectors.push({ id: '__draft__', label: 'lane', points: props.draftVector, closed: false });
      vectors.forEach((vector) => {
        if (!vector.points.length) return;
        const coordinates = vector.points.map((p) => new THREE.Vector3(...toRenderPoint(p, origin)));
        if (vector.closed && coordinates.length > 2) coordinates.push(coordinates[0].clone());
        const color = vector.id === props.selectedId ? '#ffffff' : labelColor(vector.label);
        const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(coordinates), new THREE.LineBasicMaterial({ color, depthTest: false }));
        line.userData.vectorId = vector.id;
        rt.annotationsGroup.add(line);
        const dots = new THREE.Points(new THREE.BufferGeometry().setFromPoints(coordinates), new THREE.PointsMaterial({ color, size: vector.id === '__draft__' ? 8 : vector.id === props.selectedId ? 7 : 4, sizeAttenuation: false, depthTest: false }));
        dots.userData.vectorId = vector.id === '__draft__' ? undefined : vector.id;
        rt.annotationsGroup.add(dots);
      });
    }
  }, [ready, props.annotations.boxes, props.annotations.vectors, props.selectedId, props.draftVector, props.showBoxes, props.showVectors, origin]);

  useEffect(() => {
    const rt = runtime.current;
    if (!rt) return;
    disposeObject(rt.overlaysGroup);
    rt.overlaysGroup.clear();
    if (props.overlayPoints?.length) {
      const coordinates = props.overlayPoints.map((p) => new THREE.Vector3(...toRenderPoint(p, origin)));
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
      const geometry = new THREE.BufferGeometry().setFromPoints(freeVoxels.map((voxel) => new THREE.Vector3(...toRenderPoint(voxel.center, origin))));
      const freeSpace = new THREE.Points(geometry, new THREE.PointsMaterial({
        color: '#6e9cb5', size: (props.voxelSize ?? 0.5) * 0.2,
        transparent: true, opacity: 0.26, depthWrite: false, sizeAttenuation: true,
      }));
      freeSpace.name = 'observed-free-space';
      rt.overlaysGroup.add(freeSpace);
    }
    if (voxels?.length) {
      const dimensions = props.voxelDimensions ?? [props.voxelSize ?? 0.5, props.voxelSize ?? 0.5, props.voxelSize ?? 0.5];
      const geometry = new THREE.BoxGeometry(...dimensions.map(value => value * 0.92) as Vec3);
      const material = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.66 });
      const mesh = new THREE.InstancedMesh(geometry, material, voxels.length);
      const transform = new THREE.Matrix4();
      const color = new THREE.Color();
      voxels.forEach((voxel, index) => {
        transform.makeTranslation(...toRenderPoint(voxel.center, origin));
        mesh.setMatrixAt(index, transform);
        color.set(labelColor(voxel.label));
        mesh.setColorAt(index, color);
      });
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
      rt.overlaysGroup.add(mesh);
    }
  }, [ready, props.overlayPoints, props.overlayPointLabels, props.semantic, props.voxels, props.voxelSize, props.showFree, props.voxelDimensions, origin]);

  useEffect(() => {
    if (runtime.current) runtime.current.grid.position.z = (props.groundZ ?? -1.7) - origin[2] - 0.05;
  }, [ready, props.groundZ, origin]);

  useEffect(() => {
    const rt = runtime.current;
    if (!rt) return;
    const target = rt.controls.target;
    const distance = props.sceneMode ? Math.max(rt.camera.position.distanceTo(target), 20) : 68;
    if (props.view === 'top') {
      rt.camera.position.set(target.x - 0.01, target.y, target.z + distance);
      rt.controls.enableRotate = false;
    } else {
      if (props.sceneMode) rt.camera.position.copy(target).add(new THREE.Vector3(-41, -36, 35).normalize().multiplyScalar(distance));
      else rt.camera.position.set(target.x - 41, target.y - 36, target.z + 35);
      rt.controls.enableRotate = true;
    }
    rt.controls.update();
  }, [ready, props.view, props.sceneMode]);

  useEffect(() => {
    const rt = runtime.current;
    const current = latest.current;
    if (!rt || !current.focusSelection) return;
    const box = current.annotations.boxes.find(item => item.id === current.selectedId);
    if (!box) return;
    const damping = rt.controls.enableDamping;
    rt.controls.enableDamping = false;
    rt.controls.update();
    const center = new THREE.Vector3(...toRenderPoint(box.center, origin));
    const distance = Math.max(Math.max(...box.size) * 3, 8);
    const direction = rt.camera.position.clone().sub(rt.controls.target).normalize();
    if (direction.lengthSq() < 1e-6) direction.set(-1, -1, 1).normalize();
    rt.controls.target.copy(center);
    if (current.view === 'top') rt.camera.position.copy(center).add(new THREE.Vector3(-.001, 0, distance));
    else rt.camera.position.copy(center).addScaledVector(direction, distance);
    rt.controls.update();
    rt.controls.enableDamping = damping;
    // This effect intentionally observes only the explicit focus request. Box
    // changes during pointer drags must not move the camera under the cursor.
  }, [ready, props.focusSelection]);

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
    if (current.calibrationPicking) {
      // Use the measured point itself, not the closest position on the mouse ray
      // or a ground-plane intersection. Neither is a valid 3D calibration anchor.
      const hit = rt.cloud ? raycaster.intersectObject(rt.cloud)[0] : undefined;
      if (hit?.index === undefined) {setCursor('未选中实测点，请放大点云后重试'); return;}
      const indices = rt.cloud!.userData.sourceIndices as number[] | null | undefined;
      const index = indices?.[hit.index] ?? hit.index;
      const offset = index * 4;
      const point = current.binaryPoints
        ? [current.binaryPoints[offset], current.binaryPoints[offset + 1], current.binaryPoints[offset + 2]] as Vec3
        : current.frame?.points[index]?.slice(0, 3) as Vec3 | undefined;
      if (point?.every(Number.isFinite)) current.onCalibrationPoint?.(point);
      return;
    }
    if (current.tool === 'select' || !current.tool) {
      const hit = raycaster.intersectObjects(rt.annotationsGroup.children, true).find((item) => item.object.userData.boxId || item.object.userData.vectorId);
      current.onSelect?.(hit?.object.userData.boxId ?? hit?.object.userData.vectorId ?? null);
      return;
    }
    const cloudHit = rt.cloud ? raycaster.intersectObject(rt.cloud)[0] : undefined;
    const ground = raycaster.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 0, 1), -((current.groundZ ?? -1.7) - origin[2])), new THREE.Vector3());
    const worldGround = ground?.clone().add(new THREE.Vector3(...origin));
    let point = cloudHit?.point.clone().add(new THREE.Vector3(...origin)) ?? worldGround;
    if (current.tool === 'vector' && current.drawOnGround && worldGround) {
      if (current.binaryPoints) {
        const radiusSquared = .36;
        let nearest = radiusSquared, height = worldGround.z;
        for (let offset = 0; offset < current.binaryPoints.length; offset += 4) {
          const x = current.binaryPoints[offset], y = current.binaryPoints[offset + 1], z = current.binaryPoints[offset + 2];
          if (Math.abs(z - worldGround.z) > (current.groundBand ?? .35)) continue;
          const distance = (x - worldGround.x) ** 2 + (y - worldGround.y) ** 2;
          if (distance < nearest) {nearest = distance; height = z;}
        }
        point = new THREE.Vector3(worldGround.x, worldGround.y, height);
      } else point = new THREE.Vector3(...groundDrawingPoint(current.frame?.points ?? [], [worldGround.x, worldGround.y, worldGround.z], current.groundBand));
    }
    if (!point) return;
    const coordinates: Vec3 = [point.x, point.y, point.z];
    if (current.tool === 'box') current.onCreateBox?.(coordinates);
    else if (current.tool === 'vector') current.onAddVectorPoint?.(coordinates);
    else if (current.tool === 'brush' && !current.sceneMode) {
      const radius = current.brushRadius ?? 1.5;
      const indices: number[] = [];
      const sourceIndices = rt.cloud?.userData.sourceIndices as number[] | null | undefined;
      const candidateCount = sourceIndices?.length ?? (current.binaryPoints ? current.binaryPoints.length / 4 : current.frame?.points.length ?? 0);
      for (let displayIndex = 0; displayIndex < candidateCount; displayIndex++) {
        const index = sourceIndices?.[displayIndex] ?? displayIndex;
        const offset = index * 4;
        const p = current.binaryPoints
          ? [current.binaryPoints[offset], current.binaryPoints[offset + 1], current.binaryPoints[offset + 2]]
          : current.frame!.points[index];
        if (Math.hypot(p[0] - point.x, p[1] - point.y, p[2] - point.z) <= radius) indices.push(index);
      }
      if (indices.length) current.onPaint?.(indices);
    }
  };

  return <div
    ref={container}
    className={props.className}
    data-testid="point-cloud-view"
    data-coordinate-frame={props.sceneMode ? 'world' : 'lidar'}
    data-cloud-point-count={pointCount}
    data-visible-point-count={visibleCount}
    data-color-mode={colorMode}
    data-render-origin={origin.join(',')}
    data-active-frame-id={props.activeFrameId ?? props.frame?.id ?? ''}
    data-annotation-box-count={props.showBoxes === false ? 0 : props.annotations.boxes.length}
    data-annotation-vector-count={props.showVectors === false ? 0 : props.annotations.vectors.length}
    data-vector-editable={props.vectorEditable ? 'true' : 'false'}
    data-point-label-count={pointLabelCount}
    data-calibration-point-count={props.calibrationPoints?.length ?? 0}
    data-calibration-picking={props.calibrationPicking ? 'true' : 'false'}
    style={{ position: 'relative', width: '100%', height: '100%', minHeight: 240, overflow: 'hidden', cursor: props.calibrationPicking || props.tool && props.tool !== 'select' ? 'crosshair' : 'grab' }}
    onPointerDown={(event) => { pointerStart.current = [event.clientX, event.clientY]; }}
    onClick={onClick}
    onPointerMove={(event) => {
      if (!runtime.current || event.buttons) return;
      const ray = rayAt(event.clientX, event.clientY);
      const point = ray?.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 0, 1), -((latest.current.groundZ ?? -1.7) - origin[2])), new THREE.Vector3());
      if (point) point.add(new THREE.Vector3(...origin));
      if (point) setCursor(`X ${point.x.toFixed(1)} · Y ${point.y.toFixed(1)} m`);
    }}
    onPointerLeave={() => setCursor('')}
  >
    {ready && runtime.current && !props.calibrationPicking && props.editable !== false && props.showBoxes !== false &&
      (props.tool === 'select' || !props.tool) && props.onUpdateBox && props.annotations.boxes.find(box => box.id === props.selectedId) &&
      <BoxTransformGizmo key={`${props.selectedId}/${props.activeFrameId ?? props.frame?.id ?? ''}`}
        box={props.annotations.boxes.find(box => box.id === props.selectedId)!}
        renderOrigin={origin} mode={props.transformMode ?? 'scale'} camera={runtime.current.camera} canvas={runtime.current.renderer.domElement}
        controls={runtime.current.controls} onUpdate={props.onUpdateBox} onStart={props.onTransformStart} onEnd={props.onTransformEnd}/>}
    {ready && runtime.current && !props.calibrationPicking && props.vectorEditable && props.showVectors !== false &&
      (props.tool === 'select' || !props.tool) && props.onUpdateVectorPoint && props.annotations.vectors.find(vector => vector.id === props.selectedId) &&
      <VectorVertexGizmo key={`${props.selectedId}/${props.activeFrameId ?? props.frame?.id ?? ''}`}
        vector={props.annotations.vectors.find(vector => vector.id === props.selectedId)!} renderOrigin={origin}
        camera={runtime.current.camera} canvas={runtime.current.renderer.domElement} controls={runtime.current.controls}
        onUpdate={props.onUpdateVectorPoint} onStart={props.onTransformStart} onEnd={props.onTransformEnd}/>}
    {error && <div role="alert" style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', padding: 24, color: '#f3c2b2', background: '#0b131f', textAlign: 'center' }}>{error}</div>}
    <div style={{ position: 'absolute', right: 14, bottom: 12, pointerEvents: 'none', color: '#8294a9', font: '10px ui-monospace, monospace', display: 'flex', gap: 16 }}>
      {cursor && <span>{cursor}</span>}<span>X 前 · Y 左 · Z 上</span>
    </div>
    {!props.frame && !error && <div style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', color: '#8294a9', pointerEvents: 'none' }}>请选择一个 clip</div>}
  </div>;
}

export default PointCloudView;
