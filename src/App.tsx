import {useCallback, useEffect, useMemo, useRef, useState} from 'react';
import {CameraView} from './components/CameraView';
import {PointCloudView} from './components/PointCloudView';
import {CalibrationEditor, type CalibrationDocument} from './components/CalibrationEditor';
import {ProjectionCalibrationEditor, type ProjectionCalibrationDocument, type ProjectionCorrespondence, type ProjectionPickMode} from './components/ProjectionCalibrationEditor';
import {AutomaticCalibrationPanel} from './components/AutomaticCalibrationPanel';
import {LaneMergePanel, type LaneMergeResult} from './components/LaneMergePanel';
import {LatrBatchPanel, type BatchProposal} from './components/LatrBatchPanel';
import {applyCalibration, type Matrix4} from './geometry/calibration';
import {projectFramePoints, projectMapPreview} from './geometry/liveProjection';
import {Icon, type IconName} from './components/Icon';
import {request, requestPointNpy, jsonRequest, download} from './api';
import {annotationId} from './ids';
import {loadPlaybackFrame, type PlaybackFrame} from './playback';
import {useClipPlayback} from './useClipPlayback';
import {reviewWorldAnnotations} from './reviewAnnotations';
import {EMPTY_ANNOTATIONS as EMPTY, IDENTITY, evaluateScene, evaluateTrack, inverseRigid, sceneInLidar, transformBox, transformPoint, upsertKeyframe, worldFromLidar} from './scene';
import type {Annotations, Box, Clip, ClipSummary, Frame, ModelConnection, ModelInfo, SceneAnnotations, SceneMap, SceneTrack, Vec3, Vector, Voxel} from './types';

const tasks: {id: string; name: string; short: string; icon: IconName; subtitle: string}[] = [
  {id: 'detection', name: '3D 目标检测', short: '3D', icon: 'cube', subtitle: '世界坐标轨迹 · 拖动编辑 · 时序关键帧'},
  {id: 'fusion', name: '4D 点云地图', short: '4D', icon: 'layers', subtitle: '整个 clip 重建为统一场景'},
  {id: 'lane', name: '车道线标注', short: 'LANE', icon: 'lane', subtitle: '全场景共享的三维车道线'},
  {id: 'vector', name: '矢量地图', short: 'MAP', icon: 'map', subtitle: '世界坐标矢量与道路面要素'},
  {id: 'image', name: '图像语义分割', short: '2D', icon: 'image', subtitle: '原始图像语义区域与 SAM2 辅助'},
  {id: 'semantic', name: '点云语义分割', short: 'SEG', icon: 'points', subtitle: '源点云逐点标签 · 模型辅助'},
  {id: 'occupancy', name: 'OCC 真值转换', short: 'OCC', icon: 'grid', subtitle: '观测 OCC 与 OpenOccupancy 模型结果'},
];
const objectClasses = [{id: 'vehicle', name: '车辆', color: '#37ddbf'}, {id: 'car', name: '汽车', color: '#37ddbf'}, {id: 'truck', name: '卡车', color: '#fbad5c'}, {id: 'pedestrian', name: '行人', color: '#f476aa'}, {id: 'cyclist', name: '骑行者', color: '#aa96ff'}, {id: 'unknown', name: '未知目标', color: '#ff3b5c'}];
const semanticClasses = [{id: 'road', name: '路面', color: '#65acff'}, {id: 'vehicle', name: '车辆', color: '#37ddbf'}, {id: 'building', name: '建筑', color: '#c39dff'}, {id: 'vegetation', name: '植被', color: '#a3d465'}, {id: 'pedestrian', name: '行人', color: '#f476aa'}, {id: 'unlabeled', name: '未分类', color: '#98a8b8'}];
const vectorClasses = [{id: 'lane', name: '车道线', color: '#f7d16a'}, {id: 'lane_solid', name: '实线', color: '#f7d16a'}, {id: 'lane_dashed', name: '虚线', color: '#f7d16a'}, {id: 'road_boundary', name: '道路边界', color: '#37ddbf'}, {id: 'stop_line', name: '停止线', color: '#f476aa'}, {id: 'crosswalk', name: '人行横道', color: '#aa96ff'}, {id: 'direction_arrow', name: '导向箭头轮廓', color: '#aa96ff'}];
const modelNames: Record<string, string> = {sam2: 'SAM2 · 图像分割', latr: 'LATR · 三维车道线', mmdetection3d: 'MMDetection3D · 三维检测', openoccupancy: 'OpenOccupancy · 语义占用'};
type Tool = 'select' | 'box' | 'vector' | 'brush';
type JobView = {id: string; kind: string; status: string; error?: string; result?: Record<string, unknown>};
type Proposal = {annotations: Annotations; frameId: string; clipId: string; revision: number; sceneRevision: number; sceneMode: boolean; model: string; weightVariant?: string; occupancy?: {voxels: Voxel[]; voxel_size: number; voxel_dimensions?: Vec3}; warnings?: string[]};
type LanePreview = {scopeKey: string; source: Vector[]; result: LaneMergeResult};
type LaneMergeUndo = {scopeKey: string; before: Vector[]; after: Vector[]};
const NO_VECTORS: Vector[] = [];
const uid = annotationId;
const errorText = (error: unknown) => error instanceof Error ? error.message : String(error);
const modelStatus = (model?: ModelInfo) => model?.available ? '推理就绪' : model?.status_detail || model?.reason || (model?.configured ? '服务未就绪，请检查部署' : '尚未配置推理服务');

async function loadCachedSceneMap(id: string): Promise<SceneMap | null> {
  const encoded = encodeURIComponent(id);
  const response = await fetch(`/api/clips/${encoded}/scene/map?include_points=false`);
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`读取场景地图失败 HTTP ${response.status}`);
  const metadata = await response.json() as Omit<SceneMap, 'points'>;
  const pointBuffer = await requestPointNpy(`/clips/${encoded}/scene/map.npy`);
  if (pointBuffer.length / 4 !== metadata.point_count) {
    throw new Error(`场景点云数量不一致：元数据 ${metadata.point_count}，二进制 ${pointBuffer.length / 4}`);
  }
  return {...metadata, points: [], point_buffer: pointBuffer};
}

export default function App() {
  const [clips, setClips] = useState<ClipSummary[]>([]);
  const [clip, setClip] = useState<Clip | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [annotations, setAnnotations] = useState<Annotations>(EMPTY);
  const [scene, setScene] = useState<SceneAnnotations | null>(null);
  const [sceneMap, setSceneMap] = useState<SceneMap | null>(null);
  const [registeredMatrices, setRegisteredMatrices] = useState<Record<string, number[][]>>({});
  const [calibrationOpen, setCalibrationOpen] = useState(false);
  const [automaticCalibrationOpen, setAutomaticCalibrationOpen] = useState(false);
  const [calibrationDirty, setCalibrationDirty] = useState(false);
  const [calibrationPreview, setCalibrationPreview] = useState<CalibrationDocument | null>(null);
  const calibrationBase = useRef<CalibrationDocument | null>(null);
  const calibrationClipBase = useRef<Clip | null>(null);
  const [projectionOpen, setProjectionOpen] = useState(false);
  const [projectionDirty, setProjectionDirty] = useState(false);
  const [projectionBusy, setProjectionBusy] = useState(false);
  const [projectionPickMode, setProjectionPickMode] = useState<ProjectionPickMode>(null);
  const [projectionPickedPoint, setProjectionPickedPoint] = useState<Vec3 | null>(null);
  const [projectionPickedPixel, setProjectionPickedPixel] = useState<[number, number] | null>(null);
  const [projectionPreview, setProjectionPreview] = useState<{clipId: string; frameId: string; cameraId: string; matrix: Matrix4} | null>(null);
  const [projectionAnchors, setProjectionAnchors] = useState<{anchors: ProjectionCorrespondence[]; pending: Vec3 | null; selected: string | null}>({anchors: [], pending: null, selected: null});
  const [sceneMode, setSceneMode] = useState(true);
  const [frameDirty, setFrameDirty] = useState(false);
  const [sceneDirty, setSceneDirty] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [task, setTask] = useState('detection');
  const [tool, setTool] = useState<Tool>('select');
  const [view, setView] = useState<'perspective' | 'top'>('perspective');
  const [transformMode, setTransformMode] = useState<'translate' | 'scale' | 'rotate'>('scale');
  const [focusSelection, setFocusSelection] = useState(0);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [classId, setClassId] = useState('vehicle');
  const [draftVector, setDraftVector] = useState<Vec3[]>([]);
  const [draftPolygon, setDraftPolygon] = useState<number[][]>([]);
  const [activeCamera, setActiveCamera] = useState('');
  const [expandedCamera, setExpandedCamera] = useState<string | null>(null);
  const [annotationProjection, setAnnotationProjection] = useState(true);
  const [rawProjection, setRawProjection] = useState(false);
  const [showBoxes, setShowBoxes] = useState(true);
  const [showVectors, setShowVectors] = useState(true);
  const [playing, setPlaying] = useState(false);
  const [reviewing, setReviewing] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [playbackLoop, setPlaybackLoop] = useState(false);
  const [playbackCloud, setPlaybackCloud] = useState<'scene' | 'frame'>('frame');
  const [playbackSession, setPlaybackSession] = useState(0);
  const [buffering, setBuffering] = useState(false);
  const [preparedFrame, setPreparedFrame] = useState<PlaybackFrame | null>(null);
  const [reviewStarting, setReviewStarting] = useState(false);
  const [playbackEnded, setPlaybackEnded] = useState(false);
  const reviewSpace = useRef(true);
  const reviewColor = useRef<'intensity' | 'height' | 'semantic'>('intensity');
  const reviewGroundOnly = useRef(false);
  const reviewFitReference = useRef<number[][] | null>(null);
  const reviewInitialIndex = useRef<number | null>(null);
  const seekController = useRef<AbortController | null>(null);
  const reviewState = useRef(false);
  reviewState.current = reviewing;
  const [brushRadius, setBrushRadius] = useState(1.5);
  const [voxelSize, setVoxelSize] = useState(0.3);
  const [preserveDetail, setPreserveDetail] = useState(true);
  const [fusionMode, setFusionMode] = useState<'full' | 'detail' | 'surface' | 'centroid'>('full');
  const [detailVoxelSize, setDetailVoxelSize] = useState(0.05);
  const [pointColorMode, setPointColorMode] = useState<'intensity' | 'height' | 'semantic'>('intensity');
  const [groundOnly, setGroundOnly] = useState(false);
  const [groundBand, setGroundBand] = useState(0.35);
  const [showFree, setShowFree] = useState(false);
  const [registration, setRegistration] = useState<'poses' | 'icp'>('poses');
  const [excludeDynamic, setExcludeDynamic] = useState(true);
  const [sceneGroundZ, setSceneGroundZ] = useState(0);
  const [frameGroundZ, setFrameGroundZ] = useState(-1.7);
  const [notice, setNotice] = useState<{message: string; error?: boolean} | null>(null);
  const [job, setJob] = useState<JobView | null>(null);
  const [reconstruction, setReconstruction] = useState<JobView | null>(null);
  const [occupancy, setOccupancy] = useState<{frameId: string; voxels: Voxel[]; size: number; dimensions?: Vec3} | null>(null);
  const [workerModel, setWorkerModel] = useState('sam2');
  const [workerLabel, setWorkerLabel] = useState('road');
  const [latrWeight, setLatrWeight] = useState<'once' | 'openlane'>('once');
  const [scoreThreshold, setScoreThreshold] = useState(0.3);
  const [modelList, setModelList] = useState<ModelInfo[] | null>(null);
  const [connections, setConnections] = useState<ModelConnection[]>([]);
  const [connectionBusy, setConnectionBusy] = useState(false);
  const [modelPromptMode, setModelPromptMode] = useState<'point' | 'box' | null>(null);
  const [negativePrompt, setNegativePrompt] = useState(false);
  const [modelPoints, setModelPoints] = useState<number[][]>([]);
  const [modelPointLabels, setModelPointLabels] = useState<number[]>([]);
  const [modelBox, setModelBox] = useState<number[] | undefined>();
  const [proposal, setProposal] = useState<Proposal | null>(null);
  const proposalRef = useRef<Proposal | null>(null);
  proposalRef.current = proposal;
  const [proposalApplying, setProposalApplying] = useState(false);
  const [autoMergeLatr, setAutoMergeLatr] = useState(true);
  const [latrBatchOpen, setLatrBatchOpen] = useState(false);
  const [laneMergeOpen, setLaneMergeOpen] = useState(false);
  const [lanePreview, setLanePreview] = useState<LanePreview | null>(null);
  const [laneMergeUndo, setLaneMergeUndo] = useState<LaneMergeUndo | null>(null);
  const [modal, setModal] = useState<'models' | 'help' | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const clipGeneration = useRef(0);
  const frameGeneration = useRef(0);
  const context = useRef('');
  const clipContext = useRef('');
  const frameDirtyRef = useRef(false);
  const sceneDirtyRef = useRef(false);
  const draftPending = useRef(false);
  const editGeneration = useRef(0);
  const runningJob = useRef(false);
  const runningReconstruction = useRef(false);
  const sourceFrame = clip?.frames[frameIndex] ?? null;
  const frame = useMemo(() => sourceFrame && projectionPreview && projectionPreview.clipId === clip?.id && projectionPreview.frameId === sourceFrame.id
    ? {...sourceFrame, projection_camera_from_lidar: {...sourceFrame.projection_camera_from_lidar, [projectionPreview.cameraId]: projectionPreview.matrix}} : sourceFrame,
  [sourceFrame, projectionPreview, clip?.id]);
  const dirty = frameDirty || sceneDirty;
  const busy = loading || saving || reviewStarting || proposalApplying;
  const jobBusy = job?.status === 'running' || job?.status === 'queued';
  const reconstructing = reconstruction?.status === 'running' || reconstruction?.status === 'queued';
  const editBlocked = reviewing || busy || jobBusy || projectionOpen || automaticCalibrationOpen || latrBatchOpen || !clip || !frame || (sceneMode && !scene);
  const currentKey = clip && frame ? `${clip.id}/${frame.id}` : '';
  const laneScopeKey = sceneMode ? `${clip?.id}/world` : currentKey;
  const laneVectors = sceneMode ? scene?.vectors ?? NO_VECTORS : annotations.vectors;
  const activeLanePreview = lanePreview?.scopeKey === laneScopeKey && lanePreview.source === laneVectors ? lanePreview : null;
  const geometryEditBlocked = editBlocked || Boolean(activeLanePreview);
  const laneScope = useRef({scopeKey: laneScopeKey, source: laneVectors});
  laneScope.current = {scopeKey: laneScopeKey, source: laneVectors};
  const previewLanes = useCallback((result: LaneMergeResult | null) => {
    setLanePreview(result ? {...laneScope.current, result} : null);
  }, []);
  context.current = currentKey;
  clipContext.current = clip?.id ?? '';
  frameDirtyRef.current = frameDirty;
  sceneDirtyRef.current = sceneDirty;
  draftPending.current = draftVector.length > 0 || draftPolygon.length > 0;

  const worldAnnotations = useMemo(() => evaluateScene(scene && activeLanePreview && sceneMode
    ? {...scene, vectors: activeLanePreview.result.vectors} : scene, frame, clip), [scene, frame, clip, activeLanePreview, sceneMode]);
  const frameAnnotations = useMemo(() => activeLanePreview && !sceneMode
    ? {...annotations, vectors: activeLanePreview.result.vectors} : annotations, [annotations, activeLanePreview, sceneMode]);
  // Registration belongs to the current clip, while map metadata can predate a
  // calibration change. Never let a cached map replace newer per-frame poses.
  const coordinateMap = useMemo(() => ({world_from_lidar: calibrationPreview?.world_from_lidar
    ?? {...sceneMap?.world_from_lidar, ...registeredMatrices}}), [calibrationPreview, sceneMap, registeredMatrices]);
  const suppliedMap = sceneMap?.source_map === 'supplied' || sceneMap?.method === 'supplied reconstructed local-world map';
  const calibrationLidarPreview = Boolean(calibrationPreview && calibrationBase.current &&
    JSON.stringify(calibrationPreview.lidar_to_ego) !== JSON.stringify(calibrationBase.current.lidar_to_ego));
  const usingSceneMap = Boolean(sceneMap && (!calibrationLidarPreview || suppliedMap) && (!reviewing || playbackCloud === 'scene'));
  const reviewAnnotations = useMemo(() => reviewing && frame ? reviewWorldAnnotations(worldAnnotations, annotations, frame, coordinateMap) : worldAnnotations,
    [reviewing, worldAnnotations, annotations, frame, coordinateMap]);
  const projections = useMemo(() => {
    if (!rawProjection || !clip || !frame) return {};
    if (!frame.points.length && sceneMap?.point_buffer) {
      const camera = clip.cameras.find(item => item.id === activeCamera);
      return camera ? {[camera.id]: projectMapPreview(frame, camera, sceneMap.point_buffer, inverseRigid(worldFromLidar(frame, coordinateMap)))} : {};
    }
    return projectFramePoints(frame, clip.cameras);
  }, [rawProjection, clip?.cameras, frame?.points, frame?.lidar_to_ego, frame?.projection_camera_from_lidar, frame?.annotation_camera_from_lidar, sceneMap?.point_buffer, activeCamera, coordinateMap]);
  const imageAnnotations = useMemo(() => {
    if (!sceneMode || !frame) return frameAnnotations;
    return {...sceneInLidar(reviewAnnotations, frame, coordinateMap), image_polygons: annotations.image_polygons, point_labels: annotations.point_labels};
  }, [sceneMode, reviewAnnotations, frame, coordinateMap, annotations, frameAnnotations]);
  const sceneFrame = useMemo<Frame | null>(() => {
    if (sceneMap && usingSceneMap) return {id: 'scene-world', timestamp: 0, ego_to_world: IDENTITY, lidar_to_ego: IDENTITY, points: [], point_count: sceneMap.point_count, images: {}};
    if (!frame) return null;
    const matrix = worldFromLidar(frame, coordinateMap);
    return {...frame, id: `world-${frame.id}`, ego_to_world: IDENTITY, lidar_to_ego: IDENTITY,
      points: frame.points.map(point => [...transformPoint(point, matrix), point[3] ?? 0])};
  }, [sceneMap, frame, coordinateMap, usingSceneMap]);
  const renderedAnnotations = sceneMode ? {...reviewAnnotations, point_labels: usingSceneMap ? sceneMap?.point_labels ?? {} : annotations.point_labels} : frameAnnotations;
  const projectionMarkers = useMemo(() => {
    if (!projectionOpen || !frame) return [];
    const matrix = sceneMode ? worldFromLidar(frame, coordinateMap) : IDENTITY;
    const markers = projectionAnchors.anchors.map((anchor, index) => ({id: anchor.id, point: transformPoint(anchor.point, matrix),
      label: String(index + 1), selected: projectionAnchors.selected === anchor.id, enabled: anchor.enabled}));
    if (projectionAnchors.pending) markers.push({id: '__pending__', point: transformPoint(projectionAnchors.pending, matrix), label: '+', selected: true, enabled: true});
    return markers;
  }, [projectionOpen, projectionAnchors, frame, coordinateMap, sceneMode]);
  const cameraProjectionAnchors = useMemo(() => projectionOpen ? projectionAnchors.anchors.map((anchor, index) => ({...anchor,
    label: String(index + 1), selected: projectionAnchors.selected === anchor.id})) : [], [projectionOpen, projectionAnchors]);
  const selectedBox = renderedAnnotations.boxes.find(b => b.id === selectedId);
  const selectedVector = renderedAnnotations.vectors.find(v => v.id === selectedId);
  const selectedPolygon = annotations.image_polygons.find(p => p.id === selectedId);
  const selectedTrack = sceneMode ? scene?.tracks.find(t => t.id === selectedId) : undefined;
  const currentIsKeyframe = selectedTrack?.keyframes.some(k => k.frame_id === frame?.id) ?? false;
  const taskInfo = tasks.find(t => t.id === task)!;
  const classes = ['lane', 'vector'].includes(task) ? vectorClasses : ['image', 'semantic', 'occupancy'].includes(task) ? semanticClasses : objectClasses;
  const selectedModel = modelList?.find(m => m.id === workerModel);
  const groundZ = sceneMode ? sceneGroundZ : frameGroundZ;
  const vectorTask = task === 'lane' || task === 'vector';
  const closedDraft = classId === 'crosswalk' || classId === 'direction_arrow';
  const displayedPointBuffer = ((sceneMode && usingSceneMap) || (!sceneMode && clip?.source === '4d-clip-import')) ? sceneMap?.point_buffer : undefined;
  const surfaceInfo = useMemo(() => {
    if (displayedPointBuffer) {
      if (!groundOnly) return {count: displayedPointBuffer.length / 4, contrast: sceneMap?.intensity_has_contrast ?? true};
      let min = Infinity, max = -Infinity, count = 0;
      for (let offset = 0; offset < displayedPointBuffer.length; offset += 4) {
        if (Math.abs(displayedPointBuffer[offset + 2] - groundZ) > groundBand) continue;
        count++;
        const intensity = displayedPointBuffer[offset + 3];
        if (Number.isFinite(intensity)) {min = Math.min(min, intensity); max = Math.max(max, intensity);}
      }
      return {count, contrast: Number.isFinite(min) && max - min > 1e-6};
    }
    const points = sceneMode ? sceneFrame?.points : frame?.points;
    let min = Infinity, max = -Infinity, count = 0;
    for (const p of points ?? []) {
      if (groundOnly && Math.abs(p[2] - groundZ) > groundBand) continue;
      count++;
      if (p.length > 3 && Number.isFinite(p[3])) {min = Math.min(min, p[3]); max = Math.max(max, p[3]);}
    }
    return {count, contrast: Number.isFinite(min) && max - min > 1e-6};
  }, [displayedPointBuffer, sceneMap?.intensity_has_contrast, sceneMode, sceneFrame?.points, frame?.points, groundOnly, groundZ, groundBand]);

  const flash = useCallback((message: string, error = false) => setNotice({message, error}), []);
  const projectionScope = useRef({clipId: '', frameId: '', cameraId: ''});
  projectionScope.current = {clipId: clip?.id ?? '', frameId: sourceFrame?.id ?? '', cameraId: activeCamera};
  const previewProjection = useCallback((matrix: Matrix4 | null) => {
    setProjectionPreview(matrix ? {...projectionScope.current, matrix} : null);
  }, []);
  const consumeProjectionPick = useCallback(() => {setProjectionPickedPoint(null); setProjectionPickedPixel(null);}, []);
  const changeProjectionAnchors = useCallback((anchors: ProjectionCorrespondence[], pending: Vec3 | null, selected: string | null) => {
    setProjectionAnchors({anchors, pending, selected});
  }, []);
  const savedProjection = useCallback(async (document: ProjectionCalibrationDocument) => {
    setClip(current => current?.id === document.clip_id ? {...current, frames: current.frames.map(item => {
      if (item.id !== document.frame_id) return item;
      const matrices = {...item.projection_camera_from_lidar};
      const matrix = document.projection_matrix ?? document.saved_matrix;
      if (matrix) matrices[document.camera_id] = matrix;
      else delete matrices[document.camera_id];
      return {...item, projection_camera_from_lidar: matrices};
    })} : current);
    setProjectionDirty(false);
    await refreshProjectionParameters(document.clip_id);
    flash(document.clip_scope_active ? '当前相机的整段 clip 投影已更新，已有逐帧精调保留。'
      : '当前帧相机投影已保存，点云与三维标注共用该修正。');
  }, [flash]);
  useEffect(() => {if (!notice) return; const timer = setTimeout(() => setNotice(null), notice.error ? 10000 : 5000); return () => clearTimeout(timer);}, [notice]);
  const refreshModels = useCallback(async () => {
    const [catalog, config] = await Promise.all([request<{models: ModelInfo[]}>('/models'), request<{connections: ModelConnection[]}>('/model-connections')]);
    setModelList(catalog.models); setConnections(config.connections);
  }, []);
  const refreshClips = useCallback(async () => {const rows = await request<ClipSummary[]>('/clips'); setClips(rows); return rows;}, []);

  const reconstruct = useCallback(async (id: string, generation: number, method: 'poses' | 'icp', size: number, exclude: boolean, detail = true, detailSize = 0.05, fusion: 'full' | 'detail' | 'surface' | 'centroid' = 'full') => {
    if (runningReconstruction.current) return;
    runningReconstruction.current = true;
    setReconstruction({id: '', kind: 'reconstruct', status: 'queued'});
    try {
      let next = await jsonRequest<JobView>(`/clips/${encodeURIComponent(id)}/scene/reconstruct`, {voxel_size: size, registration: method, exclude_dynamic: exclude, preserve_detail: detail, detail_voxel_size: detailSize, fusion_mode: fusion});
      if (generation === clipGeneration.current) setReconstruction(next);
      for (let attempts = 0; next.status === 'queued' || next.status === 'running'; attempts++) {
        if (attempts > 1200) throw new Error('场景重建等待超时，请保留任务 ID 后检查后台');
        await new Promise(resolve => setTimeout(resolve, 500));
        next = await request<JobView>(`/jobs/${next.id}`);
        if (generation === clipGeneration.current && (next.status === 'queued' || next.status === 'running')) setReconstruction(next);
      }
      if (next.status === 'failed') throw new Error(next.error || '场景重建失败');
      if (generation !== clipGeneration.current) return;
      const [updated, rebuiltMap] = await Promise.all([
        request<SceneAnnotations>(`/clips/${encodeURIComponent(id)}/scene/annotations`),
        loadCachedSceneMap(id),
      ]);
      if (generation !== clipGeneration.current) return;
      if (!rebuiltMap) throw new Error('重建已完成，但场景地图文件不存在');
      setSceneMap(rebuiltMap);
      setRegisteredMatrices(rebuiltMap.world_from_lidar);
      setPreserveDetail(detail); setDetailVoxelSize(detailSize); setVoxelSize(size);
      setFusionMode(fusion);
      if (!sceneDirtyRef.current) setScene(updated);
      setReconstruction(next);
      flash('整段 clip 已重建为统一世界场景，时间轴只切换时刻与相机');
    } catch (error) {
      if (generation === clipGeneration.current) {setReconstruction(j => ({id: j?.id ?? '', kind: 'reconstruct', status: 'failed', error: errorText(error)})); flash(`场景重建失败：${errorText(error)}`, true);}
    } finally {runningReconstruction.current = false;}
  }, [flash]);

  const loadClip = useCallback(async (id: string) => {
    const generation = ++clipGeneration.current;
    ++frameGeneration.current;
    seekController.current?.abort(); setReviewing(false); setPreparedFrame(null); setBuffering(false); setPlaybackSession(value => value + 1);
    setCalibrationOpen(false); setCalibrationDirty(false); setCalibrationPreview(null); calibrationBase.current = null;
    setAutomaticCalibrationOpen(false);
    setLatrBatchOpen(false);
    calibrationClipBase.current = null;
    setProjectionOpen(false); setProjectionDirty(false); setProjectionBusy(false); setProjectionPreview(null); setProjectionPickMode(null);
    setProjectionPickedPoint(null); setProjectionPickedPixel(null);
    setPlaying(false); setLoading(true); seekController.current = null; setSceneMap(null); setRegisteredMatrices({}); setReconstruction(null); setModelPromptMode(null);
    try {
      const next = await request<Clip>(`/clips/${encodeURIComponent(id)}`);
      const [labels, sceneLabels, map, registrationResponse] = await Promise.all([
        request<Annotations>(`/clips/${encodeURIComponent(id)}/frames/${next.frames[0].id}/annotations`),
        request<SceneAnnotations>(`/clips/${encodeURIComponent(id)}/scene/annotations`),
        loadCachedSceneMap(id),
        fetch(`/api/clips/${encodeURIComponent(id)}/scene/registration`),
      ]);
      if (!registrationResponse.ok && registrationResponse.status !== 404) throw new Error(`读取场景位姿失败 HTTP ${registrationResponse.status}`);
      const registered = registrationResponse.ok ? await registrationResponse.json() as {world_from_lidar: Record<string, number[][]>} : null;
      if (next.source === '4d-clip-import' && !map) throw new Error('外部预重建场景地图不可用；请重启后端以启用全量 NPY 点云接口');
      const skipDuplicateGlobalCloud = next.source === '4d-clip-import' && map;
      const framePoints = skipDuplicateGlobalCloud ? null : await request<{points: number[][]; point_count: number}>(`/clips/${encodeURIComponent(id)}/frames/${next.frames[0].id}/points`);
      if (generation !== clipGeneration.current) return;
      if (framePoints) next.frames[0] = {...next.frames[0], points: framePoints.points, point_count: framePoints.point_count};
      sceneDirtyRef.current = false; frameDirtyRef.current = false;
      setClip(next); setFrameIndex(0); setAnnotations(labels); setScene(sceneLabels); setSceneMap(map); setSceneMode(true);
      setRegisteredMatrices(registered?.world_from_lidar ?? map?.world_from_lidar ?? {});
      if (map) {setPreserveDetail(map.preserve_detail ?? false); setDetailVoxelSize(map.detail_voxel_size ?? 0.05); setVoxelSize(map.voxel_size); setFusionMode(map.fusion_mode ?? (map.preserve_detail ? 'detail' : 'centroid'));}
      setFrameDirty(false); setSceneDirty(false); setSelectedId(null); setDraftVector([]); setDraftPolygon([]);
      setActiveCamera(next.cameras[0]?.id ?? ''); setExpandedCamera(null); setOccupancy(null); setProposal(null); setJob(null);
      setModelPoints([]); setModelPointLabels([]); setModelBox(undefined);
      const url = new URL(window.location.href);
      url.searchParams.set('clip', next.id);
      window.history.replaceState(window.history.state, '', `${url.pathname}${url.search}${url.hash}`);
      if (!map) void reconstruct(id, generation, 'poses', 0.3, true);
    } catch (error) {if (generation === clipGeneration.current) flash(`读取 clip 失败：${errorText(error)}`, true);}
    finally {if (generation === clipGeneration.current) setLoading(false);}
  }, [flash, reconstruct]);

  useEffect(() => {
    let active = true;
    Promise.all([
      refreshClips(),
      request<{default_clip_id?: string}>('/workspace').catch((): {default_clip_id?: string} => ({})),
    ]).then(([rows, workspace]) => {
      if (!active) return;
      const requestedId = new URLSearchParams(window.location.search).get('clip');
      const initial = rows.find(row => row.id === requestedId)
        ?? rows.find(row => row.id === workspace.default_clip_id) ?? rows[0];
      if (initial) return loadClip(initial.id);
      setLoading(false);
    }).catch(error => {if (active) {flash(`连接服务失败：${errorText(error)}`, true); setLoading(false);}});
    void refreshModels().catch(() => {});
    return () => {active = false;};
  }, [loadClip, refreshClips, refreshModels, flash]);
  useEffect(() => {
    const guard = (event: BeforeUnloadEvent) => {if (frameDirtyRef.current || sceneDirtyRef.current || draftPending.current) {event.preventDefault(); event.returnValue = '';}};
    window.addEventListener('beforeunload', guard); return () => window.removeEventListener('beforeunload', guard);
  }, []);

  const confirmDiscard = (frameOnly = false) => {
    if (projectionBusy) return false;
    if (!(frameOnly ? frameDirty || projectionDirty : dirty || calibrationDirty || projectionDirty) && !draftVector.length && !draftPolygon.length) return true;
    return window.confirm(frameOnly ? '当前帧有未保存的像素/点级标注或未完成绘制，放弃并切帧？全场景轨迹修改会继续保留。' : '存在未保存标注、外参或未完成绘制，放弃这些修改并继续？');
  };
  const applyPlaybackFrame = useCallback((bundle: PlaybackFrame) => {
    if (!reviewState.current || clipContext.current !== bundle.clipId) return;
    if (reviewInitialIndex.current !== null) reviewFitReference.current = [];
    setClip(current => current?.id === bundle.clipId ? {...current,
      frames: current.frames.map((item, index) => index === bundle.index
        ? {...item, points: bundle.points, point_count: bundle.pointCount} : {...item, points: []})} : current);
    setFrameIndex(bundle.index); setAnnotations(bundle.annotations); setPreparedFrame(bundle);
    reviewInitialIndex.current = null;
    setFrameDirty(false); frameDirtyRef.current = false;
  }, []);
  const seekPlayback = useCallback(async (index: number, resume = false) => {
    if (!clip || index < 0 || index >= clip.frames.length) return;
    setPlaying(false); setPlaybackEnded(false); setLoading(true);
    reviewFitReference.current = [];
    setPlaybackSession(value => value + 1);
    reviewInitialIndex.current = null;
    seekController.current?.abort();
    const controller = new AbortController();
    seekController.current = controller;
    try {
      const bundle = await loadPlaybackFrame(clip, index, controller.signal);
      if (!controller.signal.aborted && reviewState.current && clipContext.current === clip.id) {
        applyPlaybackFrame(bundle);
        if (resume) setPlaying(true);
      }
    } catch (error) {if (!controller.signal.aborted) {controller.abort(); flash(`回放帧读取失败：${errorText(error)}`, true);}}
    finally {controller.abort(); if (seekController.current === controller) {seekController.current = null; setLoading(false);}}
  }, [clip, applyPlaybackFrame, flash]);
  useClipPlayback({clip, index: reviewInitialIndex.current ?? frameIndex, playing: reviewing && playing, session: playbackSession,
    speed: playbackSpeed, loop: playbackLoop, onFrame: applyPlaybackFrame, onBuffering: setBuffering,
    onEnd: () => {setPlaying(false); setPlaybackEnded(true);},
    onError: error => {setPlaying(false); flash(`回放失败：${errorText(error)}`, true);},
  });
  useEffect(() => () => seekController.current?.abort(), []);
  useEffect(() => {
    setLaneMergeOpen(false); setLanePreview(null); setLaneMergeUndo(null);
  }, [laneScopeKey, task, reviewing]);
  const moveFrame = useCallback(async (index: number, playback = false) => {
    if (reviewing) {await seekPlayback(index); return;}
    if (!clip || busy || jobBusy || projectionBusy || index === frameIndex || index < 0 || index >= clip.frames.length) return;
    if (!playback && !confirmDiscard(true)) return;
    if (playback && (frameDirtyRef.current || sceneDirtyRef.current || projectionOpen)) {setPlaying(false); return;}
    setProjectionPreview(null); setProjectionPickMode(null); consumeProjectionPick();
    const generation = ++frameGeneration.current;
    const clipId = clip.id;
    setLoading(true);
    try {
      const frameId = clip.frames[index].id;
      const skipDuplicateGlobalCloud = clip.source === '4d-clip-import' && Boolean(sceneMap?.point_buffer);
      const [next, framePoints] = await Promise.all([
        request<Annotations>(`/clips/${encodeURIComponent(clipId)}/frames/${frameId}/annotations`),
        skipDuplicateGlobalCloud ? Promise.resolve(null) : request<{points: number[][]; point_count: number}>(`/clips/${encodeURIComponent(clipId)}/frames/${frameId}/points`),
      ]);
      if (generation !== frameGeneration.current || clipContext.current !== clipId) return;
      setClip(current => current?.id === clipId ? {
        ...current,
        frames: current.frames.map((item, position) => position === index
          ? {...item, points: framePoints?.points ?? [], point_count: framePoints?.point_count ?? item.point_count}
          : {...item, points: []}),
      } : current);
      setFrameIndex(index); setAnnotations(next); setFrameDirty(false); frameDirtyRef.current = false;
      if (!sceneMode) setSelectedId(null);
      setDraftVector([]); setDraftPolygon([]); setModelPoints([]); setModelPointLabels([]); setModelBox(undefined); setProposal(null);
      return true;
    } catch (error) {flash(`切换帧失败：${errorText(error)}`, true); setPlaying(false);}
    finally {if (generation === frameGeneration.current) setLoading(false);}
  }, [clip, busy, jobBusy, projectionBusy, projectionDirty, projectionOpen, frameIndex, frameDirty, sceneMode, sceneMap, draftVector.length, draftPolygon.length, flash, reviewing, seekPlayback]);

  const editFrame = (change: (a: Annotations) => Annotations) => {
    if (reviewing || busy || jobBusy) return;
    setAnnotations(change); setFrameDirty(true); frameDirtyRef.current = true; editGeneration.current++; setPlaying(false);
  };
  const editScene = (change: (s: SceneAnnotations) => SceneAnnotations) => {
    if (editBlocked) return;
    setScene(previous => previous ? change(previous) : previous); setSceneDirty(true); sceneDirtyRef.current = true; editGeneration.current++; setPlaying(false);
  };
  const save = async (): Promise<boolean> => {
    if (!clip || !frame || saving || loading) return false;
    if (activeLanePreview) {flash('请先应用或取消车道线融合预览，再保存。', true); return false;}
    const draft = draftVector.length ? commitVectorDraft() : null;
    if (draftVector.length && !draft) return false;
    const sceneToSave = draft?.scene ?? scene;
    const frameToSave = draft?.annotations ?? annotations;
    const shouldSaveScene = sceneDirtyRef.current;
    const shouldSaveFrame = frameDirtyRef.current;
    if (!shouldSaveScene && !shouldSaveFrame) return true;
    const key = currentKey, generation = editGeneration.current;
    setSaving(true); setPlaying(false);
    try {
      if (shouldSaveScene && sceneToSave) {
        const saved = await jsonRequest<SceneAnnotations>(`/clips/${encodeURIComponent(clip.id)}/scene/annotations`, sceneToSave, 'PUT');
        if (clipContext.current === clip.id && generation === editGeneration.current) {setScene(saved); setSceneDirty(false); sceneDirtyRef.current = false;}
      }
      if (shouldSaveFrame) {
        const saved = await jsonRequest<Annotations>(`/clips/${encodeURIComponent(clip.id)}/frames/${frame.id}/annotations`, frameToSave, 'PUT');
        if (context.current === key && generation === editGeneration.current) {setAnnotations(saved); setFrameDirty(false); frameDirtyRef.current = false;}
      }
      flash('全场景轨迹 / 当前帧标注已分别保存'); return true;
    } catch (error) {flash(`保存失败：${errorText(error)}。尚未保存的修改仍保留，可下载草稿备份。`, true); return false;}
    finally {setSaving(false);}
  };
  const startReview = async () => {
    if (!clip || busy || jobBusy || reviewStarting) return;
    if (reviewing) {await seekPlayback(0, true); return;}
    if (calibrationDirty) {flash('请先保存或还原外参修改，再回放已保存的标注。', true); return;}
    if (projectionOpen) {flash('请先关闭逐帧投影校正，再开始回放。', true); return;}
    if (draftPolygon.length) {flash('请先完成或取消图像多边形绘制，再回放。', true); return;}
    setReviewStarting(true);
    try {
      if (!(await save())) return;
      const id = clip.id;
      const [savedScene, poses, mapCheck] = await Promise.all([
        request<SceneAnnotations>(`/clips/${encodeURIComponent(id)}/scene/annotations`),
        request<{world_from_lidar: Record<string, number[][]>}>(`/clips/${encodeURIComponent(id)}/scene/registration`),
        fetch(`/api/clips/${encodeURIComponent(id)}/scene/map/points?offset=0&limit=1`),
      ]);
      if (clipContext.current !== id) return;
      setRegisteredMatrices(poses.world_from_lidar);
      // Saved point labels or dynamic tracks can invalidate the old fused map.
      // Review uses current source labels immediately; stale map colours must
      // not be presented as the newly saved result.
      if (!mapCheck.ok) setSceneMap(null);
      reviewSpace.current = sceneMode; reviewFitReference.current = [];
      reviewColor.current = pointColorMode; reviewGroundOnly.current = groundOnly;
      reviewInitialIndex.current = 0;
      setScene(savedScene); setReviewing(true); setSceneMode(true); setTool('select'); setSelectedId(null);
      setPointColorMode('semantic'); setGroundOnly(false);
      setAnnotationProjection(true); setShowBoxes(true); setShowVectors(true);
      setCalibrationOpen(false); setModal(null); setExpandedCamera(null); setModelPromptMode(null); setProposal(null);
      setPreparedFrame(null); setPlaybackEnded(false); setPlaybackSession(value => value + 1); setPlaying(true);
    } catch (error) {flash(`无法开始回放：${errorText(error)}`, true);}
    finally {setReviewStarting(false);}
  };
  const exitReview = () => {
    setPlaying(false); seekController.current?.abort(); seekController.current = null;
    setLoading(false); setReviewing(false); setPreparedFrame(null); setBuffering(false);
    setPlaybackSession(value => value + 1); setSceneMode(reviewSpace.current); reviewInitialIndex.current = null;
    setPointColorMode(reviewColor.current); setGroundOnly(reviewGroundOnly.current);
  };
  const togglePlayback = () => {
    if (!reviewing) {void startReview(); return;}
    if (playing) {setPlaying(false); return;}
    setPlaybackEnded(false);
    if (clip && frameIndex === clip.frames.length - 1) void seekPlayback(0, true);
    else setPlaying(true);
  };

  const switchTask = (next: string) => {
    if ((draftVector.length || draftPolygon.length) && !window.confirm('切换任务将放弃未完成绘制，是否继续？')) return;
    setTask(next); setTool('select'); setDraftVector([]); setDraftPolygon([]); setPlaying(false); setModelPromptMode(null);
    setClassId(['lane', 'vector'].includes(next) ? 'lane' : ['image', 'semantic', 'occupancy'].includes(next) ? 'road' : 'vehicle');
    if (['lane', 'vector'].includes(next)) {setView('top'); setGroundOnly(true); setPointColorMode('intensity'); setShowVectors(true); setSelectedId(null);}
    else setGroundOnly(false);
    if (next === 'semantic' || next === 'occupancy') {setSceneMode(false); setPointColorMode('semantic');}
    else if (next === 'fusion') setSceneMode(true);
  };
  const switchSpace = async (next: boolean) => {
    if (draftVector.length && !window.confirm('切换坐标空间将放弃未完成矢量绘制，是否继续？')) return;
    if (clip?.source === '4d-clip-import') {
      if (next) {
        setClip(current => current ? {...current, frames: current.frames.map(item => ({...item, points: []}))} : current);
      } else if (frame && !frame.points.length && !sceneMap?.point_buffer) {
        setLoading(true);
        try {
          const loaded = await request<{points: number[][]; point_count: number}>(`/clips/${encodeURIComponent(clip.id)}/frames/${frame.id}/points`);
          setClip(current => current?.id === clip.id ? {...current, frames: current.frames.map(item => item.id === frame.id
            ? {...item, points: loaded.points, point_count: loaded.point_count} : {...item, points: []})} : current);
        } catch (error) {flash(`读取全量点云失败：${errorText(error)}`, true); return;}
        finally {setLoading(false);}
      }
    }
    setSceneMode(next); setSelectedId(null); setDraftVector([]); setTool('select'); setPlaying(false);
  };
  const createBox = (center: Vec3) => {
    if (geometryEditBlocked) return;
    const id = uid('track');
    const box: Box = {id, track_id: id, label: classId, center: [center[0], center[1], center[2] + 0.8], size: [4.2, 1.8, 1.6], yaw: 0};
    if (sceneMode && frame) editScene(s => ({...s, tracks: [...s.tracks, {id, label: classId, mode: 'dynamic', keyframes: [{frame_id: frame.id, center: box.center, size: box.size, yaw: 0}]}]}));
    else editFrame(a => ({...a, boxes: [...a.boxes, box]}));
    setSelectedId(id); setTransformMode('scale'); setTool('select'); setShowBoxes(true);
  };
  const selectAnnotation = (id: string | null) => {
    if (vectorTask && tool === 'vector' && draftVector.length) {flash('请先完成或取消正在绘制的车道线，再选择已有标注。', true); return;}
    setSelectedId(id);
    if (vectorTask && renderedAnnotations.vectors.some(vector => vector.id === id)) {setTool('select'); setPlaying(false);}
    if (task === 'detection' && id !== selectedId && renderedAnnotations.boxes.some(box => box.id === id)) {
      setTransformMode('scale'); setTool('select'); setPlaying(false);
    }
  };
  const updateBoxById = (id: string, patch: Partial<Box>) => {
    if (geometryEditBlocked) return;
    if (sceneMode && frame && clip) {
      editScene(s => ({...s, tracks: s.tracks.map(track => {
        if (track.id !== id) return track;
        const current = evaluateTrack(track, frame, clip);
        return current ? upsertKeyframe(track, {...current, ...patch}, frame.id) : track;
      })}));
      if (patch.track_id) setSelectedId(patch.track_id);
    } else editFrame(a => ({...a, boxes: a.boxes.map(box => box.id === id ? {...box, ...patch} : box)}));
  };
  const updateBox = (patch: Partial<Box>) => {if (selectedId) updateBoxById(selectedId, patch);};
  const deleteSelected = () => {
    if (geometryEditBlocked) return;
    if (sceneMode && (selectedBox || selectedVector)) editScene(s => ({...s, tracks: s.tracks.filter(t => t.id !== selectedId), vectors: s.vectors.filter(v => v.id !== selectedId)}));
    else editFrame(a => ({...a, boxes: a.boxes.filter(b => b.id !== selectedId), vectors: a.vectors.filter(v => v.id !== selectedId), image_polygons: a.image_polygons.filter(p => p.id !== selectedId)}));
    setSelectedId(null);
  };
  const chooseCamera = (id: string) => {
    if (activeCamera === id) return true;
    if (projectionBusy) return false;
    if (projectionOpen && projectionDirty && !window.confirm('切换相机会丢弃尚未保存的对应点和投影候选，确定切换？')) return false;
    if ((draftPolygon.length || modelPoints.length || modelBox) && !window.confirm('切换相机将清除未完成多边形和 SAM2 提示，是否继续？')) return false;
    setActiveCamera(id); setDraftPolygon([]); setModelPoints([]); setModelPointLabels([]); setModelBox(undefined); return true;
  };
  const commitVectorDraft = () => {
    if (editBlocked) return null;
    const closed = closedDraft;
    if (draftVector.length < (closed ? 3 : 2)) {flash(closed ? '面至少需要 3 个顶点' : '线至少需要 2 个顶点', true); return null;}
    const id = uid('vector'), vector = {id, label: classId, points: draftVector, closed};
    const nextScene = sceneMode && scene ? {...scene, vectors: [...scene.vectors, vector]} : scene;
    const nextAnnotations = !sceneMode ? {...annotations, vectors: [...annotations.vectors, vector]} : annotations;
    if (sceneMode && nextScene) editScene(() => nextScene); else editFrame(() => nextAnnotations);
    setDraftVector([]); setSelectedId(id); setTool('select');
    return {scene: nextScene, annotations: nextAnnotations};
  };
  const finishVector = () => {commitVectorDraft();};
  const startVector = () => {
    if (geometryEditBlocked) return;
    setTool('vector'); setView('top'); setPlaying(false); setSelectedId(null); setShowVectors(true);
  };
  useEffect(() => {
    if (!vectorTask || tool !== 'vector' || modal || expandedCamera || calibrationOpen || projectionOpen) return;
    const key = (event: KeyboardEvent) => {
      if (editBlocked || event.ctrlKey || event.metaKey || event.altKey || (event.target instanceof HTMLElement && event.target.closest('input, textarea, select, [contenteditable="true"]'))) return;
      if (event.key === 'Enter') {event.preventDefault(); finishVector();}
      else if (event.key === 'Backspace') {event.preventDefault(); setDraftVector(p => p.slice(0, -1));}
      else if (event.key === 'Escape') {event.preventDefault(); setDraftVector([]); setTool('select');}
    };
    window.addEventListener('keydown', key);
    return () => window.removeEventListener('keydown', key);
  }, [vectorTask, tool, editBlocked, draftVector, closedDraft, classId, sceneMode, finishVector, modal, expandedCamera, calibrationOpen]);
  const finishPolygon = () => {
    if (draftPolygon.length < 3) {flash('图像区域至少需要 3 个顶点', true); return;}
    const id = uid('mask');
    editFrame(a => ({...a, image_polygons: [...a.image_polygons, {id, camera_id: activeCamera, label: classId, points: draftPolygon}]}));
    setDraftPolygon([]); setSelectedId(id);
  };
  const updateElement = (patch: {label?: string; closed?: boolean; points?: number[][]}) => {
    if (geometryEditBlocked) return;
    if (selectedVector && sceneMode) editScene(s => ({...s, vectors: s.vectors.map(v => v.id === selectedId ? {...v, ...patch, points: (patch.points ?? v.points) as Vec3[]} : v)}));
    else editFrame(a => ({...a, vectors: a.vectors.map(v => v.id === selectedId ? {...v, ...patch, points: (patch.points ?? v.points) as Vec3[]} : v), image_polygons: a.image_polygons.map(p => p.id === selectedId ? {...p, ...patch} : p)}));
  };
  const updateVectorPoint = (id: string, index: number, point: Vec3) => {
    if (geometryEditBlocked || !vectorTask || !point.every(Number.isFinite)) return;
    const update = (vectors: Vector[]) => vectors.map(vector => vector.id === id && index >= 0 && index < vector.points.length
      ? {...vector, points: vector.points.map((value, i) => i === index ? point : value)} : vector);
    if (sceneMode) editScene(value => ({...value, vectors: update(value.vectors)}));
    else editFrame(value => ({...value, vectors: update(value.vectors)}));
  };
  const applyLaneMerge = (result: LaneMergeResult) => {
    if (editBlocked || activeLanePreview?.result !== result) return;
    const before = laneVectors;
    if (sceneMode) editScene(value => ({...value, vectors: result.vectors}));
    else editFrame(value => ({...value, vectors: result.vectors}));
    setLaneMergeUndo({scopeKey: laneScopeKey, before, after: result.vectors});
    setLanePreview(null); setLaneMergeOpen(false); setSelectedId(null); setTool('select');
    flash(`已融合 ${result.report.merged_count} 条重复车道线，请检查投影后保存`);
  };
  const undoLaneMerge = () => {
    if (geometryEditBlocked || !laneMergeUndo || laneMergeUndo.scopeKey !== laneScopeKey || laneMergeUndo.after !== laneVectors) return;
    if (sceneMode) editScene(value => ({...value, vectors: laneMergeUndo.before}));
    else editFrame(value => ({...value, vectors: laneMergeUndo.before}));
    setLaneMergeUndo(null); setSelectedId(null); flash('本次车道线融合已撤销');
  };

  const importFile = async (file?: File) => {
    if (!file || !confirmDiscard()) return;
    setPlaying(false); setLoading(true);
    try {const body = new FormData(); body.append('file', file); const next = await request<Clip>('/clips/import', {method: 'POST', body}); await refreshClips(); await loadClip(next.id); flash(`已导入 ${next.name}，正在准备整个场景`);}
    catch (error) {flash(`导入失败：${errorText(error)}`, true); setLoading(false);}
  };
  const exportFile = async (path: string, filename: string) => {
    if (!(await save())) return;
    try {const response = await fetch(`/api${path}`); if (!response.ok) throw new Error(`HTTP ${response.status}`); download(filename, await response.blob(), 'application/zip');}
    catch (error) {flash(`导出失败：${errorText(error)}`, true);}
  };
  const exportClip = () => {
    if (!clip || draftVector.length || draftPolygon.length) {flash('请完成或取消绘制后再导出', true); return;}
    void exportFile(`/clips/${clip.id}/export.zip`, `${clip.id}-dataset.zip`);
  };
  const runJob = async (kind: string) => {
    if (reviewing || !clip || !frame || runningJob.current || busy || reconstructing || !(await save())) return;
    if (kind === 'propagate' && !clip.frames[frameIndex + 1]) {flash('当前是最后一帧', true); return;}
    runningJob.current = true; setPlaying(false); setJob({id: '', kind, status: 'queued'});
    const key = currentKey, clipId = clip.id, frameId = frame.id, initialSceneRevision = scene?.revision ?? 0;
    try {
      let next = await jsonRequest<JobView>(`/clips/${clip.id}/jobs`, {kind, frame_id: frame.id, voxel_size: voxelSize,
        ...(kind === 'prelabel' ? {model_id: workerModel, prompt: workerLabel, scene_mode: sceneMode, options: {camera_id: activeCamera, label: workerLabel, points: modelPoints, point_labels: modelPointLabels, ...(modelBox ? {box: modelBox} : {}), score_threshold: scoreThreshold, ...(workerModel === 'latr' ? {weight_variant: latrWeight} : {})}} : {}),
        ...(kind === 'propagate' ? {target_frame_id: clip.frames[frameIndex + 1].id} : {})});
      setJob(next);
      for (let attempts = 0; next.status === 'queued' || next.status === 'running'; attempts++) {
        if (attempts > 1200) throw new Error('任务等待超时，请检查后台任务 ID');
        await new Promise(resolve => setTimeout(resolve, 500)); next = await request<JobView>(`/jobs/${next.id}`); setJob(next);
      }
      if (next.status === 'failed') throw new Error(next.error || '任务失败');
      if (context.current !== key) {flash('任务已完成，请返回原帧查看'); return;}
      if (kind === 'occupancy') setOccupancy({frameId, voxels: (next.result?.voxels ?? []) as Voxel[], size: Number(next.result?.voxel_size ?? voxelSize)});
      else if (kind === 'prelabel') setProposal({annotations: next.result?.proposal as Annotations, frameId, clipId,
        revision: Number((next.result?.proposal as Annotations)?.revision), sceneRevision: Number(next.result?.scene_revision ?? initialSceneRevision), sceneMode,
        model: String(next.result?.model_id ?? workerModel), weightVariant: next.result?.weight_variant as string | undefined, occupancy: next.result?.occupancy as Proposal['occupancy'], warnings: next.result?.warnings as string[]});
      else if (!frameDirtyRef.current) {
        const before = editGeneration.current;
        const saved = await request<Annotations>(`/clips/${clipId}/frames/${frameId}/annotations`);
        if (context.current === key && !frameDirtyRef.current && before === editGeneration.current) setAnnotations(saved);
      }
      flash(kind === 'prelabel' ? '真实模型推理完成，请审核候选后应用' : '计算完成');
    } catch (error) {setJob(j => ({id: j?.id ?? '', kind, status: 'failed', error: errorText(error)})); flash(`任务失败：${errorText(error)}`, true);}
    finally {runningJob.current = false;}
  };
  const applyProposal = async () => {
    if (busy || jobBusy || geometryEditBlocked) return;
    if (!proposal || !clip || !frame || proposal.clipId !== clip.id || proposal.frameId !== frame.id || dirty || proposal.sceneMode !== sceneMode || proposal.revision !== annotations.revision || (sceneMode && proposal.sceneRevision !== scene?.revision)) {flash('候选已过期或标注存在修改，请保存后重新生成', true); return;}
    const isLatr = proposal.model === 'latr';
    if (!window.confirm(isLatr ? `将 LATR 候选加入${sceneMode ? '世界场景' : '当前帧'}${autoMergeLatr ? '并融合重叠车道线' : ''}，保留其他任务标注。应用后请审核并保存。继续？` : sceneMode ? '将候选三维目标加入当前时刻关键帧，保留其他时刻轨迹；图像与点级结果替换当前帧对应标注。应用后仍需审核和保存。继续？' : '将候选应用为当前帧未保存标注，人工审核后再保存。继续？')) return;
    const key = currentKey, generation = editGeneration.current;
    let mergedCount = 0;
    setProposalApplying(true);
    try {
      const matrix = worldFromLidar(frame, coordinateMap);
      const used = new Set([...laneVectors.map(vector => vector.id), ...(sceneMode
        ? scene?.tracks.map(track => track.id) ?? []
        : [...annotations.boxes.map(box => box.id), ...annotations.image_polygons.map(polygon => polygon.id)])]);
      const incoming = proposal.annotations.vectors.map(vector => {
        const id = isLatr && used.has(vector.id) ? uid('lane') : vector.id;
        used.add(id);
        return {...vector, id, points: sceneMode ? vector.points.map(point => transformPoint(point, matrix)) : vector.points};
      });
      const combined = new Map(laneVectors.map(vector => [vector.id, vector]));
      incoming.forEach(vector => combined.set(vector.id, vector));
      let nextVectors = [...combined.values()];
      const beforeFusion = nextVectors;
      if (isLatr && autoMergeLatr && incoming.length) {
        const result = await jsonRequest<LaneMergeResult>(`/clips/${encodeURIComponent(clip.id)}/lanes/merge-preview`, {
          vectors: nextVectors, candidate_ids: incoming.map(vector => vector.id),
          ...(sceneMode ? {scene_revision: scene?.revision} : {}),
          max_distance: .25, max_angle_degrees: 15, min_overlap: 1,
        });
        nextVectors = result.vectors; mergedCount = result.report.merged_count;
      }
      if (context.current !== key || generation !== editGeneration.current || laneScope.current.scopeKey !== laneScopeKey
          || laneScope.current.source !== laneVectors || proposalRef.current !== proposal) {
        flash('标注上下文已改变，候选未应用，请重新检查。', true); return;
      }
      if (sceneMode) {
        editScene(value => {
          const tracks = [...value.tracks];
          if (!isLatr) for (const predicted of proposal.annotations.boxes) {
            const world = transformBox(predicted, matrix), index = tracks.findIndex(track => track.id === predicted.track_id);
            const base: SceneTrack = index >= 0 ? tracks[index] : {id: predicted.track_id, label: predicted.label, mode: 'dynamic', keyframes: []};
            const updated = upsertKeyframe(base, world, frame.id);
            if (index >= 0) tracks[index] = updated; else tracks.push(updated);
          }
          return {...value, tracks, vectors: nextVectors};
        });
        if (!isLatr) editFrame(value => ({...value, image_polygons: proposal.annotations.image_polygons, point_labels: proposal.annotations.point_labels}));
      } else if (isLatr) editFrame(value => ({...value, vectors: nextVectors}));
      else editFrame(() => ({...proposal.annotations, revision: annotations.revision}));
      if (isLatr && mergedCount) setLaneMergeUndo({scopeKey: laneScopeKey, before: beforeFusion, after: nextVectors});
      if (proposal.occupancy) {setOccupancy({frameId: frame.id, voxels: proposal.occupancy.voxels, size: proposal.occupancy.voxel_size, dimensions: proposal.occupancy.voxel_dimensions}); setTask('occupancy'); setSceneMode(false);}
      if (isLatr) {setTask('lane'); setShowVectors(true); setTool('select'); setSelectedId(null); setLaneMergeOpen(false);}
      setProposal(null); setModal(null); setModelPromptMode(null);
      flash(isLatr ? `LATR 候选已应用${mergedCount ? `，融合 ${mergedCount} 条重复线` : ''}，请核验后保存` : '候选已应用为未保存修改，请核验后保存');
    } catch (error) {flash(`候选未应用：${errorText(error)}。原标注和候选已保留，可重试。`, true);}
    finally {setProposalApplying(false);}
  };
  const openLatrBatch = async () => {
    if (!clip || !frame || reviewing || busy || jobBusy || reconstructing || projectionBusy) return;
    if (calibrationDirty || projectionDirty || activeLanePreview) {flash('请先保存或还原标定、投影和融合预览，再开始批量预标注。', true); return;}
    if (draftVector.length || draftPolygon.length) {flash('请先完成或取消正在绘制的标注，再开始批量预标注。', true); return;}
    if (!(await save())) return;
    setModal(null); setCalibrationOpen(false); setProjectionOpen(false); setAutomaticCalibrationOpen(false);
    setTask('lane'); setSceneMode(true); setView('top'); setClassId('lane'); setTool('select');
    setShowVectors(true); setAnnotationProjection(true); setPlaying(false); setSelectedId(null); setModelPromptMode(null);
    setLaneMergeOpen(false); setLatrBatchOpen(true);
  };
  const applyLatrBatch = (value: BatchProposal): boolean => {
    if (!clip || !scene || busy || jobBusy || reviewing || reconstructing || calibrationDirty || projectionDirty
        || value.clip_id !== clip.id || value.coordinate_frame !== 'world' || value.scene_revision !== scene.revision
        || dirty || draftVector.length || draftPolygon.length) {
      flash('批量候选与当前场景版本不一致，或存在未保存修改。请检查任务状态后重新应用。', true); return false;
    }
    if (!value.vectors.length) {flash('此批次没有可应用的车道线候选。', true); return false;}
    const used = new Set([...scene.vectors.map(vector => vector.id), ...scene.tracks.map(track => track.id)]);
    if (value.vectors.some(vector => used.has(vector.id)) || new Set(value.vectors.map(vector => vector.id)).size !== value.vectors.length) {
      flash('此批次候选已应用，或其 ID 与已有标注冲突；未重复追加。', true); return false;
    }
    if (scene.vectors.length + value.vectors.length > 5000) {flash('场景矢量超过当前格式的 5000 条限制，请先融合已有线，或减少本次批量帧数。', true); return false;}
    // Batch candidates already use the registered world coordinate system.
    // Keep every observation separate until the operator chooses to fuse it.
    setScene(current => current ? {...current, vectors: [...current.vectors, ...value.vectors]} : current);
    setSceneDirty(true); sceneDirtyRef.current = true; editGeneration.current++;
    setLatrBatchOpen(false); setTask('lane'); setSceneMode(true); setTool('select'); setSelectedId(null);
    setShowVectors(true); setAnnotationProjection(true); setPlaying(false); setProposal(null); setLaneMergeUndo(null);
    setLaneMergeOpen(true);
    flash(`已加入 ${value.source_frame_ids.length} 帧、${value.vectors.length} 条独立候选。请预览融合，再审核保存。`);
    return true;
  };
  const openModels = async () => {setModal('models'); try {await refreshModels();} catch (error) {flash(`模型状态读取失败：${errorText(error)}`, true);}};
  const saveConnection = async (modelId: string) => {
    setConnectionBusy(true);
    try {await jsonRequest('/model-connections', {connections: connections.map(({model_id, endpoint}) => ({model_id, endpoint}))}, 'PUT'); await request(`/model-connections/${encodeURIComponent(modelId)}/check`, {method: 'POST'}); await refreshModels(); flash('服务连接已保存并检查');}
    catch (error) {flash(`模型连接失败：${errorText(error)}`, true);}
    finally {setConnectionBusy(false);}
  };
  const rebuildScene = async () => {if (reviewing || !clip || reconstructing || !(await save())) return; void reconstruct(clip.id, clipGeneration.current, registration, voxelSize, excludeDynamic, preserveDetail, detailVoxelSize, fusionMode);};

  const openProjectionCalibration = async () => {
    if (!clip || !frame || reviewing || busy || jobBusy || reconstructing || projectionBusy) return;
    if (calibrationDirty) {flash('请先保存或还原全局外参，再进行逐帧投影校正。', true); return;}
    if (draftVector.length || draftPolygon.length) {flash('请先完成或取消正在绘制的标注，再进行投影校正。', true); return;}
    if (!(await save())) return;
    setCalibrationOpen(false); setProjectionOpen(true); setPlaying(false); setTool('select');
    setModelPromptMode(null); setRawProjection(true); setAnnotationProjection(true); setSelectedId(null);
    setProjectionPickMode(null); consumeProjectionPick();
  };

  const openAutomaticCalibration = async () => {
    if (!clip || !frame || reviewing || busy || jobBusy || reconstructing || projectionBusy) return;
    if (calibrationDirty || projectionDirty) {flash('请先保存或还原当前外参修改与投影候选，再进行自动标定。', true); return;}
    if (draftVector.length || draftPolygon.length) {flash('请先完成或取消正在绘制的标注，再进行自动标定。', true); return;}
    if (!(await save())) return;
    setCalibrationOpen(false); setProjectionOpen(false); setProjectionPreview(null); setProjectionPickMode(null); consumeProjectionPick();
    setAutomaticCalibrationOpen(true); setPlaying(false); setTool('select'); setModelPromptMode(null);
    setRawProjection(true); setAnnotationProjection(true);
  };
  const refreshProjectionParameters = async (clipId: string) => {
    const [updated, registration] = await Promise.all([
      request<Clip>(`/clips/${encodeURIComponent(clipId)}`),
      request<{world_from_lidar: Record<string, Matrix4>}>(`/clips/${encodeURIComponent(clipId)}/scene/registration`),
    ]);
    const frames = new Map(updated.frames.map(item => [item.id, item]));
    // Preserve loaded point buffers and current labels while refreshing projection parameters.
    if (clipContext.current !== clipId) return;
    setClip(current => current?.id === clipId ? {...current, cameras: updated.cameras,
      frames: current.frames.map(item => ({...item, ...frames.get(item.id), points: item.points}))} : current);
    setRegisteredMatrices(registration.world_from_lidar);
  };
  const automaticCalibrationSaved = async () => {
    if (!clip) return;
    await refreshProjectionParameters(clip.id);
    setProjectionPreview(null);
    flash('自动标定参数已保存，点云与三维标注投影已更新。');
  };
  const inspectAutomaticCalibration = async (frameId: string, cameraId: string) => {
    if (!clip) return;
    const index = clip.frames.findIndex(item => item.id === frameId);
    if (index < 0 || !clip.cameras.some(item => item.id === cameraId)) throw new Error('该帧或相机已不存在，请重新读取场景。');
    if (index !== frameIndex && !(await moveFrame(index))) return;
    setActiveCamera(cameraId); setAutomaticCalibrationOpen(false); setProjectionOpen(true);
    setProjectionPreview(null); setProjectionPickMode(null); consumeProjectionPick();
    setRawProjection(true); setAnnotationProjection(true); setPlaying(false); setTool('select');
  };
  const closeProjectionCalibration = () => {
    setProjectionOpen(false); setProjectionDirty(false); setProjectionPreview(null); setProjectionPickMode(null); consumeProjectionPick();
  };

  const previewCalibration = (draft: Clip | null, document?: CalibrationDocument) => {
    if (!clip) return;
    if (!draft || !document) {
      const base = calibrationBase.current;
      if (base?.clip_id === clip.id) {
        const original = calibrationClipBase.current;
        setClip(current => current?.id === base.clip_id ? (original ?? applyCalibration(current, base)) : current);
        setRegisteredMatrices(base.world_from_lidar);
      }
      calibrationBase.current = null;
      calibrationClipBase.current = null;
      setCalibrationPreview(null);
      return;
    }
    if (draft.id !== clip.id) return;
    if (!calibrationBase.current) {calibrationClipBase.current = clip; calibrationBase.current = {
      schema_version: '1.0', clip_id: clip.id, revision: '',
      cameras: Object.fromEntries(clip.cameras.map(camera => [camera.id, camera.camera_from_ego])),
      lidar_to_ego: Object.fromEntries(clip.frames.map(item => [item.id, item.lidar_to_ego])),
      world_from_lidar: Object.fromEntries(clip.frames.map(item => [item.id, worldFromLidar(item, coordinateMap)])),
    };}
    setClip(current => current?.id === document.clip_id ? draft : current);
    setCalibrationPreview(document); setPlaying(false);
  };
  const calibrationSaved = async (document: CalibrationDocument, options: {lidarChanged: boolean}) => {
    if (document.clip_id !== clip?.id) return;
    setClip(current => current?.id === document.clip_id ? applyCalibration(current, document, document) : current);
    setRegisteredMatrices(document.world_from_lidar);
    const rebuild = document.map_rebuild_required ?? options.lidarChanged;
    if (rebuild) {setSceneMap(null); setReconstruction(null);}
    calibrationBase.current = null;
    calibrationClipBase.current = null;
    setCalibrationPreview(null); setCalibrationDirty(false);
    flash(rebuild ? '外参已保存并保留原始标定备份。请重建完整场景，复核世界坐标标注。' : '外参已保存，整段 clip 的投影已更新，可导出标定。');
    try {await refreshProjectionParameters(document.clip_id);}
    catch (error) {flash(`外参已保存，但最新投影参数刷新失败：${errorText(error)}。请重新加载场景。`, true);}
  };

  const cameraCards = (expanded = false) => clip?.cameras.filter(c => !expanded || c.id === expandedCamera).map(camera => <div className={`camera-card ${activeCamera === camera.id ? 'camera-active' : ''}`} key={camera.id}>
    <CameraView camera={camera} frame={frame} annotations={imageAnnotations} preparedImage={reviewing && preparedFrame?.frameId === frame?.id ? preparedFrame?.images[camera.id] : undefined} projections={projections[camera.id] ?? []}
      selected={activeCamera === camera.id} tool={task === 'image' && !projectionOpen && !reviewing && !busy && !jobBusy && !modelPromptMode ? 'polygon' : 'select'}
      calibrationPicking={projectionOpen && projectionPickMode === '2d' && activeCamera === camera.id}
      onCalibrationPixel={pixel => {if (!projectionBusy) setProjectionPickedPixel(pixel);}}
      calibrationAnchors={projectionOpen && activeCamera === camera.id ? cameraProjectionAnchors : []}
      onSelect={() => chooseCamera(camera.id)} onAddPolygonPoint={point => {if (busy || jobBusy) return; if (activeCamera !== camera.id) {setActiveCamera(camera.id); setDraftPolygon([point]);} else setDraftPolygon(p => [...p, point]); setPlaying(false);}}
      onFinishPolygon={finishPolygon} draftPolygon={activeCamera === camera.id ? draftPolygon : []}
      showAnnotationProjection={annotationProjection} showProjection={rawProjection} showBoxes={showBoxes} showVectors={showVectors} selectedId={selectedId}
      modelPromptMode={!reviewing && activeCamera === camera.id ? modelPromptMode : null} modelPoints={activeCamera === camera.id ? modelPoints : []} modelPointLabels={activeCamera === camera.id ? modelPointLabels : []} modelBox={activeCamera === camera.id ? modelBox : undefined}
      onModelPoint={(point, label) => {setModelPoints(p => [...p, point]); setModelPointLabels(p => [...p, negativePrompt || label === 0 ? 0 : 1]); setPlaying(false);}}
      onModelBox={box => {setModelBox(box); setPlaying(false);}}/>
    <button className="camera-expand icon-button" aria-label={`放大 ${camera.name}`} onClick={() => {if (chooseCamera(camera.id)) setExpandedCamera(expanded ? null : camera.id);}}><Icon name={expanded ? 'close' : 'expand'} size={14}/></button>
  </div>);

  const modelControls = <>
    <label className="field-label">预标注模型<select aria-label="预标注模型" value={workerModel} onChange={e => {setWorkerModel(e.target.value); setModelPromptMode(null); if (e.target.value === 'latr') setWorkerLabel('lane');}}>{Object.entries(modelNames).map(([id, name]) => <option value={id} key={id}>{name}</option>)}</select></label>
    <div className={`model-readiness ${selectedModel?.available ? 'ready' : ''}`}><span className="status-dot"/>{modelStatus(selectedModel)}</div>
    <label className="field-label">候选类别<input aria-label="候选类别" value={workerLabel} maxLength={80} onChange={e => setWorkerLabel(e.target.value)}/></label>
    {workerModel === 'sam2' ? <><p className="micro-copy">在左侧所选相机添加正/负提示点或拖出提示框。无提示时执行自动掩码生成；类别名称不会触发文本识别。</p><div className="prompt-tools"><button className={`button subtle ${modelPromptMode === 'point' && !negativePrompt ? 'active' : ''}`} aria-label="SAM2正提示点" onClick={() => {setModelPromptMode('point'); setNegativePrompt(false);}}>＋ 正点</button><button className={`button subtle ${modelPromptMode === 'point' && negativePrompt ? 'active' : ''}`} aria-label="SAM2负提示点" onClick={() => {setModelPromptMode('point'); setNegativePrompt(true);}}>− 负点</button><button className={`button subtle ${modelPromptMode === 'box' ? 'active' : ''}`} aria-label="SAM2提示框" onClick={() => setModelPromptMode('box')}>提示框</button><button className="button ghost" onClick={() => {setModelPromptMode(null); setModelPoints([]); setModelPointLabels([]); setModelBox(undefined);}}>清除</button></div><p className="micro-copy">{modelPoints.length} 个提示点 · {modelBox ? '已选提示框' : '未选提示框'} · {activeCamera}</p></> : <>{workerModel === 'latr' && <><p className="micro-copy">请选择当前帧的前视相机。LATR 使用图像与真实标定生成 LiDAR 坐标三维车道折线。</p><label className="closed-field"><input type="checkbox" aria-label="应用LATR时融合重复车道线" checked={autoMergeLatr} disabled={busy || jobBusy} onChange={e => setAutoMergeLatr(e.target.checked)}/>单帧候选应用时融合重复车道线</label><label className="field-label">LATR 推理权重<select aria-label="LATR推理权重" value={latrWeight} onChange={e => setLatrWeight(e.target.value as 'once' | 'openlane')}><option value="once">ONCE · once.pth</option><option value="openlane">OpenLane · openlane.pth</option></select></label></>}<label className="field-label">置信度阈值<input type="number" aria-label="模型置信度阈值" min="0" max="1" step="0.05" value={scoreThreshold} onChange={e => {if (Number.isFinite(e.target.valueAsNumber)) setScoreThreshold(Math.min(1, Math.max(0, e.target.valueAsNumber)));}}/></label></>}
    <button className="button primary full-width" aria-label="生成候选标注" disabled={!clip || busy || jobBusy || reconstructing || !workerLabel.trim() || !selectedModel?.available} onClick={() => void runJob('prelabel')}><Icon name="model" size={15}/>{jobBusy ? '模型推理中…' : workerModel === 'latr' ? '运行当前帧预标注' : '运行模型预标注'}</button>
    {workerModel === 'latr' && <button className="button subtle full-width" aria-label="LATR批量预标注" disabled={!clip || !activeCamera || busy || jobBusy || reconstructing || reviewing} onClick={() => void openLatrBatch()}><Icon name="layers" size={15}/>当前相机 · 批量预标注</button>}
    <button className="button ghost full-width" onClick={() => void openModels()}>配置 / 检查模型服务</button>
  </>;

  return <div className={`app-shell scene-app ${reviewing ? 'is-review' : ''}`}>
    <header className="topbar"><div className="brand"><span className="brand-mark"><Icon name="cube" size={26}/></span><span>Omni<span className="brand-light">Label</span><small>4D SCENE ANNOTATION</small></span><span className="version">4D</span></div><div className="workspace-breadcrumb"><span>工作空间</span><Icon name="chevron" size={12}/><strong>整场景多任务标注</strong></div><div className="topbar-actions"><span className={`save-status ${dirty || calibrationDirty ? 'is-dirty' : ''}`}><span className="status-dot"/>{saving ? '正在保存…' : dirty || calibrationDirty ? '有未保存修改' : '已同步至工作区'}</span><button className="button ghost" onClick={() => setModal('help')}><Icon name="help" size={16}/>使用说明</button><button className="button subtle" aria-label="回放标注clip" disabled={!clip || busy || jobBusy || reviewStarting} onClick={() => void startReview()}><Icon name="play" size={15}/>{reviewStarting ? '正在保存…' : reviewing ? '从头回放' : dirty || draftVector.length ? '保存并回放' : '回放标注'}</button><button className="button primary" aria-label="保存标注" disabled={reviewing || busy || jobBusy || Boolean(activeLanePreview) || (!dirty && !draftVector.length)} onClick={() => void save()}><Icon name="save" size={15}/>{saving ? '保存中' : '保存标注'}</button><span className="avatar">OL</span></div></header>
    <aside className="sidebar"><div className="nav-label">标注任务 <span>WORKSPACE</span></div><nav aria-label="标注任务">{tasks.map(t => <button key={t.id} className={`task-button ${task === t.id ? 'active' : ''}`} disabled={reviewing || reviewStarting} onClick={() => switchTask(t.id)}><Icon name={t.icon} size={19}/><span>{t.name}</span><small>{t.short}</small>{task === t.id && <i/>}</button>)}</nav><div className="sidebar-divider"/><div className="nav-label">自动化工具</div><button className="task-button model-link" disabled={reviewing} onClick={() => void openModels()}><Icon name="model" size={19}/><span>模型预标注</span><span className="small-badge">4</span></button><div className="sidebar-bottom"><div className="sensor-orbit"><Icon name="cube" size={25}/><span/><span/><span/></div><strong>一个场景 · 连续时空</strong><p>统一地图 / 轨迹关键帧<br/>3D 标签与七路图像实时联动</p><div className="local-indicator"><span className="status-dot"/>世界场景 · 持久化标注</div></div></aside>
    <main className="main-workspace">
      <section className="dataset-bar"><div className="dataset-select"><span className="dataset-icon"><Icon name="layers" size={20}/></span><div><label htmlFor="clip-select">当前场景 CLIP</label><select id="clip-select" aria-label="选择clip" value={clip?.id ?? ''} disabled={busy || jobBusy || reconstructing} onChange={e => {if (confirmDiscard()) void loadClip(e.target.value);}}>{!clip && <option value="">{loading ? '正在加载…' : '导入 clip 开始重建'}</option>}{clips.map(c => <option value={c.id} key={c.id}>{c.name}</option>)}</select></div><span className="demo-badge">{clip?.source === 'synthetic' || clip?.id === 'demo' ? '合成示例' : 'CLIP'}</span></div><div className="dataset-metrics"><span><strong>{clip?.frames.length ?? '—'}</strong> 帧</span><span><strong>{clip?.cameras.length ?? '—'}</strong> 相机</span><span><strong>{(sceneMode ? sceneFrame?.point_count : frame?.point_count)?.toLocaleString() ?? '—'}</strong> {reviewing && playbackCloud === 'frame' ? '当前扫描点' : sceneMode ? '场景点' : '源点'}</span></div><div className="dataset-actions"><button className="button subtle" aria-label="逐帧投影校准" disabled={reviewing || !clip || !frame || busy || jobBusy || reconstructing || projectionBusy} onClick={() => void openProjectionCalibration()}><Icon name="settings" size={15}/>逐帧校准</button><button className="button subtle" aria-label="全clip自动标定" disabled={reviewing || !clip || busy || jobBusy || reconstructing || projectionBusy} onClick={() => void openAutomaticCalibration()}><Icon name="settings" size={15}/>自动标定</button><button className="button subtle" aria-label="编辑标定外参" disabled={reviewing || !clip || busy || jobBusy || projectionBusy} onClick={() => {if (projectionDirty) {flash('请先保存或丢弃逐帧投影候选，再编辑全局外参。', true); return;} closeProjectionCalibration(); setCalibrationOpen(true); setRawProjection(true); setAnnotationProjection(true); setPlaying(false); setModelPromptMode(null);}}><Icon name="settings" size={15}/>外参校准</button><input ref={fileInput} type="file" accept=".json,.zip" hidden onChange={e => {void importFile(e.target.files?.[0]); e.target.value = '';}}/><button className="button ghost" aria-label="导入clip" disabled={reviewing || busy || jobBusy || reconstructing} onClick={() => fileInput.current?.click()}><Icon name="upload" size={15}/>导入</button><button className="button subtle" aria-label="导出clip" disabled={!clip || busy || jobBusy || reconstructing} onClick={exportClip}><Icon name="download" size={15}/>导出场景</button></div></section>
      <div className="workspace-grid scene-layout">
         <aside className="camera-rail" aria-label="左侧相机标注投影"><div className="camera-rail-heading"><h2><Icon name="image" size={16}/>相机标注投影 <span>{clip?.cameras.length ?? 0}</span></h2><div className="camera-overlay-toggles"><button className={`toggle-button ${annotationProjection ? 'on' : ''}`} aria-label="切换标注投影" onClick={() => setAnnotationProjection(!annotationProjection)}>标注结果<i className="toggle-switch"/></button><button className={`toggle-button ${rawProjection ? 'on' : ''}`} aria-label="切换点云投影" onClick={() => setRawProjection(!rawProjection)}>原始点云<i className="toggle-switch"/></button></div><p>框、车道线与矢量地图随当前时刻投影</p>{modelPromptMode && <div className="seed-banner">SAM2 {modelPromptMode === 'box' ? '拖动提示框' : negativePrompt ? '负提示点' : '正提示点'} · {activeCamera}<button onClick={() => setModelPromptMode(null)}>退出</button></div>}</div><div className="camera-grid camera-scroll" data-testid="camera-scroll">{cameraCards()}{!clip && <div className="camera-placeholder">导入场景后显示同步相机</div>}{clip && !clip.cameras.length && <div className="camera-placeholder">此场景未包含相机图像</div>}</div>{task === 'image' && <div className="image-draw-footer"><span>{draftPolygon.length} 个区域顶点</span><button className="button subtle" disabled={draftPolygon.length < 3} onClick={finishPolygon}>完成多边形</button><button className="button ghost" onClick={() => setDraftPolygon([])}>取消</button><button className="button ghost" aria-label="导出语义掩码" disabled={!clip || busy || !activeCamera} onClick={() => clip && frame && activeCamera && void exportFile(`/clips/${clip.id}/frames/${frame.id}/masks/${activeCamera}.zip`, `${frame.id}-${activeCamera}-mask.zip`)}>导出掩码</button></div>}</aside>
        <section className="view-workspace"><div className="section-header"><div><span className="section-eyebrow">{taskInfo.short} / {sceneMode ? 'WORLD SCENE' : 'LIDAR FRAME'}</span><h1>{taskInfo.name}<span>{taskInfo.subtitle}</span></h1></div><span className="coordinate-tag"><span/>{sceneMode ? 'WORLD' : 'LiDAR'} · 米</span></div>
          {reviewing && <section className="clip-playback" data-testid="clip-playback" data-playing={playing} data-frame-id={frame?.id}><div className="playback-heading"><strong><Icon name="play" size={14}/>标注回放</strong><span role="status">{buffering ? '缓冲中 · 保持当前帧' : playing ? '正在播放' : playbackEnded ? '回放完成' : '已暂停'}</span><button className="button ghost" aria-label="退出回放" onClick={exitReview}>退出回放</button></div><div className="playback-options"><label>速度<select aria-label="回放速度" value={playbackSpeed} onChange={e => setPlaybackSpeed(Number(e.target.value))}><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1">1×</option><option value="2">2×</option></select></label><label><input type="checkbox" aria-label="循环回放" checked={playbackLoop} onChange={e => setPlaybackLoop(e.target.checked)}/>循环</label><label>点云<select aria-label="回放点云视图" value={playbackCloud} onChange={e => setPlaybackCloud(e.target.value as 'frame' | 'scene')}><option value="frame">当前扫描 · 跟随车辆</option><option value="scene" disabled={!sceneMap}>{sceneMap ? '整场景地图' : '整场景地图（需重建）'}</option></select></label><span>只读 · 标注投影已开启</span></div></section>}
          {!reviewing && <div className="scene-space-bar"><div className="segmented"><button aria-label="整场景4D" className={sceneMode ? 'active' : ''} onClick={() => switchSpace(true)}>整场景 4D</button><button aria-label="单帧点云" className={!sceneMode ? 'active' : ''} onClick={() => switchSpace(false)}>单帧点云</button></div><span>{sceneMode ? '地图在切帧时保持不变，目标沿轨迹移动' : '源点级标签 / 当前帧独立标注'}</span></div>}
          {!reviewing && task === 'detection' && <div className="box-edit-guide" data-testid="box-edit-guide"><Icon name="cube" size={18}/><div><strong>{selectedBox ? '拖动框面的彩色方块，直接调整长 / 宽 / 高' : '先点击点云中的目标框，或在右侧列表选中目标'}</strong><p>{selectedBox ? '红色改长、绿色改宽、蓝色改高；中心 ↔ 移动位置。右侧数值与相机投影实时更新。' : '选中后直接显示尺寸控制柄，无需手动输入尺寸。新建框也可立即拖动调整。'}{selectedBox && view === 'top' && ' 俯视图只能调整长宽，调整高度请切换「透视 3D」。'}</p></div>{selectedBox && <button className="button subtle" aria-label="聚焦选中目标" onClick={() => setFocusSelection(n => n + 1)}><Icon name="expand" size={14}/>聚焦目标</button>}</div>}
          {!reviewing && vectorTask && <div className="vector-actions"><button className="button primary" aria-label={task === 'lane' ? '开始绘制车道线' : '开始绘制地图矢量'} disabled={geometryEditBlocked} onClick={startVector}><Icon name="pen" size={15}/>{tool === 'vector' ? '继续绘制' : task === 'lane' ? '开始绘制车道线' : '开始绘制矢量'}</button>{clip?.id !== 'demo-ground-detail' && clips.some(c => c.id === 'demo-ground-detail') && <button className="button ghost" aria-label="加载道路标线示例" disabled={busy || reconstructing} onClick={() => {if (confirmDiscard()) void loadClip('demo-ground-detail');}}>标线示例</button>}</div>}
          {calibrationPreview && <div className="calibration-preview-tag" role="status">{calibrationLidarPreview ? 'LiDAR 外参草稿 · 当前帧点云与相机投影实时预览，保存后重建完整地图' : '相机外参草稿 · 标注和原始点云投影实时更新'}</div>}
          <div className="scene-panel"><div className="scene-toolbar"><div className="segmented"><button className={view === 'perspective' ? 'active' : ''} onClick={() => setView('perspective')}>透视 3D</button><button className={view === 'top' ? 'active' : ''} onClick={() => setView('top')}>俯视 BEV</button></div><div className="tool-buttons"><button className={`icon-button ${tool === 'select' ? 'active' : ''}`} aria-label="选择工具" onClick={() => setTool('select')}><Icon name="select" size={17}/></button>{task === 'detection' && <button className={`icon-button ${tool === 'box' ? 'active' : ''}`} aria-label="新建3D框" disabled={geometryEditBlocked} onClick={() => {setTool('box'); setShowBoxes(true); setPlaying(false);}}><Icon name="plus" size={19}/></button>}{['lane', 'vector'].includes(task) && <button className={`icon-button ${tool === 'vector' ? 'active' : ''}`} aria-label="绘制矢量" disabled={geometryEditBlocked} onClick={startVector}><Icon name="pen" size={17}/></button>}{task === 'semantic' && !sceneMode && <button className={`icon-button ${tool === 'brush' ? 'active' : ''}`} aria-label="点云语义笔刷" onClick={() => {setTool('brush'); setPlaying(false);}}><Icon name="brush" size={17}/></button>}</div><div className="scene-toolbar-right"><button className={`toggle-button ${showBoxes ? 'on' : ''}`} onClick={() => setShowBoxes(!showBoxes)}><Icon name="cube" size={14}/>目标框</button><button className={`toggle-button ${showVectors ? 'on' : ''}`} onClick={() => setShowVectors(!showVectors)}><Icon name="lane" size={14}/>矢量</button></div></div>
            <div className="surface-toolbar"><label>点云着色<select aria-label="点云着色" value={pointColorMode} onChange={e => setPointColorMode(e.target.value as typeof pointColorMode)}><option value="intensity">反射强度 · 查看标线</option><option value="height">高度</option><option value="semantic">语义类别</option></select></label><label className="ground-toggle"><input type="checkbox" aria-label="仅看地面" checked={groundOnly} onChange={e => setGroundOnly(e.target.checked)}/>仅看地面</label><span>{surfaceInfo.count.toLocaleString()} 点</span>{pointColorMode === 'intensity' && <span className="intensity-legend">低<i/>高</span>}</div>
            {groundOnly && surfaceInfo.count === 0 && !reconstructing && <p className="surface-hint" role="status">当前地面高度范围内没有点，请调整右侧「地面参考高度 / 高度半带宽」，或关闭「仅看地面」。</p>}
            {pointColorMode === 'intensity' && surfaceInfo.count > 0 && !surfaceInfo.contrast && <p className="surface-hint" role="status">这些点没有可区分的反射强度。仅有 XYZ 或固定强度无法显示路面油漆纹理；请导入包含真实强度的 XYZI 点云，并参考左侧原图标注。</p>}
            {!reviewing && selectedBox && task === 'detection' && <div className="transform-toolbar"><span>鼠标直接编辑</span>{(['translate', 'scale', 'rotate'] as const).map((mode, i) => <button key={mode} className={transformMode === mode ? 'active' : ''} aria-label={['平移标注框', '缩放标注框', '旋转标注框'][i]} disabled={geometryEditBlocked} onClick={() => {setTransformMode(mode); setTool('select');}}>{['平移', '拖动改尺寸', '旋转'][i]}</button>)}<small>{sceneMode ? currentIsKeyframe ? '当前关键帧' : '拖动即创建关键帧' : '当前帧'}</small></div>}
            <div className={`scene-canvas ${editBlocked && !reviewing && !projectionOpen ? 'scene-disabled' : ''}`}>
              <PointCloudView calibrationPicking={projectionOpen && projectionPickMode === '3d' && !projectionBusy} onCalibrationPoint={point => {if (!projectionBusy && frame) setProjectionPickedPoint(sceneMode ? transformPoint(point, inverseRigid(worldFromLidar(frame, coordinateMap))) : point);}} calibrationPoints={projectionMarkers} frame={sceneMode ? sceneFrame : frame} binaryPoints={displayedPointBuffer} cloudBounds={displayedPointBuffer ? sceneMap?.bounds : undefined} intensityRangeHint={displayedPointBuffer ? sceneMap?.intensity_range : undefined} annotations={renderedAnnotations} selectedId={selectedId} onSelect={selectAnnotation} onCreateBox={createBox} onUpdateBox={updateBoxById} focusSelection={focusSelection} transformMode={transformMode} editable={!geometryEditBlocked && task === 'detection'} vectorEditable={!geometryEditBlocked && !calibrationOpen && vectorTask} onUpdateVectorPoint={updateVectorPoint} sceneMode={sceneMode} trajectory={sceneMode ? sceneMap?.trajectory : undefined} activeFrameId={frame?.id} followPosition={reviewing && playbackCloud === 'frame' && frame ? worldFromLidar(frame, coordinateMap).slice(0, 3).map(row => row[3]) as Vec3 : undefined} autoFitReference={reviewing && playbackCloud === 'frame' ? reviewFitReference.current ?? frame?.points : displayedPointBuffer ?? frame?.points} onTransformStart={() => setPlaying(false)} onAddVectorPoint={point => {if (!editBlocked) setDraftVector(p => [...p, point]);}} onPaint={indices => {if (!sceneMode && indices.length) editFrame(a => ({...a, point_labels: {...a.point_labels, ...Object.fromEntries(indices.map(i => [String(i), classId]))}}));}} tool={geometryEditBlocked ? 'select' : tool} view={view} groundZ={groundZ} voxels={!sceneMode && task === 'occupancy' && occupancy?.frameId === frame?.id ? occupancy?.voxels : undefined} voxelSize={occupancy?.size ?? voxelSize} voxelDimensions={occupancy?.dimensions} showFree={showFree} showBoxes={showBoxes} showVectors={showVectors} colorMode={pointColorMode} groundOnly={groundOnly} groundBand={groundBand} drawOnGround={vectorTask} draftVector={draftVector} brushRadius={brushRadius}/>
              <div className="scene-top-info"><span className="live-pill"><i/>{sceneMode ? 'WORLD SCENE' : `FRAME ${String(frameIndex + 1).padStart(4, '0')}`}</span><span>{sceneMode && sceneMap?.rendered_point_count != null && sceneMap.rendered_point_count !== sceneMap.point_count ? `${sceneMap.rendered_point_count.toLocaleString()} / ${sceneMap.point_count.toLocaleString()}` : (sceneMode ? sceneFrame?.point_count : frame?.point_count)?.toLocaleString() ?? 0} POINTS</span></div>
              {loading && <div className="loading-overlay"><span className="spinner"/>正在加载多模态数据…</div>}
              {!clip && !loading && <div className="empty-scene"><Icon name="layers" size={38}/><h2>{clip ? '场景地图尚未生成' : '从一个 clip 开始'}</h2><p>{reconstruction?.error ?? '将整段点云配准、融合为可标注场景'}</p>{clip && <button className="button primary" onClick={() => void rebuildScene()}>重建整个场景</button>}</div>}
              {vectorTask && tool === 'vector' && <div className="drawing-bar"><span>{draftVector.length} 个{sceneMode ? '世界' : 'LiDAR'}顶点</span><button disabled={geometryEditBlocked || draftVector.length < (closedDraft ? 3 : 2)} onClick={finishVector}>完成绘制</button><button disabled={!draftVector.length} onClick={() => setDraftVector(p => p.slice(0, -1))}>撤回一点</button><button onClick={() => {setDraftVector([]); setTool('select');}}>取消</button></div>}
            </div><div className="scene-footer"><span><span className="keycap">左键</span>{reviewing ? '转动视角 / 选择查看' : selectedBox && task === 'detection' ? '拖动控制柄编辑框' : selectedVector && vectorTask ? '拖动顶点编辑车道线' : tool === 'vector' ? '添加矢量顶点' : tool === 'brush' ? '语义涂刷' : '旋转 / 选择'}<span className="keycap">右键</span>平移<span className="keycap">滚轮</span>缩放</span><span>{sceneMode ? '统一世界坐标 · 时间连续' : '当前 LiDAR 坐标'}</span></div></div>
          <div className="scene-summary"><span><Icon name="layers" size={14}/>{sceneMap ? `${sceneMap.frame_count} 帧已融合 · ${sceneMap.point_count.toLocaleString()} 个静态地图点` : sceneMode && frame ? (reconstructing ? '后台重建中 · 当前扫描已转为世界坐标，可继续标注' : '当前扫描 · 世界坐标') : '等待场景重建'}</span><span>{scene?.tracks.length ?? 0} 条轨迹 · {scene?.vectors.length ?? 0} 个全局矢量</span>{sceneDirty && <b>场景修改保留中，切帧不会丢失</b>}</div>
          {proposal && <div className="inline-proposal"><Icon name="model" size={18}/><div><strong>{modelNames[proposal.model] ?? proposal.model}{proposal.weightVariant ? ` · ${proposal.weightVariant.toUpperCase()}` : ''} 候选已生成</strong><p>{proposal.annotations.boxes.length} 个框 / {proposal.annotations.vectors.length} 条三维矢量 / {proposal.annotations.image_polygons.length} 个图像区域，需人工审核</p></div><button className="button primary" disabled={busy || jobBusy} onClick={() => void applyProposal()}>应用候选标注</button><button className="button ghost" disabled={busy || jobBusy} onClick={() => setProposal(null)}>丢弃</button></div>}
        </section>
        <aside className={`inspector ${calibrationOpen || projectionOpen ? 'calibration-active' : ''}`}>{projectionOpen && clip && frame && <ProjectionCalibrationEditor clipId={clip.id} frameId={frame.id} cameras={clip.cameras} cameraId={activeCamera} disabled={busy || jobBusy || reconstructing} pickedPoint={projectionPickedPoint} pickedPixel={projectionPickedPixel} onCameraChange={id => {setActiveCamera(id); setProjectionPreview(null); setProjectionPickMode(null); consumeProjectionPick();}} onPickModeChange={setProjectionPickMode} onConsumedPick={consumeProjectionPick} onPreview={previewProjection} onSaved={savedProjection} onClose={closeProjectionCalibration} onDirtyChange={setProjectionDirty} onBusyChange={setProjectionBusy} onError={message => flash(message, true)} onAnchorsChange={changeProjectionAnchors}/>} {calibrationOpen && clip && frame && <CalibrationEditor clip={clip} frameId={frame.id} disabled={busy || jobBusy || reconstructing} onPreview={previewCalibration} onSaved={calibrationSaved} onClose={() => setCalibrationOpen(false)} onError={message => flash(message, true)} onDirtyChange={setCalibrationDirty}/>}<div className="inspector-heading"><h2>{reviewing ? '标注结果' : '标注属性'}</h2><span className="count-badge">{renderedAnnotations.boxes.length + renderedAnnotations.vectors.length + annotations.image_polygons.length}</span><button className="icon-button" aria-label="标注帮助" onClick={() => setModal('help')}><Icon name="settings" size={15}/></button></div><div className="inspector-body">
          {vectorTask && !reviewing && <section className="inspector-section"><div className="mini-heading"><span>车道线融合</span><small>{sceneMode ? '整个场景' : '当前帧'}</small></div>{laneMergeOpen && clip ? <LaneMergePanel clipId={clip.id} scopeKey={laneScopeKey} vectors={laneVectors} sceneRevision={sceneMode ? scene?.revision : undefined} disabled={editBlocked || calibrationOpen} onPreview={previewLanes} onApply={applyLaneMerge} onClose={() => {setLaneMergeOpen(false); setLanePreview(null);}} onError={message => flash(message, true)}/> : <button className="button subtle full-width" aria-label="融合重复车道线" disabled={editBlocked || calibrationOpen || laneVectors.length < 2} onClick={() => {if (draftVector.length || draftPolygon.length) {flash('请先完成或取消绘制，再融合车道线', true); return;} setLaneMergeOpen(true); setTool('select'); setPlaying(false);}}>融合重复车道线</button>}{laneMergeUndo?.scopeKey === laneScopeKey && laneMergeUndo.after === laneVectors && <button className="button ghost full-width" aria-label="撤销本次融合" disabled={geometryEditBlocked} onClick={undoLaneMerge}>撤销本次融合</button>}</section>}
          <section className="inspector-section"><div className="mini-heading"><span>类别标签</span><small>{sceneMode ? 'WORLD SCENE' : 'FRAME LABEL'}</small></div><div className="class-list">{classes.map(c => <button key={c.id} className={`class-option ${classId === c.id ? 'selected' : ''}`} onClick={() => {setClassId(c.id); if (selectedBox && task === 'detection') updateBox({label: c.id});}}><i style={{background: c.color}}/><span>{c.name}</span><small>{c.id}</small>{classId === c.id && <Icon name="check" size={13}/>}</button>)}</div></section>
          {selectedBox ? <section className="inspector-section"><div className="mini-heading"><span>{sceneMode ? '4D 轨迹 / 当前状态' : '目标属性'}</span><span className="text-cyan">{sceneMode ? 'WORLD' : 'LiDAR'}</span></div><label className="field-label">轨迹 ID<input aria-label="轨迹ID" value={selectedBox.track_id} disabled={geometryEditBlocked} onChange={e => updateBox({track_id: e.target.value})}/></label>{selectedTrack && <><label className="field-label">时序模式<select aria-label="轨迹时序模式" value={selectedTrack.mode} disabled={geometryEditBlocked} onChange={e => {const mode = e.target.value as SceneTrack['mode']; editScene(s => ({...s, tracks: s.tracks.map(t => t.id === selectedId ? {...t, mode, keyframes: mode === 'static' && frame ? [{frame_id: frame.id, center: selectedBox.center, size: selectedBox.size, yaw: selectedBox.yaw}] : t.keyframes} : t)}));}}><option value="dynamic">动态轨迹 · 关键帧插值</option><option value="static">静态目标 · 全场景共享</option></select></label><div className="keyframe-actions"><span>{selectedTrack.keyframes.length} 个关键帧 · {currentIsKeyframe ? '当前已设关键帧' : '当前为插值状态'}</span><button className="button subtle" aria-label="添加当前关键帧" disabled={geometryEditBlocked} onClick={() => updateBox({})}>设为关键帧</button><button className="button ghost" disabled={geometryEditBlocked || !currentIsKeyframe || selectedTrack.keyframes.length < 2} onClick={() => editScene(s => ({...s, tracks: s.tracks.map(t => t.id === selectedId ? {...t, keyframes: t.keyframes.filter(k => k.frame_id !== frame?.id)} : t)}))}>删除本关键帧</button></div></>}
            <div className="coordinate-fields">{(['center', 'size'] as const).map(key => <div key={key}><label>{key === 'center' ? `中心位置 · ${sceneMode ? '世界' : 'LiDAR'}米` : '尺寸 · 米（拖动实时更新）'}</label><div className="numeric-triplet">{(key === 'center' ? ['X', 'Y', 'Z'] : ['长', '宽', '高']).map((axis, i) => <label key={axis}><span>{axis}</span><input type="number" step="0.1" min={key === 'size' ? '0.01' : undefined} aria-label={`${key === 'center' ? '位置' : '尺寸'}${axis}`} value={Number(selectedBox[key][i].toFixed(3))} disabled={geometryEditBlocked} onChange={e => {if (!Number.isFinite(e.target.valueAsNumber) || (key === 'size' && e.target.valueAsNumber <= 0)) return; const values = [...selectedBox[key]] as Vec3; values[i] = e.target.valueAsNumber; updateBox({[key]: values});}}/></label>)}</div></div>)}</div><label className="field-label">航向角 · 度<input type="number" step="1" aria-label="航向角" value={Number((selectedBox.yaw * 180 / Math.PI).toFixed(2))} disabled={geometryEditBlocked} onChange={e => {if (Number.isFinite(e.target.valueAsNumber)) updateBox({yaw: e.target.valueAsNumber * Math.PI / 180});}}/></label><button className="button danger full-width" disabled={geometryEditBlocked} onClick={deleteSelected}><Icon name="trash" size={14}/>{sceneMode ? '删除选中轨迹（所有时刻）' : '删除选中目标'}</button></section> : selectedVector || selectedPolygon ? <section className="inspector-section"><div className="mini-heading"><span>{selectedVector && sceneMode ? '全场景矢量' : '选中要素'}</span></div><label className="field-label">要素类别<input aria-label="要素类别" value={selectedVector?.label ?? selectedPolygon?.label ?? ''} disabled={geometryEditBlocked} onChange={e => updateElement({label: e.target.value})}/></label>{selectedVector && <label className="closed-field"><input type="checkbox" checked={selectedVector.closed} disabled={geometryEditBlocked || selectedVector.points.length < 3} onChange={e => updateElement({closed: e.target.checked})}/>闭合面要素</label>}<div className="vertex-list">{(selectedVector?.points ?? selectedPolygon?.points ?? []).map((point, index) => <div className="vertex-row" key={index}><span>{String(index + 1).padStart(2, '0')}</span>{point.map((value, axis) => <label key={axis}><small>{['X', 'Y', 'Z'][axis]}</small><input type="number" step="0.1" aria-label={`顶点${index + 1}${['X', 'Y', 'Z'][axis]}`} value={Number(value.toFixed(3))} disabled={geometryEditBlocked} onChange={e => {if (!Number.isFinite(e.target.valueAsNumber)) return; const points = (selectedVector?.points ?? selectedPolygon!.points).map((p, i) => i === index ? p.map((v, j) => j === axis ? e.target.valueAsNumber : v) : p); updateElement({points});}}/></label>)}</div>)}</div><button className="button danger full-width" disabled={geometryEditBlocked} onClick={deleteSelected}><Icon name="trash" size={14}/>删除选中要素</button></section> : <section className="inspector-section selection-empty"><Icon name={taskInfo.icon} size={29}/><p>{reviewing ? '选择轨迹或矢量查看属性' : sceneMode ? '选择场景中的轨迹或矢量以编辑' : '选择目标或开启绘制工具'}</p>{task === 'detection' && <button className="button subtle full-width" disabled={!frame || editBlocked} onClick={() => {setTool('box'); setShowBoxes(true); setPlaying(false);}}><Icon name="plus" size={15}/>新建 3D 目标框</button>}</section>}
          <section className="inspector-section object-section"><div className="mini-heading"><span>{sceneMode ? '场景轨迹 / 矢量' : '当前帧标注'}</span><small>{renderedAnnotations.boxes.length} / {renderedAnnotations.vectors.length}</small></div><div className="object-list">{renderedAnnotations.boxes.map((box, i) => <button key={box.id} className={`object-row ${selectedId === box.id ? 'selected' : ''}`} onClick={() => {selectAnnotation(box.id); setClassId(box.label);}}><Icon name="cube" size={14}/><span>{box.label}<small>{box.track_id}</small></span><em>{String(i + 1).padStart(2, '0')}</em></button>)}{renderedAnnotations.vectors.map(v => <button key={v.id} className={`object-row ${selectedId === v.id ? 'selected' : ''}`} onClick={() => selectAnnotation(v.id)}><Icon name="lane" size={14}/><span>{v.label}<small>{v.points.length} 个顶点</small></span></button>)}{annotations.image_polygons.map(p => <button key={p.id} className={`object-row ${selectedId === p.id ? 'selected' : ''}`} onClick={() => {setSelectedId(p.id); chooseCamera(p.camera_id);}}><Icon name="image" size={14}/><span>{p.label}<small>{p.camera_id} · {p.points.length} 点</small></span></button>)}</div>{!sceneMode && <p className="point-label-count">{Object.keys(annotations.point_labels).length.toLocaleString()} 个源点已有语义标签</p>}</section>
          <section className="inspector-section reconstruction-panel"><div className="mini-heading"><span>整场景重建</span><small>CLIP → WORLD</small></div><label className="field-label">配准方式<select aria-label="场景配准方式" value={registration} onChange={e => setRegistration(e.target.value as 'poses' | 'icp')}><option value="poses">使用输入车辆位姿</option><option value="icp">ICP 点云配准修正</option></select></label><label className="field-label">融合方式<select aria-label="场景融合方式" value={fusionMode} onChange={e => {const next = e.target.value as typeof fusionMode; setFusionMode(next); setPreserveDetail(next === 'detail');}}><option value="full">全量实测点 · 不降采样</option><option value="detail">标线反射细节</option><option value="surface">表面保真 · 体素代表点</option><option value="centroid">质心平滑</option></select></label><label className="field-label">地图体素 · 米<select aria-label="体素分辨率" value={voxelSize} onChange={e => setVoxelSize(Number(e.target.value))}><option value="0.1">0.1 m</option><option value="0.2">0.2 m</option><option value="0.3">0.3 m</option><option value="0.5">0.5 m</option><option value="1">1.0 m</option></select></label>{fusionMode === 'detail' && <label className="field-label">细节体素 · 米<select aria-label="细节体素分辨率" value={detailVoxelSize} onChange={e => setDetailVoxelSize(Number(e.target.value))}><option value="0.05">0.05 m · 标线优先</option><option value="0.1">0.10 m</option><option value="0.2">0.20 m · 较少点</option></select></label>}<label className="closed-field"><input type="checkbox" checked={excludeDynamic} onChange={e => setExcludeDynamic(e.target.checked)}/>剔除标注动态目标内点</label><button className="button subtle full-width" aria-label="重建整个场景" disabled={reviewing || !clip || busy || jobBusy || reconstructing} onClick={() => void rebuildScene()}><Icon name="layers" size={15}/>{reconstructing ? '整场景重建中…' : '重建整个场景'}</button><p className="micro-copy">上传后自动融合全 clip。地图保持固定，时间轴控制目标轨迹与同步相机。全量模式保留全部实测静态点；细节模式保留标线强度，表面模式避免质心偏离实测表面。</p>{reconstruction && <div className={`job-status ${reconstruction.status}`}><span className="status-dot"/><strong>{reconstruction.status === 'completed' ? '场景重建完成' : reconstruction.status === 'failed' ? '重建失败' : '正在重建'}</strong><small>{reconstruction.error ?? registration}</small></div>}{sceneMap && <p className="micro-copy">当前地图：有效体素 {(sceneMap.effective_voxel_size ?? sceneMap.voxel_size).toFixed(2)} m · {sceneMap.fusion_mode === 'full' ? '全量实测点，无体素降采样' : sceneMap.fusion_mode === 'surface' ? '表面实测代表点' : sceneMap.preserve_detail ? '保留实测强反射点' : '体素均值融合'}。</p>}{sceneMap && <button className="button ghost full-width" aria-label="导出场景地图" onClick={() => download(`${clip?.id}-scene-map.json`, JSON.stringify(sceneMap))}><Icon name="download" size={13}/>导出场景地图与轨迹</button>}</section>
          <section className="inspector-section automation-section"><div className="mini-heading"><span>模型预标注</span><Icon name="model" size={15}/></div>{modelControls}</section>
          <section className="inspector-section"><div className="mini-heading"><span>辅助设置</span><small>{sceneMode ? 'WORLD' : 'LiDAR'}</small></div>{(groundOnly || ['detection', 'lane', 'vector'].includes(task)) && <label className="field-label">地面参考高度 · 米<input type="number" aria-label="地面参考高度" step="0.1" value={groundZ} onChange={e => {if (Number.isFinite(e.target.valueAsNumber)) (sceneMode ? setSceneGroundZ : setFrameGroundZ)(e.target.valueAsNumber);}}/></label>}{groundOnly && <label className="field-label">地面高度半带宽 · 米<input type="number" min="0.05" max="10" step="0.05" aria-label="地面高度半带宽" value={groundBand} onChange={e => {if (Number.isFinite(e.target.valueAsNumber) && e.target.valueAsNumber >= 0.05 && e.target.valueAsNumber <= 10) setGroundBand(e.target.valueAsNumber);}}/><small>显示地面高度 ± 此值内的点；坡道需调节高度或关闭筛选。</small></label>}{task === 'semantic' && <><label className="field-label">笔刷半径 · 米<span>{brushRadius.toFixed(1)}</span><input type="range" min="0.3" max="6" step="0.1" value={brushRadius} onChange={e => setBrushRadius(Number(e.target.value))}/></label><button className="button subtle full-width" disabled={!clip || busy || jobBusy} onClick={() => void runJob('ground-segmentation')}>运行 CPU 地面辅助</button></>}{task === 'occupancy' && <><button className="button subtle full-width" disabled={!clip || busy || jobBusy} onClick={() => void runJob('occupancy')}>转换当前帧观测 OCC</button><div className="occupancy-legend"><label className="closed-field"><input type="checkbox" checked={showFree} onChange={e => setShowFree(e.target.checked)}/>显示观测自由空间</label><p><span>占据 {occupancy?.voxels.filter(v => v.state === 'occupied').length.toLocaleString() ?? 0}</span><span>自由 {occupancy?.voxels.filter(v => v.state === 'free').length.toLocaleString() ?? 0}</span></p><small>未观测空间为未知；模型 OCC 候选需审核。</small></div></>}{job && <div className={`job-status ${job.status}`}><span className="status-dot"/><strong>{job.status === 'completed' ? '处理完成' : job.status === 'failed' ? '处理失败' : '后台计算中'}</strong><small>{job.error ?? job.kind}</small>{job.status === 'completed' && job.result && <button aria-label="导出计算结果" onClick={() => download(`${clip?.id}-${job.kind}.json`, JSON.stringify(job.result))}><Icon name="download" size={14}/></button>}</div>}</section>
        </div><div className="inspector-bottom"><Icon name="check" size={13}/><span>场景 v{scene?.revision ?? 0} / 帧 v{annotations.revision}</span>{dirty && <button className="draft-download" onClick={() => download(`${clip?.id}-draft.json`, JSON.stringify({scene_annotations: scene, frame_id: frame?.id, annotations}, null, 2))}><Icon name="download" size={10}/>备份草稿</button>}</div></aside>
      </div>
      <section className="timeline" aria-label="clip时间轴"><div className="timeline-top"><div className="playback-controls"><button className="icon-button" aria-label="上一帧" disabled={!clip || frameIndex === 0 || busy || jobBusy} onClick={() => void moveFrame(frameIndex - 1)}><Icon name="prev" size={16}/></button><button className={`play-button ${playing ? 'playing' : ''}`} aria-label={playing ? '暂停' : '播放'} disabled={!clip || (!playing && (busy || jobBusy || reviewStarting))} onClick={togglePlayback}><Icon name={playing ? 'pause' : 'play'} size={15}/></button><button className="icon-button" aria-label="下一帧" disabled={!clip || frameIndex >= clip.frames.length - 1 || busy || jobBusy} onClick={() => void moveFrame(frameIndex + 1)}><Icon name="next" size={16}/></button><span className="frame-position"><strong>{String(frameIndex + 1).padStart(4, '0')}</strong><span>/ {String(clip?.frames.length ?? 0).padStart(4, '0')}</span></span></div><span className="timeline-title">4D SCENE TIMELINE <span>● 关键帧 / 区间插值</span></span><span className="timestamp">t = {frame?.timestamp.toFixed(3) ?? '0.000'} s <span className="timebase">{reviewing ? `${playbackSpeed}×` : `${clip?.fps ?? 0} Hz`}</span></span></div><div className="timeline-track"><div className="timeline-ruler">{[0, 1, 2, 3, 4].map(i => <span key={i}>{((clip && clip.frames.length > 1 ? clip.frames.at(-1)!.timestamp - clip.frames[0].timestamp : 0) * i / 4).toFixed(1)}s</span>)}</div><div className="frame-cells">{clip?.frames.map((f, i) => <button key={f.id} className={`${frameIndex === i ? 'current' : ''} ${selectedTrack?.keyframes.some(k => k.frame_id === f.id) ? 'has-keyframe' : ''}`} aria-label={`跳转到第${i + 1}帧`} disabled={busy || jobBusy} title={`${f.id} · ${f.timestamp.toFixed(3)}s`} onClick={() => void moveFrame(i)}><i/>{selectedTrack?.keyframes.some(k => k.frame_id === f.id) ? <b>◆</b> : null}{frameIndex === i && <span>{String(i + 1).padStart(2, '0')}</span>}</button>)}</div></div></section>
    </main>
    {latrBatchOpen && clip && <LatrBatchPanel key={clip.id} clip={clip} cameraId={activeCamera} label={workerLabel} scoreThreshold={scoreThreshold} weightVariant={latrWeight} disabled={busy || jobBusy || reconstructing || reviewing || calibrationDirty || projectionDirty} modelAvailable={Boolean(modelList?.find(model => model.id === 'latr')?.available)} onApply={applyLatrBatch} onClose={() => setLatrBatchOpen(false)} onError={message => flash(message, true)}/>}
    {automaticCalibrationOpen && clip && <AutomaticCalibrationPanel clip={clip} frameId={frame?.id} cameraId={activeCamera} disabled={busy || jobBusy || reconstructing} onSaved={automaticCalibrationSaved} onInspect={inspectAutomaticCalibration} onClose={() => setAutomaticCalibrationOpen(false)}/>}
    {notice && <div className={`toast ${notice.error ? 'error' : ''}`} role={notice.error ? 'alert' : 'status'}><Icon name={notice.error ? 'help' : 'check'} size={18}/><span>{notice.message}</span><button className="icon-button" aria-label="关闭提示" onClick={() => setNotice(null)}><Icon name="close" size={15}/></button></div>}
    {expandedCamera && <div className="modal-backdrop" onClick={() => setExpandedCamera(null)}><div className="expanded-camera" onClick={e => e.stopPropagation()}>{cameraCards(true)}{task === 'image' && <div className="image-draw-footer"><span>{draftPolygon.length} 个区域顶点</span><button className="button primary" disabled={draftPolygon.length < 3} onClick={finishPolygon}>完成多边形</button></div>}</div></div>}
    {modal && <div className="modal-backdrop" onClick={() => setModal(null)}><div className="modal" role="dialog" aria-modal="true" aria-label={modal === 'models' ? '模型与预标注' : '使用说明'} onClick={e => e.stopPropagation()}>
      <div className="modal-heading"><div><span className="section-eyebrow">{modal === 'models' ? 'MODEL INFERENCE' : '4D WORKFLOW'}</span><h2>{modal === 'models' ? '真实模型服务与预标注' : '4D 整场景标注工作流'}</h2></div><button className="icon-button" aria-label="关闭对话框" onClick={() => setModal(null)}><Icon name="close"/></button></div>
      {modal === 'models' ? <>
        <p className="modal-intro">SAM2、LATR、MMDetection3D、OpenOccupancy 使用各自的实际模型运行时。服务状态、依赖和权重检查通过后才可执行。推理失败不会以几何结果冒充模型结果。</p>
        <div className="model-list">{Object.keys(modelNames).map(id => {const model = modelList?.find(m => m.id === id); const connection = connections.find(c => c.model_id === id); return <div className="model-card" key={id}><div><Icon name="model" size={19}/><h3>{modelNames[id].split(' · ')[0]}</h3><span className={model?.available ? 'available-tag' : 'unavailable-tag'}>{model?.available ? '推理就绪' : model?.configured ? '未就绪' : '未配置'}</span></div><p>{modelStatus(model)}</p><label className="field-label">推理服务完整地址<input aria-label={`${id}服务地址`} placeholder="https://gpu.example/predict" value={connection?.endpoint ?? ''} onChange={e => setConnections(rows => [...rows.filter(c => c.model_id !== id), {model_id: id, endpoint: e.target.value}])}/></label><button className="button subtle full-width" disabled={connectionBusy} onClick={() => void saveConnection(id)}>保存并检查连接</button><button className="button ghost full-width" onClick={() => {setWorkerModel(id); setModal(null); if (id === 'sam2') {setModelPromptMode('point'); setNegativePrompt(false);} if (id === 'latr') {setWorkerLabel('lane'); setTask('lane');}}}>选择此模型</button>{model?.url && <a href={model.url} target="_blank" rel="noreferrer">官方项目 ↗</a>}</div>;})}</div>
        <div className="worker-panel">{modelControls}</div>
        {proposal && <div className="proposal-review"><strong>候选标注等待审核</strong><p>{proposal.model}{proposal.weightVariant ? `/${proposal.weightVariant}` : ''} · {proposal.frameId} · {proposal.annotations.boxes.length} 个框 / {proposal.annotations.vectors.length} 条三维矢量 / {proposal.annotations.image_polygons.length} 个图像区域</p>{proposal.warnings?.map(w => <p key={w}>{w}</p>)}<button className="button primary" disabled={busy || jobBusy} onClick={() => void applyProposal()}>应用候选标注</button><button className="button ghost" disabled={busy || jobBusy} onClick={() => setProposal(null)}>丢弃候选</button></div>}
        <p className="micro-copy">鉴权 token 在服务端安全变量中配置，不写入浏览器。部署方式见 docs/model-worker.md。</p>
      </> : <div className="help-content"><ol><li><strong>整场景重建：</strong>导入 clip 自动用输入车辆位姿融合整段点云；需要配准修正时选择 ICP 重建。切换时刻不会重置地图。</li><li><strong>4D 轨迹：</strong>世界坐标目标在关键帧间按时间插值。编辑任一时刻自动建立关键帧；静态目标或道路矢量在整个场景共享。</li><li><strong>拖动框：</strong>选中目标即显示框面尺寸控制柄：红色改长、绿色改宽、蓝色改高，中心 ↔ 移动位置；无需输入尺寸。切换旋转可调整航向角，俯视图调整高度需切回透视 3D。右侧数值与左侧图像投影同步更新。</li><li><strong>车道线：</strong>进入车道线标注，选择类别，点击「开始绘制车道线」，在俯视路面上逐点点击，Enter 完成、Backspace 撤回、Esc 取消，最后保存。完成后选中车道线，直接拖动点云中的编号顶点修改位置，图像投影实时跟随；拖动保持该顶点高度，也可在右侧调整 XYZ。世界坐标车道线在整个 clip 共享。</li><li><strong>地面标线：</strong>选择反射强度着色并开启「仅看地面」，按坐标设置地面参考高度和半带宽。重建时启用「保留标线反射细节」。车道线和箭头需要源数据测得的强度差异，几何重建无法补出未采样纹理。</li><li><strong>相机栏：</strong>七路图像位于左侧可滚动栏，默认显示框和矢量标注投影。原始点云投影是独立可选图层。</li><li><strong>车道线融合：</strong>车道线任务中点击「融合重复车道线」，预览点云和图像中的融合结果，应用后再保存。默认保留原线的重叠段并延长覆盖范围；LATR 应用候选时可自动融合重叠线。</li><li><strong>模型预标注：</strong>配置实际模型 Worker，检查就绪状态，选择模型和类别。SAM2 支持图像提示；LATR 使用前视图像和标定生成三维车道线；所有结果需人工应用并保存。</li></ol><div className="help-coordinate"><strong>坐标与数据</strong><p>场景地图和轨迹使用世界米坐标；源帧使用 LiDAR 坐标；相机为光学坐标。投影通过校准矩阵链，并采用重建后的实际配准变换。逐点语义标签始终对应源帧点索引。</p></div></div>}
    </div></div>}
  </div>;
}
