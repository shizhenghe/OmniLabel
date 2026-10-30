export type Vec3 = [number, number, number];
/** Preserve Float64 map coordinates until the renderer subtracts its local origin. */
export type PointBuffer = Float32Array | Float64Array;

export interface Camera {
  id: string;
  name: string;
  width: number;
  height: number;
  intrinsics: number[][];
  camera_from_ego: number[][];
  distortion?: number[];
}

export interface Frame {
  id: string;
  timestamp: number;
  ego_to_world: number[][];
  lidar_to_ego: number[][];
  points: number[][];
  point_count: number;
  calibration_pointcloud_file?: string | null;
  calibration_point_count?: number | null;
  calibration_lidar_to_ego?: number[][] | null;
  calibration_timestamp?: number | null;
  images: Record<string, string>;
  annotation_camera_from_lidar?: Record<string, number[][]>;
  /** Per-frame physical correction, shared by raw points and 3D labels. */
  projection_camera_from_lidar?: Record<string, number[][]>;
}

export interface Clip {
  id: string;
  name: string;
  description: string;
  fps: number;
  cameras: Camera[];
  frames: Frame[];
  source?: string;
}

export interface ClipSummary {
  id: string;
  name: string;
  description: string;
  fps: number;
  frame_count: number;
  camera_count: number;
}

export interface Box {
  id: string;
  track_id: string;
  label: string;
  center: Vec3;
  size: Vec3;
  yaw: number;
}

export interface Vector {
  id: string;
  label: string;
  points: Vec3[];
  closed: boolean;
}

export interface ImagePolygon {
  id: string;
  camera_id: string;
  label: string;
  points: number[][];
}

export interface Annotations {
  boxes: Box[];
  vectors: Vector[];
  image_polygons: ImagePolygon[];
  point_labels: Record<string, string>;
  revision: number;
}

export interface ProjectionPoint {
  index: number;
  u: number;
  v: number;
  depth: number;
  label?: string;
}

export interface Voxel {
  index: [number, number, number];
  center: Vec3;
  state: 'occupied' | 'free';
  label: string;
}

export interface JobResult {
  points?: number[][];
  frame_count?: number;
  coordinate_frame?: string;
  voxels?: Voxel[];
  voxel_size?: number;
  updated_frames?: number;
  [key: string]: unknown;
}

export interface Job {
  id: string;
  kind: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  result?: JobResult;
  error?: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  tasks: string[];
  available: boolean;
  description: string;
  url: string;
  configured?: boolean;
  status?: string;
  status_detail?: string;
  reason?: string;
  supports?: {point_prompt?: boolean; box_prompt?: boolean; text_prompt?: boolean; occupancy?: boolean; lane_vectors?: boolean; weight_variants?: string[]; active_weight_variant?: string};
}

export interface SceneKeyframe { frame_id: string; center: Vec3; size: Vec3; yaw: number; }
export interface SceneTrack { id: string; label: string; mode: 'static' | 'dynamic'; keyframes: SceneKeyframe[]; }
export interface SceneAnnotations { revision: number; coordinate_frame: 'world'; vectors: Vector[]; tracks: SceneTrack[]; }
export interface SceneMap {
  points: number[][];
  /** Full XYZI cloud loaded from NPY; avoids expanding millions of rows into JSON objects. */
  point_buffer?: PointBuffer;
  point_labels: Record<string, string>;
  coordinate_frame: 'world';
  frame_count: number;
  point_count: number;
  rendered_point_count?: number;
  trajectory: {frame_id: string; timestamp: number; position: Vec3}[];
  bounds: {min: Vec3; max: Vec3};
  method: string;
  source_map?: string;
  voxel_size: number;
  scene_revision?: number;
  preserve_detail?: boolean;
  fusion_mode?: 'full' | 'detail' | 'surface' | 'centroid';
  detail_voxel_size?: number;
  effective_voxel_size?: number;
  intensity_available?: boolean;
  intensity_has_contrast?: boolean;
  intensity_range?: [number, number] | null;
  fusion_version?: number;
  point_reduction?: string;
  world_from_lidar: Record<string, number[][]>;
  registration?: Record<string, unknown>;
}
export interface ModelConnection {model_id: string; endpoint: string; configured?: boolean; source?: string;}
