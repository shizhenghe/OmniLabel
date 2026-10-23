export type Vec3 = [number, number, number];

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
  images: Record<string, string>;
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
}
