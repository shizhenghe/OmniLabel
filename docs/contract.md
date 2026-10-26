# OmniLabel development contract

Coordinates are right-handed ego/LiDAR coordinates in metres: x forward, y left,
z up. Column-vector transforms are row-major nested arrays. Camera coordinates
are x right, y down, z forward. Camera pixels are in original image dimensions.
Yaw is radians about +z. `size` is [length, width, height].

```ts
type Vec3 = [number, number, number];
type Camera = {id: string; name: string; width: number; height: number;
  intrinsics: number[][]; camera_from_ego: number[][];
  distortion?: number[]}; // Brown-Conrady k1,k2,p1,p2,k3
type Frame = {id: string; timestamp: number; ego_to_world: number[][];
  lidar_to_ego: number[][]; points: number[][]; images: Record<string,string>;
  annotation_camera_from_lidar?: Record<string,number[][]>};
type Clip = {id: string; name: string; description: string; fps: number;
  cameras: Camera[]; frames: Frame[]; source?: string};
type Box = {id: string; track_id: string; label: string; center: Vec3;
  size: Vec3; yaw: number};
type Vector = {id: string; label: string; points: Vec3[]; closed: boolean};
type ImagePolygon = {id: string; camera_id: string; label: string; points: number[][]};
type Annotations = {boxes: Box[]; vectors: Vector[];
  image_polygons: ImagePolygon[]; point_labels: Record<string,string>;
  revision: number};
```

Frame points are LiDAR coordinates, `[x,y,z,intensity]`; boxes, vectors and point
indices refer to this same per-frame LiDAR space. Each camera extrinsic maps ego
to optical camera coordinates; projection is K * camera_from_ego * lidar_to_ego.
When supplied, `annotation_camera_from_lidar[camera_id]` replaces those two
extrinsics for annotation overlays only; raw point-cloud projection is unchanged.

API base `/api`:

- `GET /health` → `{status:"ok"}`.
- `GET /clips` → summary array `{id,name,description,fps,frame_count,camera_count}`.
- `GET /clips/{id}` → Clip metadata; each frame has `points:[]` and full `point_count`.
- `GET /clips/{id}/frames/{frame_id}/points` → all source rows by default:
  `{points,point_count,offset,returned_point_count}`. Optional `offset`/`count`
  request contiguous ranges without reindexing point labels. No implicit point quota or preview sampling.
- `POST /clips/import` multipart `file` (.json or .zip). Returns Clip. ZIP has
  `manifest.json`, relative image paths, and optional `.bin` float32 XYZI,
  `.npy` Nx3/4 or ASCII `.pcd` point paths in `frame.points_path`.
- `GET /clips/{id}/frames/{frame_id}/annotations` → Annotations.
- `PUT` same URL, JSON Annotations → saved Annotations. Revision conflict is 409.
- `POST /clips/{id}/project` with `{frame_id,camera_id,stride?:number}` →
  `{points:[{index,u,v,depth,label?}],width,height}` (visible projected points).
- `POST /clips/{id}/jobs` with `{kind:"map"|"occupancy"|"propagate"|"ground-segmentation"|"prelabel",
  frame_id?:string,voxel_size?:number,target_frame_id?:string,model_id?:string,prompt?:string}` → job with
  `{id,kind,status:"queued"|"running"|"completed"|"failed",result?,error?}`.
- `GET /jobs/{id}` → job. Results: map `{points,frame_count,coordinate_frame:"world"}`;
  occupancy `{voxels:[{index:[i,j,k],center:Vec3,state:"occupied"|"free",label:string}],
  voxel_size,coordinate_frame:"lidar"}`; propagation/ground segmentation
  `{updated_frames:number}`. Jobs use saved annotations.
- `GET /clips/{id}/export` → streamed JSON `{schema_version:"1.1",clip,annotations,...}`
  with full inline point data even when the store uses external NPY files.
- `GET /clips/{id}/export.zip` → portable ZIP with manifest, annotations,
  complete NPY point assets and local images. Manifests use `points_path`, not server-local filenames.
- `GET /clips/{id}/frames/{frame_id}/masks/{camera_id}.zip` → `mask.png`
  (16-bit class IDs) and `labels.json` from a single saved annotation snapshot.
  Separate `.png` and `.json` routes are also available.
- `GET /models` → `{models:[{id,name,tasks:string[],available:boolean,description,url}]}`.
- Assets returned in clips use `/api/assets/...` paths; import confines files to
  the clip data directory. External inference is optional; no model weights are bundled.

`prelabel` calls an optional administrator-configured HTTP model worker, not a
bundled model. Its result includes `{proposal:Annotations,model_id,model_version,
review_required:true,source:"external-worker",frame_id,clip_id}`. Proposals are
never automatically persisted. The UI rejects stale proposals and requires
explicit application, manual checking and save. See [model-worker.md](model-worker.md).

Frontend: React + TypeScript + Vite + Three.js; root package.json owns scripts.
Backend: FastAPI + NumPy, backend/omnilabel; persisted data in ignored data/.
`OMNILABEL_DATA_DIR` overrides data location. Backend serves built dist/ when
present. Vite proxies /api to 127.0.0.1:8000 in development.
