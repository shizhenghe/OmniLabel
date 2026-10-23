# 4D scene / model / manipulation extension contract

Existing per-frame annotation APIs remain compatible. Scene editing uses a
world coordinate frame, one persistent vector layer and temporal track keyframes.
Camera views always use active-frame LiDAR coordinates after world-to-LiDAR
conversion. Never mix scene points and LiDAR annotations in the renderer.

```ts
interface SceneKeyframe { frame_id:string; center:Vec3; size:Vec3; yaw:number }
interface SceneTrack { id:string; label:string; mode:'static'|'dynamic'; keyframes:SceneKeyframe[] }
interface SceneAnnotations { revision:number; coordinate_frame:'world'; vectors:Vector[]; tracks:SceneTrack[] }
interface SceneMap { points:number[][]; point_labels:Record<string,string>;
  coordinate_frame:'world'; frame_count:number; point_count:number;
  trajectory:{frame_id:string;timestamp:number;position:Vec3}[];
  bounds:{min:Vec3;max:Vec3}; method:string; voxel_size:number;
  preserve_detail:boolean; detail_voxel_size:number; effective_voxel_size:number;
  point_reduction:'brightest-measured-point'|'centroid-mean'; fusion_version:number;
  intensity_available:boolean; intensity_has_contrast:boolean; intensity_range:[number,number]|null;
  registration?:Record<string,unknown>; [key:string]:unknown }
```

- `GET /api/clips/{id}/scene/annotations`: persistent scene state, initializes
  from saved per-frame boxes/vectors in world coordinates without deleting originals.
- `PUT` same endpoint: revision-checked scene save; rejects unknown frame IDs.
- `GET /api/clips/{id}/scene/frame/{frame_id}`: scene-derived `Annotations` in
  active LiDAR coordinates (keyframe interpolation, persistent world vectors).
- `POST /api/clips/{id}/scene/reconstruct` JSON
  `{voxel_size:number,registration:'poses'|'icp',exclude_dynamic:boolean,
    preserve_detail?:boolean,detail_voxel_size?:number}`:
  returns persisted job `{id,kind:'reconstruct',status,...}`, polls existing jobs API.
  Pose mode fuses input poses; ICP mode registers scans, reports fit/degeneracy
  and never silently claims SLAM. Successful result is SceneMap and is cached.
- Detail preservation defaults to true with 0.05m detail voxels (range 0.05–0.2m).
  Effective voxel resolution is `min(voxel_size,detail_voxel_size)`. The brightest
  observed sample per cell retains its paired XYZ/intensity/semantic label. Disabling
  it restores centroids and mean intensity; legacy `fuse_map` callers still default
  to that behavior. No reflectance or lane geometry absent from input is inferred.
- `intensity_available` indicates a nonzero signal after filtering, not provenance of
  the original file: normalized XYZ rows are zero-filled, and native all-zero intensity
  is likewise treated as unusable. `intensity_has_contrast` reports actual variation.
- `GET /api/clips/{id}/scene/map`: latest reconstructed SceneMap, 404 if absent.
  Old fusion-version caches also return 404 to request refresh; saved registered
  transforms and scene annotations are retained independently.
- Exports should include `scene_annotations` and optional `scene_map` metadata;
  import validates and restores scene state. Existing source data is preserved.

Scene tracks hold endpoint keyframes outside their keyed interval; dynamic
keyframes interpolate centers/sizes and shortest-path yaw by actual timestamps.
Static tracks use their first keyframe. Scene frame projections use current edits
immediately on the client, not only saved server labels.

PointCloudView extends props `onUpdateBox?(id:string,patch:Partial<Box>):void`,
`transformMode?:'translate'|'scale'|'rotate'`, `editable?:boolean`,
`sceneMode?:boolean`, `trajectory?:...`, `activeFrameId?:string`.
Drag handles update controlled box data, suppress orbit/click selection while
dragging and notify completion if needed. No tests should invoke edit callbacks
directly in place of actual pointer drags.

Model workers are actual optional model runtimes. Per-model endpoint settings
and capability checks use `/api/model-connections` and
`POST /api/model-connections/{model_id}/check`. Model deployment, prompt payloads,
separate dependency environments and review rules are documented in
[model-worker.md](model-worker.md).
