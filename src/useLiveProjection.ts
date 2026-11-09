import {useEffect, useMemo, useRef, useState} from 'react';
import type {Camera, Frame, PointBuffer, ProjectionPoint, ProjectionRaster} from './types';
import {cameraFromLidar} from './geometry/cameraProjection';
import {multiply} from './scene';

type CameraRequest = {camera: Camera; matrix: number[][]; key: string; raster?: boolean; labels?: Record<string, string>};
type Job = {type: 'project'; id: number; version: number; cameras: CameraRequest[]};
type Source = {points?: number[][]; map?: PointBuffer; frameId?: string};
type CachedProjection = {key: string; points: ProjectionPoint[]; source: Source; raster?: ProjectionRaster};

/** Keep coordinates in a worker and retain only the latest queued calibration. */
export function useLiveProjection(frame: Frame | null, cameras: Camera[], map: PointBuffer | undefined,
  frameFromWorld: number[][], enabled: boolean, focus?: string, labels?: Record<string, string>) {
  const worker = useRef<Worker | null>(null);
  const busy = useRef(false);
  const pending = useRef<Job | null>(null);
  const serial = useRef(0);
  const version = useRef(0);
  const source = useRef<Source>({});
  const expected = useRef(new Map<string, string>());
  const [cache, setCache] = useState<Record<string, CachedProjection>>({});
  const cacheRef = useRef(cache);
  cacheRef.current = cache;
  const [updating, setUpdating] = useState(false);
  const [error, setError] = useState('');
  const pump = () => {
    if (busy.current || !worker.current || !pending.current) return;
    const job = pending.current;
    pending.current = null;
    busy.current = true;
    worker.current.postMessage(job);
  };
  useEffect(() => {
    let instance: Worker;
    try {
      instance = new Worker(new URL('./geometry/projection.worker.ts', import.meta.url), {type: 'module'});
    } catch {
      setError('浏览器无法启动点云投影后台线程，请检查浏览器设置后重新加载。');
      return;
    }
    worker.current = instance;
    instance.onmessage = event => {
      busy.current = false;
      const data = event.data;
      // Source versions prevent results from a previous frame/map from leaking.
      if (data.version === version.current) {
        const updates: Record<string, CachedProjection> = {};
        for (const item of data.output as {id: string; key: string; buffer: Float64Array; raster?: ProjectionRaster}[]) {
          if (expected.current.get(item.id) !== item.key) {item.raster?.bitmap.close(); continue;}
          const points: ProjectionPoint[] = [];
          for (let i = 0; i < item.buffer.length; i += 4) {
            points.push({index: item.buffer[i], u: item.buffer[i+1], v: item.buffer[i+2], depth: item.buffer[i+3]});
          }
          updates[item.id] = {key: item.key, points, source: source.current, raster: item.raster ? {...item.raster, first: points[0]} : undefined};
        }
        if (Object.keys(updates).length) setCache(old => ({...old, ...updates}));
      } else for (const item of data.output) item.raster?.bitmap.close();
      if (pending.current) pump();
      else setUpdating(false);
    };
    instance.onerror = event => {
      event.preventDefault();
      busy.current = false;
      pending.current = null;
      instance.terminate();
      worker.current = null;
      setUpdating(false);
      setError('点云投影后台计算失败，请重新加载页面。');
    };
    return () => {
      instance.onmessage = null;
      instance.onerror = null;
      instance.terminate();
      worker.current = null;
      busy.current = false;
      pending.current = null;
      version.current = 0;
    };
  }, []);

  const mapSource = frame?.points.length ? undefined : map;
  useEffect(() => {
    if (!enabled || !frame || !worker.current) return;
    if (version.current && source.current.points === frame.points
        && source.current.map === mapSource && source.current.frameId === frame.id) return;
    source.current = {points: frame.points, map: mapSource, frameId: frame.id};
    setCache({});
    const currentVersion = ++version.current;
    const count = mapSource ? mapSource.length/4 : frame.points.length;
    if (mapSource) {
      const coordinates = mapSource.slice();
      worker.current.postMessage({type:'source',version:currentVersion,coordinates,stride:1,components:4},[coordinates.buffer]);
    } else {
      const coordinates = new Float64Array(count*3);
      for (let index=0;index<count;index++) for (let axis=0;axis<3;axis++) coordinates[index*3+axis]=frame.points[index][axis];
      worker.current.postMessage({type:'source',version:currentVersion,coordinates,stride:1,components:3},[coordinates.buffer]);
    }
    pending.current = null;
  }, [frame?.points, frame?.id, mapSource, enabled]);

  const labelsKey = useMemo(() => mapSource ? JSON.stringify(labels ?? {}) : '', [mapSource, labels]);
  const previousCache = useRef(cache);
  useEffect(() => {
    for (const [id, old] of Object.entries(previousCache.current)) if (old.raster && old.raster.bitmap !== cache[id]?.raster?.bitmap) old.raster.bitmap.close();
    previousCache.current = cache;
  }, [cache]);
  useEffect(() => () => {for (const entry of Object.values(cacheRef.current)) entry.raster?.bitmap.close();}, []);
  const selected = focus ? cameras.filter(camera => camera.id === focus) : cameras;
  const requests = frame ? selected.map(camera => {
    const matrix = mapSource ? multiply(cameraFromLidar(camera, frame), frameFromWorld) : cameraFromLidar(camera, frame);
    return {camera, matrix, key: JSON.stringify([frame.id, camera, matrix, labelsKey]), raster: Boolean(mapSource), labels: mapSource ? labels : undefined};
  }) : [];
  const requestKey = JSON.stringify(requests.map(item => item.key));
  useEffect(() => {
    expected.current = new Map(requests.map(item => [item.camera.id, item.key]));
    if (!enabled || !frame || !worker.current) {
      pending.current = null;
      setUpdating(false);
      return;
    }
    const changed = requests.filter(item => {
      const stored = cacheRef.current[item.camera.id];
      return stored?.key !== item.key || stored.source !== source.current;
    });
    if (!changed.length) {
      pending.current = null;
      setUpdating(false);
      return;
    }
    setError('');
    setUpdating(true);
    pending.current = {type: 'project', id: ++serial.current, version: version.current, cameras: changed};
    pump();
  }, [requestKey, enabled, frame?.points, mapSource]);

  const values: Record<string, ProjectionPoint[]> = {};
  const rasters: Record<string, ProjectionRaster> = {};
  if (enabled) for (const item of requests) {
    const stored = cache[item.camera.id];
    if (stored?.key === item.key && stored.source.points === frame?.points
        && stored.source.map === mapSource && stored.source.frameId === frame?.id) {values[item.camera.id] = stored.points; if(stored.raster)rasters[item.camera.id]=stored.raster;}
  }
  return {projections: values, rasters, updating: enabled && updating, error: enabled ? error : ''};
}
