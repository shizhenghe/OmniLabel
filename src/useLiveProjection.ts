import {useEffect, useRef, useState} from 'react';
import type {Camera, Frame, PointBuffer, ProjectionPoint} from './types';
import {cameraFromLidar} from './geometry/cameraProjection';
import {multiply} from './scene';

type CameraRequest = {camera: Camera; matrix: number[][]; key: string};
type Job = {type: 'project'; id: number; version: number; cameras: CameraRequest[]};
type Source = {points?: number[][]; map?: PointBuffer; frameId?: string};
type CachedProjection = {key: string; points: ProjectionPoint[]; source: Source};

/** Keep coordinates in a worker and retain only the latest queued calibration. */
export function useLiveProjection(frame: Frame | null, cameras: Camera[], map: PointBuffer | undefined,
  frameFromWorld: number[][], enabled: boolean, focus?: string) {
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
        for (const item of data.output as {id: string; key: string; buffer: Float64Array}[]) {
          if (expected.current.get(item.id) !== item.key) continue;
          const points: ProjectionPoint[] = [];
          for (let i = 0; i < item.buffer.length; i += 4) {
            points.push({index: item.buffer[i], u: item.buffer[i+1], v: item.buffer[i+2], depth: item.buffer[i+3]});
          }
          updates[item.id] = {key: item.key, points, source: source.current};
        }
        if (Object.keys(updates).length) setCache(old => ({...old, ...updates}));
      }
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
    // Preserve the existing map image-overlay sampling, with original indices.
    // Full source scans and the 3D map are never truncated here.
    const step = mapSource ? Math.max(1, Math.ceil(count/60000)) : 1;
    const coordinates = new Float64Array(Math.ceil(count/step)*3);
    for (let index = 0, offset = 0; index < count; index += step, offset += 3) {
      for (let axis = 0; axis < 3; axis++) {
        coordinates[offset+axis] = mapSource ? mapSource[index*4+axis] : frame.points[index][axis];
      }
    }
    worker.current.postMessage({type: 'source', version: currentVersion, coordinates, stride: step}, [coordinates.buffer]);
    pending.current = null;
  }, [frame?.points, frame?.id, mapSource, enabled]);

  const selected = focus ? cameras.filter(camera => camera.id === focus) : cameras;
  const requests = frame ? selected.map(camera => {
    const matrix = mapSource ? multiply(cameraFromLidar(camera, frame), frameFromWorld) : cameraFromLidar(camera, frame);
    return {camera, matrix, key: JSON.stringify([frame.id, camera, matrix])};
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
  if (enabled) for (const item of requests) {
    const stored = cache[item.camera.id];
    if (stored?.key === item.key && stored.source.points === frame?.points
        && stored.source.map === mapSource && stored.source.frameId === frame?.id) values[item.camera.id] = stored.points;
  }
  return {projections: values, updating: enabled && updating, error: enabled ? error : ''};
}
