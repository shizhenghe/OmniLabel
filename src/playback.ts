import type {Annotations, Clip, Frame} from './types';
import {request} from './api';

export interface PlaybackFrame {
  clipId: string;
  index: number;
  frameId: string;
  annotations: Annotations;
  points: number[][];
  pointCount: number;
  images: Record<string, HTMLImageElement | null>;
}

export function frameDurationMs(frames: Pick<Frame, 'timestamp'>[], index: number, fps: number, speed: number): number {
  const delta = index + 1 < frames.length ? frames[index + 1].timestamp - frames[index].timestamp
    : index > 0 ? frames[index].timestamp - frames[index - 1].timestamp : 1 / fps;
  return Math.max(1, (Number.isFinite(delta) && delta > 0 ? delta * 1000 : 1000 / fps) / speed);
}

export function abortableDelay(milliseconds: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const abort = () => {clearTimeout(timer); reject(new DOMException('Playback cancelled', 'AbortError'));};
    const timer = setTimeout(() => {signal.removeEventListener('abort', abort); resolve();}, milliseconds);
    if (signal.aborted) abort(); else signal.addEventListener('abort', abort, {once: true});
  });
}

async function prepareImage(source: string | undefined, signal: AbortSignal): Promise<HTMLImageElement | null> {
  if (!source) return null;
  return new Promise((resolve, reject) => {
    const image = new Image();
    const cleanup = () => {clearTimeout(timeout); signal.removeEventListener('abort', abort); image.onload = null; image.onerror = null;};
    const abort = () => {cleanup(); image.src = ''; reject(new DOMException('Playback cancelled', 'AbortError'));};
    const timeout = setTimeout(() => {cleanup(); image.src = ''; resolve(null);}, 10000);
    image.onload = async () => {
      try {await image.decode();} catch { /* The completed image can still be drawn by the browser. */ }
      if (signal.aborted) return;
      cleanup(); resolve(image);
    };
    image.onerror = () => {cleanup(); resolve(null);};
    if (signal.aborted) {abort(); return;}
    signal.addEventListener('abort', abort, {once: true});
    image.src = source;
  });
}

/** Load one complete synchronized frame, without sampling point-cloud data. */
export async function loadPlaybackFrame(clip: Clip, index: number, signal: AbortSignal): Promise<PlaybackFrame> {
  const frame = clip.frames[index];
  if (!frame) throw new Error('回放帧不存在');
  const path = `/clips/${encodeURIComponent(clip.id)}/frames/${encodeURIComponent(frame.id)}`;
  const [annotations, cloud, images] = await Promise.all([
    request<Annotations>(`${path}/annotations`, {signal}),
    request<{points: number[][]; point_count: number}>(`${path}/points`, {signal}),
    Promise.all(clip.cameras.map(async camera => [camera.id, await prepareImage(frame.images[camera.id], signal)] as const)),
  ]);
  return {clipId: clip.id, index, frameId: frame.id, annotations, points: cloud.points, pointCount: cloud.point_count,
    images: Object.fromEntries(images)};
}
