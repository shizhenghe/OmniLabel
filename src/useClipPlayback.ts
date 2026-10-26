import {useEffect, useRef} from 'react';
import {abortableDelay, frameDurationMs, loadPlaybackFrame, type PlaybackFrame} from './playback';
import type {Clip} from './types';

interface Options {
  clip: Clip | null;
  index: number;
  playing: boolean;
  session: number;
  speed: number;
  loop: boolean;
  onFrame: (frame: PlaybackFrame) => void;
  onBuffering: (buffering: boolean) => void;
  onEnd: () => void;
  onError: (error: unknown) => void;
}

/** Hold at most the current and next complete frame; never skip unreviewed frames. */
export function useClipPlayback(options: Options) {
  const latest = useRef(options);
  latest.current = options;
  const cache = useRef(new Map<number, PlaybackFrame>());
  const lastSession = useRef(-1);
  const clipId = options.clip?.id;
  const {playing, session, speed, loop} = options;
  useEffect(() => {
    if (lastSession.current !== session) {cache.current.clear(); lastSession.current = session;}
    const currentClip = latest.current.clip;
    if (!playing || !currentClip) return;
    const controller = new AbortController();
    const signal = controller.signal;
    // Snapshot only metadata. Holding the old active scan here for the entire
    // session would retain another full cloud outside the two-frame cache.
    const clip = {...currentClip, frames: currentClip.frames.map(frame => ({...frame, points: []}))};
    let active = true;
    const load = async (index: number) => {
      const cached = cache.current.get(index);
      if (cached) return cached;
      const bundle = await loadPlaybackFrame(clip, index, signal);
      if (active && !signal.aborted) cache.current.set(index, bundle);
      return bundle;
    };
    const run = async () => {
      let index = latest.current.index;
      latest.current.onBuffering(true);
      let current: PlaybackFrame | null = await load(index);
      if (!active) return;
      latest.current.onFrame(current); latest.current.onBuffering(false);
      current = null;
      while (active) {
        const next = index + 1 < clip.frames.length ? index + 1 : loop ? 0 : null;
        const duration = frameDurationMs(clip.frames, index, clip.fps, speed);
        if (next === null) {
          await abortableDelay(duration, signal);
          if (active) latest.current.onEnd();
          return;
        }
        let ready = false;
        // Attach a rejection handler immediately so an early network failure
        // cannot become an unhandled promise while the presentation timer runs.
        const pending = load(next).then(value => {ready = true; return {value};}, error => ({error}));
        await abortableDelay(duration, signal);
        if (!active) return;
        if (!ready) latest.current.onBuffering(true);
        const result = await pending;
        if (!active || signal.aborted) return;
        if ('error' in result) throw result.error;
        index = next;
        latest.current.onFrame(result.value); latest.current.onBuffering(false);
        for (const key of cache.current.keys()) if (key !== index) cache.current.delete(key);
      }
    };
    void run().catch(error => {if (active && !signal.aborted) latest.current.onError(error);});
    return () => {active = false; controller.abort(); latest.current.onBuffering(false);};
  }, [clipId, playing, session, speed, loop]);
}
