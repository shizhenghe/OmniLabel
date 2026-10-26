import type {Annotations, Frame, SceneMap} from './types';
import {transformBox, transformPoint, worldFromLidar} from './scene';

/** Include frame-only edits as well as saved scene tracks without drawing migrated labels twice. */
export function reviewWorldAnnotations(world: Annotations, source: Annotations, frame: Frame,
  registration: Pick<SceneMap, 'world_from_lidar'>): Annotations {
  const transform = worldFromLidar(frame, registration);
  const tracks = new Set(world.boxes.flatMap(box => [box.id, box.track_id]));
  const boxes = source.boxes.filter(box => !tracks.has(box.id) && !tracks.has(box.track_id)).map(box => transformBox(box, transform));
  const vectorSignature = (vector: Annotations['vectors'][number]) => JSON.stringify([vector.label, vector.closed,
    vector.points.map(point => point.map(value => Number(value.toFixed(3))))]);
  const vectorIds = new Set(world.vectors.map(vector => vector.id));
  const signatures = new Set(world.vectors.map(vectorSignature));
  const vectors = source.vectors.map(vector => ({...vector, points: vector.points.map(point => transformPoint(point, transform))}))
    .filter(vector => !vectorIds.has(vector.id) && !signatures.has(vectorSignature(vector)));
  return {...world, boxes: [...world.boxes, ...boxes], vectors: [...world.vectors, ...vectors],
    image_polygons: source.image_polygons, point_labels: source.point_labels};
}
