const COLORS: Record<string, string> = {
  car: '#37ddbf', truck: '#fbad5c', bus: '#fbad5c', vehicle: '#37ddbf',
  pedestrian: '#f476aa', cyclist: '#aa96ff', bicycle: '#aa96ff',
  road: '#65acff', ground: '#65acff', sidewalk: '#cd9fe8',
  vegetation: '#a3d465', building: '#c39dff', pole: '#f1d36a',
  lane: '#f7d16a', lane_line: '#f7d16a', lane_marking: '#f7d16a', lane_solid: '#f7d16a', lane_dashed: '#f7d16a',
  road_boundary: '#37ddbf', stop_line: '#f476aa',
  curb: '#ef8d85', crosswalk: '#aa96ff', direction_arrow: '#aa96ff', drivable_area: '#65acff',
  unknown: '#ff3b5c', unlabeled: '#98a8b8', free: '#446f8d',
};

export function labelColor(label: string): string {
  if (COLORS[label]) return COLORS[label];
  let hash = 0;
  for (let i = 0; i < label.length; i += 1) hash = (hash * 31 + label.charCodeAt(i)) | 0;
  return `hsl(${Math.abs(hash) % 360}, 68%, 66%)`;
}
