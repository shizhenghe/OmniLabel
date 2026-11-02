#!/usr/bin/env python3
"""Replay an exported snapshot or optimize existing OmniLabel lane-pair data."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from omnilabel.lane_calibration import OptimizeRequest, OptimizerConfig, build_problem
from omnilabel.lane_optimizer import optimize, write_outputs
from omnilabel.storage import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, help='Exported input_snapshot.json; retains its explicit pose convention')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--clip')
    parser.add_argument('--camera')
    parser.add_argument('--pairing', type=Path, help='Saved pairs.json or an exported pairing document')
    parser.add_argument('--settings', type=Path, help='JSON/YAML OptimizeRequest with explicit coordinate/time confirmation')
    parser.add_argument('--output', type=Path, required=True, help='New directory; an existing run is never overwritten')
    args = parser.parse_args()
    try:
        if args.snapshot:
            problem = json.loads(args.snapshot.read_text())
            problem['config'] = OptimizerConfig.model_validate(problem['config']).model_dump()
            if problem['conventions']['pose'] != 'world_from_lidar' or problem['conventions']['extrinsic'] != 'camera_from_lidar':
                raise ValueError('Snapshot must explicitly use world_from_lidar and camera_from_lidar.')
        else:
            if not all([args.data_dir, args.clip, args.camera, args.settings]):
                parser.error('Provide --snapshot, or --data-dir --clip --camera --settings.')
            import yaml
            store = Store(args.data_dir)
            path = args.pairing or args.data_dir / 'lane_calibration' / args.clip / args.camera / 'pairs.json'
            pairing = json.loads(path.read_text())
            settings = yaml.safe_load(args.settings.read_text())
            if not isinstance(settings, dict): raise ValueError('Settings must contain a JSON/YAML mapping.')
            settings.setdefault('pairing_revision', pairing['revision'])
            problem, _, _ = build_problem(store, args.clip, args.camera, pairing, OptimizeRequest.model_validate(settings))
        args.output.mkdir(parents=True, exist_ok=False)
        result = optimize(problem)
        write_outputs(problem, result, args.output)
        print(json.dumps({'output':str(args.output.resolve()),'usable':result['usable'],
            'error_reduction':result['error_reduction'],'rejection_reasons':result['rejection_reasons']},ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f'Lane calibration failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
