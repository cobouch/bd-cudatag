#!/usr/bin/env python3
"""Score bdcudatag on the NAPA AprilTag Synthetic Benchmark (tag36h11).

Reads the corpus in `datasets/napa-benchmark` directly from each group's
`metadata.jsonl`, so no download step is needed once the images are present.

  recall      ground-truth tags matched by id and centre (<= 0.75 mean side)
  corner      median over matched tags of the mean corner distance in pixels;
              corners are compared under the best of the 8 dihedral orders,
              which is what the corpus itself uses
  wrong IDs   detections whose id is not in the frame's annotations
  duplicates  extra detections of an id that already matched
  time        wall clock around one detect() call, milliseconds

Examples:
  python eval_napa.py --group main --limit 10
  python eval_napa.py --group stress --res 640x480 --scenarios stress-tiny
  python eval_napa.py --group main --group stress --markdown napa.md
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - reported below
    cv2 = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_detectors import BdcudatagBackend  # noqa: E402

GROUPS = ("main", "stress", "range")


def available_groups(root: Path) -> list[str]:
    """Group directories present under `root/data`, else the corpus defaults."""
    data = root / "data"
    if data.is_dir():
        found = sorted(p.name for p in data.iterdir() if (p / "metadata.jsonl").is_file())
        if found:
            return found
    return list(GROUPS)
ROTATIONS = ((0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2))
DIHEDRAL = ROTATIONS + tuple(tuple(reversed(order)) for order in ROTATIONS)
CORNER_ORDERS = list(dict.fromkeys(DIHEDRAL))


def corner_error(truth: np.ndarray, detected: np.ndarray) -> float:
    return min(
        float(np.linalg.norm(detected[list(order)] - truth, axis=1).mean())
        for order in CORNER_ORDERS
    )


def score(record: dict, detections: list) -> tuple[int, int, int, int, list[float]]:
    truth_ids = record["tag_ids"]
    known = set(truth_ids)
    wrong = sum(1 for detection in detections if detection.tag_id not in known)

    found = 0
    duplicates = 0
    errors: list[float] = []
    for tag_id, corners in zip(truth_ids, record["corners"]):
        truth = np.asarray(corners, dtype=np.float64)
        side = 0.5 * (
            np.linalg.norm(truth[1] - truth[0]) + np.linalg.norm(truth[2] - truth[1])
        )
        centre = truth.mean(axis=0)
        hits = [
            detection
            for detection in detections
            if detection.tag_id == tag_id
            and np.linalg.norm(detection.corners.mean(axis=0) - centre) <= 0.75 * side
        ]
        if not hits:
            continue
        found += 1
        duplicates += len(hits) - 1
        errors.append(min(corner_error(truth, hit.corners) for hit in hits))
    return found, len(truth_ids), wrong, duplicates, errors


def load_records(root: Path, group: str) -> list[dict]:
    path = root / "data" / group / "metadata.jsonl"
    if not path.is_file():
        raise SystemExit(f"missing metadata: {path}")
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def select(records: list[dict], scenarios: list[str], resolutions: list[str],
           limit: int) -> list[dict]:
    if scenarios:
        records = [r for r in records if r["scenario"] in scenarios]
    if resolutions:
        records = [r for r in records if f"{r['width']}x{r['height']}" in resolutions]

    buckets: dict[tuple[str, int, int], list[dict]] = {}
    for record in records:
        buckets.setdefault((record["scenario"], record["width"], record["height"]), []).append(record)
    chosen = []
    for bucket in buckets.values():
        # Sort by tag size so a small --limit still spans the size sweep.
        bucket.sort(key=lambda r: (min(r["mean_side_px"], default=0.0), r["frame_index"]))
        if limit and len(bucket) > limit:
            step = len(bucket) / float(limit)
            chosen.extend(bucket[int(i * step)] for i in range(limit))
        else:
            chosen.extend(bucket)
    chosen.sort(key=lambda r: (r["width"], r["height"], r["scenario"], r["frame_index"]))
    return chosen


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(np.ceil(fraction * len(ordered))) - 1))
    return ordered[index]


def evaluate(group: str, records: list[dict], root: Path, lib: str,
             warmup: int) -> list[dict]:
    rows = []
    by_resolution: dict[tuple[int, int], list[dict]] = {}
    for record in records:
        by_resolution.setdefault((record["width"], record["height"]), []).append(record)

    for (width, height), subset in sorted(by_resolution.items()):
        backend = BdcudatagBackend(lib, max_detections=64)
        backend.prepare(width, height)

        images = []
        for record in subset:
            path = root / "data" / group / record["file_name"]
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise SystemExit(f"could not read {path}")
            if image.shape != (height, width):
                raise SystemExit(f"size mismatch in {path}")
            images.append(np.ascontiguousarray(image))

        for i in range(min(warmup, len(images))):
            backend.detect(images[i % len(images)])

        buckets: dict[str, dict] = {}
        for record, image in zip(subset, images):
            start = time.perf_counter()
            detections = backend.detect(image)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            found, tags, wrong, duplicates, errors = score(record, detections)
            frame_stats = backend.stats()
            bucket = buckets.setdefault(
                record["scenario"],
                {"frames": 0, "found": 0, "tags": 0, "wrong": 0, "dups": 0,
                 "errors": [], "times": [], "runs": [], "candidates": [],
                 "quads": [], "roi_rows": []},
            )
            bucket["frames"] += 1
            bucket["found"] += found
            bucket["tags"] += tags
            bucket["wrong"] += wrong
            bucket["dups"] += duplicates
            bucket["errors"].extend(errors)
            bucket["times"].append(elapsed_ms)
            bucket["runs"].append(frame_stats["runs"])
            bucket["candidates"].append(frame_stats["candidates"])
            bucket["quads"].append(frame_stats["quads"])
            bucket["roi_rows"].append(frame_stats["roi_rows"])

        backend.close()
        for scenario, bucket in buckets.items():
            rows.append(
                {
                    "group": group,
                    "resolution": f"{width}x{height}",
                    "width": width,
                    "height": height,
                    "scenario": scenario,
                    "frames": bucket["frames"],
                    "found": bucket["found"],
                    "tags": bucket["tags"],
                    "wrong_ids": bucket["wrong"],
                    "duplicates": bucket["dups"],
                    "corner_median_px": float(np.median(bucket["errors"]))
                    if bucket["errors"] else None,
                    "mean_ms": float(np.mean(bucket["times"])),
                    "median_ms": float(np.median(bucket["times"])),
                    "p95_ms": percentile(bucket["times"], 0.95),
                    "mean_runs": float(np.mean(bucket["runs"])),
                    "mean_candidates": float(np.mean(bucket["candidates"])),
                    "mean_quads": float(np.mean(bucket["quads"])),
                    "mean_roi_rows": float(np.mean(bucket["roi_rows"])),
                }
            )
    return rows


def summary(rows: list[dict]) -> dict:
    found = sum(r["found"] for r in rows)
    tags = sum(r["tags"] for r in rows)
    errors = []
    times = []
    for row in rows:
        if row["corner_median_px"] is not None:
            errors.append(row["corner_median_px"])
        times.extend([row["mean_ms"]] * max(1, row["tags"]))
    return {
        "found": found,
        "tags": tags,
        "recall": found / tags if tags else 0.0,
        "wrong_ids": sum(r["wrong_ids"] for r in rows),
        "duplicates": sum(r["duplicates"] for r in rows),
        "corner_median_px": float(np.median(errors)) if errors else None,
        "mean_ms": float(np.mean(times)) if times else 0.0,
    }


def print_rows(rows: list[dict]) -> None:
    header = (f"{'scenario':<22} {'resolution':<10} {'frames':>6} {'tags':>5} "
              f"{'recall':>7} {'corner':>6} {'wrong':>5} {'dup':>4} {'mean ms':>8}")
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda r: (r["recall"] if "recall" in r else
                                           r["found"] / max(1, r["tags"]),
                                           r["scenario"], r["resolution"])):
        recall = 100.0 * row["found"] / row["tags"] if row["tags"] else 0.0
        corner = (f"{row['corner_median_px']:.2f}"
                  if row["corner_median_px"] is not None else "n/a")
        print(f"{row['scenario']:<22} {row['resolution']:<10} {row['frames']:>6} "
              f"{row['tags']:>5} {recall:>6.1f}% {corner:>6} {row['wrong_ids']:>5} "
              f"{row['duplicates']:>4} {row['mean_ms']:>8.3f}")


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(here / "datasets" / "napa-benchmark"),
                        help="dataset directory (default: datasets/napa-benchmark)")
    parser.add_argument("--group", action="append",
                        help="group to evaluate; repeatable (default: every group "
                             "directory in --root)")
    parser.add_argument("--scenarios", nargs="*", default=[], help="scenario filter")
    parser.add_argument("--res", nargs="*", default=[],
                        help="resolution filter, e.g. 640x480")
    parser.add_argument("--limit", type=int, default=0,
                        help="frames per scenario and resolution (0: all)")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--lib", default=None, help="path to bdcudatag.dll/.so")
    parser.add_argument("--stats", action="store_true",
                        help="print per-stage counters and a time-sorted table")
    parser.add_argument("--markdown", default=None, help="write the table here")
    parser.add_argument("--json", default=None, help="write raw rows here")
    args = parser.parse_args()

    if cv2 is None:
        print("error: OpenCV is required (`pip install opencv-python`)", file=sys.stderr)
        return 2

    lib = args.lib
    if lib is None:
        candidates = [
            here.parent / "build" / "Release" / "bdcudatag.dll",
            here.parent / "build" / "bdcudatag.dll",
            here.parent / "build" / "libbdcudatag.so",
            here.parent / "build" / "Release" / "libbdcudatag.so",
        ]
        existing = [path for path in candidates if path.exists()]
        if not existing:
            print("error: no built bdcudatag library found; pass --lib", file=sys.stderr)
            return 2
        lib = str(existing[0])

    root = Path(args.root)
    groups = args.group or available_groups(root)
    all_rows = []
    for group in groups:
        records = load_records(root, group)
        records = select(records, args.scenarios, args.res, args.limit)
        if not records:
            print(f"no records for group {group}", file=sys.stderr)
            continue
        rows = evaluate(group, records, root, lib, args.warmup)
        all_rows.extend(rows)
        totals = summary(rows)
        print(f"\n=== {group}: {totals['found']}/{totals['tags']} tags "
              f"({100.0 * totals['recall']:.1f}%), "
              f"corner {totals['corner_median_px']:.2f} px, "
              f"wrong ids {totals['wrong_ids']}, mean {totals['mean_ms']:.3f} ms ===")
        print_rows(rows)
        if args.stats:
            header = (f"{'scenario':<22} {'resolution':<10} {'mean ms':>8} "
                      f"{'runs':>8} {'cand':>6} {'quads':>6} {'roi':>5}")
            print(header)
            print("-" * len(header))
            for row in sorted(rows, key=lambda r: r["mean_ms"], reverse=True):
                print(f"{row['scenario']:<22} {row['resolution']:<10} "
                      f"{row['mean_ms']:>8.3f} {row['mean_runs']:>8.0f} "
                      f"{row['mean_candidates']:>6.1f} {row['mean_quads']:>6.1f} "
                      f"{row['mean_roi_rows']:>5.0f}")

    if all_rows:
        totals = summary(all_rows)
        print(f"\n=== all groups: {totals['found']}/{totals['tags']} tags "
              f"({100.0 * totals['recall']:.1f}%), "
              f"corner {totals['corner_median_px']:.2f} px, "
              f"wrong ids {totals['wrong_ids']}, mean {totals['mean_ms']:.3f} ms ===")

    if args.json:
        Path(args.json).write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
        print(f"rows written to {args.json}", file=sys.stderr)
    if args.markdown:
        lines = [
            "| Group | Scenario | Resolution | Tags | Recall | Corner (px) | "
            "Wrong IDs | Mean (ms) |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in all_rows:
            recall = 100.0 * row["found"] / row["tags"] if row["tags"] else 0.0
            corner = (f"{row['corner_median_px']:.2f}"
                      if row["corner_median_px"] is not None else "n/a")
            lines.append(
                f"| {row['group']} | {row['scenario']} | {row['resolution']} | "
                f"{row['tags']} | {recall:.1f}% | {corner} | {row['wrong_ids']} | "
                f"{row['mean_ms']:.3f} |"
            )
        Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"table written to {args.markdown}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
