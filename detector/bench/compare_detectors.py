#!/usr/bin/env python3
"""Cross-detector benchmark: bdcudatag vs NAPA, OpenCV ArUco and AprilTag 3.

Every backend runs in this one process, on the same exported scenes, through
the same warm-up and timing loop, and is scored with the same definitions:

  median/p95 time   wall clock around one detect() call, milliseconds
  tags found        ground-truth tags matched to a detection with the same id
                    (and a centre within 0.75 tag sides), percent
  corner error      median over matched tags of the mean distance between the
                    four detected corners and ground truth, pixels; corners
                    are matched as an unordered set, since libraries disagree
                    on the starting corner and winding
  wrong IDs         detections whose id belongs to no tag in the scene

Scenes come from `bdcudatag_scenes`; see the Benchmarks section of the project
README.  Optional backends are imported lazily, so a missing package only
drops that row.
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import importlib.metadata
import itertools
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - reported to the user below
    cv2 = None


DETECTOR_ORDER = ["bdcudatag", "napa", "napa16", "aruco", "apriltag"]
WARMUP_FRAMES = 20
CORNER_ORDERS = list(itertools.permutations(range(4)))


@dataclass
class Detection:
    tag_id: int
    corners: np.ndarray  # 4x2 float


class Backend:
    name = "backend"
    version = ""

    def __init__(self, **_kwargs) -> None:
        pass

    def prepare(self, width: int, height: int) -> None:
        pass

    def detect(self, image: np.ndarray) -> list[Detection]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class BdcudatagBackend(Backend):
    """bdcudatag through its C ABI, in the same process as the other backends."""

    name = "bdcudatag"

    class _Config(ctypes.Structure):
        _fields_ = [
            ("struct_size", ctypes.c_uint32),
            ("max_hamming", ctypes.c_int32),
            ("min_cluster_pixels", ctypes.c_int32),
            ("min_tag_pixels", ctypes.c_int32),
            ("max_detections", ctypes.c_int32),
            ("max_candidates", ctypes.c_int32),
            ("refine_edges", ctypes.c_int32),
            ("use_cuda_graph", ctypes.c_int32),
            ("min_decision_margin", ctypes.c_float),
            ("decode_sharpening", ctypes.c_float),
        ]

    class _Stats(ctypes.Structure):
        _fields_ = [
            ("frames", ctypes.c_uint64),
            ("last_gpu_ms", ctypes.c_float),
            ("average_gpu_ms", ctypes.c_float),
            ("last_total_ms", ctypes.c_float),
            ("runs", ctypes.c_int32),
            ("candidates", ctypes.c_int32),
            ("detections", ctypes.c_int32),
            ("quads", ctypes.c_int32),
            ("roi_rows", ctypes.c_int32),
        ]

    class _Detection(ctypes.Structure):
        _fields_ = [
            ("id", ctypes.c_int32),
            ("hamming", ctypes.c_int32),
            ("decision_margin", ctypes.c_float),
            ("corners", ctypes.c_float * 8),
            ("reserved", ctypes.c_int32),
            ("homography", ctypes.c_float * 9),
        ]

    def __init__(self, lib_path: str, max_detections: int = 32) -> None:
        # Python on Windows only searches the library's own directory and the
        # system paths for dependencies, not PATH, so cudart must be added
        # explicitly before loading bdcudatag.
        self._dll_dirs = []
        if hasattr(os, "add_dll_directory"):
            lib_dir = os.path.dirname(os.path.abspath(lib_path))
            self._dll_dirs.append(os.add_dll_directory(lib_dir))
            cuda_roots = [
                os.environ.get("CUDA_PATH"),
                os.environ.get("CUDA_HOME"),
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
            ]
            for root in filter(None, cuda_roots):
                candidates = [root]
                candidates += glob.glob(os.path.join(root, "v*"))
                for candidate in candidates:
                    bin_dir = os.path.join(candidate, "bin")
                    if os.path.isdir(bin_dir):
                        self._dll_dirs.append(os.add_dll_directory(bin_dir))
        self.lib = ctypes.CDLL(str(lib_path))
        self.lib.bd_version_string.restype = ctypes.c_char_p
        self.lib.bd_config_init.argtypes = [ctypes.POINTER(self._Config)]
        self.lib.bd_detector_create.argtypes = [
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(self._Config),
            ctypes.POINTER(ctypes.c_int32),
        ]
        self.lib.bd_detector_create.restype = ctypes.c_void_p
        self.lib.bd_detector_detect_host.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int32,
            ctypes.POINTER(self._Detection),
            ctypes.c_int32,
        ]
        self.lib.bd_detector_detect_host.restype = ctypes.c_int32
        self.lib.bd_detector_destroy.argtypes = [ctypes.c_void_p]
        self.lib.bd_detector_get_stats.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self._Stats),
        ]

        raw_version = self.lib.bd_version_string()
        self.version = raw_version.decode() if raw_version else "unknown"
        self.handle = None
        self.width = 0
        self.max_detections = max_detections
        self.out = (self._Detection * max_detections)()

    def prepare(self, width: int, height: int) -> None:
        self.close()
        config = self._Config()
        self.lib.bd_config_init(ctypes.byref(config))
        config.max_detections = self.max_detections
        status = ctypes.c_int32(0)
        handle = self.lib.bd_detector_create(
            0, width, height, ctypes.byref(config), ctypes.byref(status)
        )
        if not handle:
            raise RuntimeError(f"bd_detector_create failed with status {status.value}")
        self.handle = handle
        self.width = width

    def detect(self, image: np.ndarray) -> list[Detection]:
        count = self.lib.bd_detector_detect_host(
            self.handle,
            image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            self.width,
            self.out,
            self.max_detections,
        )
        if count < 0:
            raise RuntimeError(f"bd_detector_detect_host failed with status {count}")
        found = []
        for i in range(min(count, self.max_detections)):
            corners = np.array(self.out[i].corners, dtype=np.float64).reshape(4, 2)
            found.append(Detection(int(self.out[i].id), corners))
        return found

    def stats(self) -> dict:
        stats = self._Stats()
        self.lib.bd_detector_get_stats(self.handle, ctypes.byref(stats))
        return {
            "runs": int(stats.runs),
            "candidates": int(stats.candidates),
            "quads": int(stats.quads),
            "roi_rows": int(stats.roi_rows),
            "gpu_ms": float(stats.average_gpu_ms),
        }

    def close(self) -> None:
        if self.handle:
            self.lib.bd_detector_destroy(self.handle)
            self.handle = None


class NapaBackend(Backend):
    def __init__(self, min_tag_px: int = 32) -> None:
        from napa_vision import NapaDetector

        self.detector = NapaDetector(min_tag_px=min_tag_px)
        self.name = "NAPA" if min_tag_px == 32 else f"NAPA (min_tag_px={min_tag_px})"
        self.version = importlib.metadata.version("napa-vision")

    def detect(self, image: np.ndarray) -> list[Detection]:
        return [
            Detection(int(tag.tag_id), np.asarray(tag.corners, dtype=np.float64).reshape(4, 2))
            for tag in self.detector.detect(image)
        ]


class ArucoBackend(Backend):
    name = "ArUco (subpixel)"

    def __init__(self) -> None:
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)
        self.version = cv2.__version__

    def detect(self, image: np.ndarray) -> list[Detection]:
        corners, ids, _ = self.detector.detectMarkers(image)
        if ids is None:
            return []
        return [
            Detection(int(tag_id), np.asarray(c, dtype=np.float64).reshape(4, 2))
            for tag_id, c in zip(ids.ravel(), corners)
        ]


class AprilTagBackend(Backend):
    name = "AprilTag 3 (full res)"

    def __init__(self) -> None:
        import pupil_apriltags

        # quad_decimate=1.0 is full resolution; the library default of 2.0 is
        # faster but discards accuracy, so it is not the fair comparison here.
        self.detector = pupil_apriltags.Detector(
            families="tag36h11", nthreads=1, quad_decimate=1.0, refine_edges=1
        )
        self.version = importlib.metadata.version("pupil-apriltags")

    def detect(self, image: np.ndarray) -> list[Detection]:
        return [
            Detection(
                int(tag.tag_id),
                np.asarray(tag.corners, dtype=np.float64).reshape(4, 2),
            )
            for tag in self.detector.detect(image, estimate_tag_pose=False)
        ]


def make_backend(key: str, args: argparse.Namespace) -> Backend:
    if key == "bdcudatag":
        return BdcudatagBackend(args.bdcudatag_lib)
    if key == "napa":
        return NapaBackend(32)
    if key == "napa16":
        return NapaBackend(16)
    if key == "aruco":
        return ArucoBackend()
    if key == "apriltag":
        return AprilTagBackend()
    raise ValueError(f"unknown detector: {key}")


def per_tag_error(truth: np.ndarray, detected: np.ndarray) -> float:
    """Mean corner distance with corners compared as an unordered set."""
    return min(
        float(np.linalg.norm(detected[list(order)] - truth, axis=1).mean())
        for order in CORNER_ORDERS
    )


def tag_side(corners: np.ndarray) -> float:
    return 0.5 * (
        np.linalg.norm(corners[1] - corners[0]) + np.linalg.norm(corners[2] - corners[1])
    )


def score(scene: dict, detections: list[Detection]) -> tuple[int, int, int, list[float]]:
    truth = [(tag["id"], np.asarray(tag["corners"], dtype=np.float64)) for tag in scene["tags"]]
    known_ids = {tag_id for tag_id, _ in truth}
    wrong_ids = sum(1 for detection in detections if detection.tag_id not in known_ids)

    found = 0
    errors: list[float] = []
    for tag_id, corners in truth:
        matches = [d.corners for d in detections if d.tag_id == tag_id]
        centre = corners.mean(axis=0)
        limit = 0.75 * tag_side(corners)
        matches = [c for c in matches if np.linalg.norm(c.mean(axis=0) - centre) <= limit]
        if not matches:
            continue
        found += 1
        errors.append(min(per_tag_error(corners, c) for c in matches))
    return found, len(truth), wrong_ids, errors


def load_scene_sets(directory: str) -> list[dict]:
    if cv2 is None:
        raise SystemExit("OpenCV (cv2) is required to read the scene images")

    sets = []
    for manifest_path in sorted(glob.glob(os.path.join(directory, "scenes_*.json"))):
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        scenes = []
        for entry in manifest["scenes"]:
            path = os.path.join(directory, entry["file"])
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"could not read {path}")
            scenes.append(
                {
                    "image": np.ascontiguousarray(image),
                    "tags": entry["tags"],
                    "width": entry["width"],
                    "height": entry["height"],
                }
            )
        if not scenes:
            continue
        sets.append(
            {
                "label": f"{scenes[0]['width']}x{scenes[0]['height']}",
                "width": scenes[0]["width"],
                "height": scenes[0]["height"],
                "scenes": scenes,
            }
        )
    sets.sort(key=lambda s: (s["width"], s["height"]))
    if not sets:
        raise SystemExit(
            f"no scenes_*.json in {directory}; run bdcudatag_scenes {directory} first"
        )
    return sets


def run_backend(backend: Backend, scene_set: dict, frames: int) -> dict:
    width, height = scene_set["width"], scene_set["height"]
    backend.prepare(width, height)
    scenes = scene_set["scenes"]
    count = len(scenes)

    for i in range(WARMUP_FRAMES):
        backend.detect(scenes[i % count]["image"])

    times = []
    for i in range(frames):
        image = scenes[i % count]["image"]
        start = time.perf_counter()
        backend.detect(image)
        times.append((time.perf_counter() - start) * 1000.0)

    found = total = wrong = 0
    errors: list[float] = []
    for scene in scenes:
        scene_found, scene_total, scene_wrong, scene_errors = score(
            scene, backend.detect(scene["image"])
        )
        found += scene_found
        total += scene_total
        wrong += scene_wrong
        errors.extend(scene_errors)

    times_sorted = np.sort(np.asarray(times))
    return {
        "resolution": scene_set["label"],
        "detector": backend.name,
        "version": backend.version,
        "median_ms": float(np.median(times_sorted)),
        "p95_ms": float(np.percentile(times_sorted, 95)),
        "recall": found / total if total else 0.0,
        "found": found,
        "tags": total,
        "corner_median_px": float(np.median(errors)) if errors else None,
        "wrong_ids": wrong,
    }


def format_table(results: list[dict]) -> str:
    lines = [
        "| Resolution | Detector | Median (ms) | p95 (ms) | Tags found | "
        "Corner error (px) | Wrong IDs |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        corner = (
            f"{result['corner_median_px']:.2f}"
            if result["corner_median_px"] is not None
            else "n/a"
        )
        lines.append(
            f"| {result['resolution']} | {result['detector']} | "
            f"{result['median_ms']:.2f} | {result['p95_ms']:.2f} | "
            f"{100.0 * result['recall']:.1f}% | {corner} | {result['wrong_ids']} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", default="scenes", help="scene directory")
    parser.add_argument("--frames", type=int, default=200, help="timed frames per resolution")
    parser.add_argument(
        "--detectors",
        default=",".join(DETECTOR_ORDER),
        help=f"comma-separated subset of {', '.join(DETECTOR_ORDER)}",
    )
    parser.add_argument(
        "--bdcudatag-lib",
        default=None,
        help="path to bdcudatag.dll / libbdcudatag.so (default: detector/build/Release)",
    )
    parser.add_argument("--markdown", default=None, help="write the table to this file")
    parser.add_argument("--json", default=None, help="write raw results to this file")
    args = parser.parse_args()

    if cv2 is None:
        print("error: OpenCV is required (`pip install opencv-python`)", file=sys.stderr)
        return 2

    if args.bdcudatag_lib is None:
        here = Path(__file__).resolve().parent.parent
        candidates = [
            here / "build" / "Release" / "bdcudatag.dll",
            here / "build" / "bdcudatag.dll",
            here / "build" / "libbdcudatag.so",
            here / "build" / "Release" / "libbdcudatag.so",
        ]
        existing = [path for path in candidates if path.exists()]
        if not existing:
            print(
                "error: no built bdcudatag library found; pass --bdcudatag-lib",
                file=sys.stderr,
            )
            return 2
        args.bdcudatag_lib = str(existing[0])

    keys = [key.strip() for key in args.detectors.split(",") if key.strip()]
    unknown = [key for key in keys if key not in DETECTOR_ORDER]
    if unknown:
        print(f"error: unknown detectors: {', '.join(unknown)}", file=sys.stderr)
        return 2

    scene_sets = load_scene_sets(args.scenes)
    print(
        f"scene sets: {', '.join(s['label'] for s in scene_sets)} "
        f"({len(scene_sets[0]['scenes'])} scenes each)",
        file=sys.stderr,
    )
    print(
        f"host: {platform.system()} {platform.machine()}, Python "
        f"{platform.python_version()}, {args.frames} timed frames",
        file=sys.stderr,
    )

    backends: list[tuple[str, Backend]] = []
    for key in keys:
        try:
            backends.append((key, make_backend(key, args)))
        except Exception as error:  # noqa: BLE001 - optional backends may be absent
            print(f"skip {key}: {error}", file=sys.stderr)

    if not backends:
        print("error: no backend could be loaded", file=sys.stderr)
        return 1

    results = []
    for scene_set in scene_sets:
        for key, backend in backends:
            try:
                results.append(run_backend(backend, scene_set, args.frames))
            finally:
                backend.close()
            print(
                f"  {scene_set['label']:<10} {backend.name:<24} "
                f"{results[-1]['median_ms']:6.2f} ms median",
                file=sys.stderr,
            )

    table = format_table(results)
    print(table)
    if args.markdown:
        Path(args.markdown).write_text(table + "\n", encoding="utf-8")
        print(f"table written to {args.markdown}", file=sys.stderr)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"raw results written to {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
