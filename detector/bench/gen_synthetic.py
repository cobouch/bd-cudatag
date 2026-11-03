#!/usr/bin/env python3
"""Generate a hard synthetic AprilTag corpus in NAPA's directory format.

NAPA's corpus stops at moderately hard conditions, and bdcudatag's remaining
misses are concentrated in four families at the low resolutions: lifted black
levels, heavy sensor noise, underexposure and small tags.  This generator
re-renders those families with harsher parameters, so a regression shows up in
seconds on synthetic data instead of on a camera.

The output layout matches `datasets/napa-benchmark`, so `eval_napa.py` scores it
without changes:

    datasets/synthetic-hard/
      data/hard/metadata.jsonl
      data/hard/<scenario>-<W>x<H>-<idx>.png

Rendering is 4x supersampled: each tag bitmap is warped into a 4x canvas with
`warpPerspective`, then the canvas is area-downsampled.  Ground-truth corners
are the black square's corners under that same homography, in NAPA's
pixel-centre convention (the centre of the top-left pixel is (0, 0)), so the
labels are exact for the rendered pixels rather than an estimate.

Usage:
    python gen_synthetic.py                       # 640x480 + 1280x720
    python gen_synthetic.py --res 640x480 --frames 10 --scenarios noise tiny
    python gen_synthetic.py --out datasets/synthetic-hard --frames 40
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

SUPERSAMPLE = 4
TAG_CELLS = 8  # tag36h11 marker incl. its one-cell black border
FAMILY = cv2.aruco.DICT_APRILTAG_36h11

# name -> rendering parameters.  Tag sizes are in pixels at 640 px width and
# scale with the frame, so a scenario has the same relative size at every
# resolution.  Every field is optional except count/side.
SCENARIOS: dict[str, dict] = {
    # The resolution floor: clean image, tags at 10-22 px.
    "tiny": dict(count=(2, 4), side=(10, 22), bg=248, noise=2.0),
    # Lifted black level with a bright background (NAPA's hardest scenario).
    "lowcontrast": dict(count=(1, 3), side=(40, 140), bg=242,
                        black=(96, 118), white=(160, 182), noise=2.0),
    # Dark frame: tag white cells sit well below the bright background.
    "dark-noisy": dict(count=(1, 3), side=(40, 150), bg=(28, 48),
                       black=0, white=(52, 88), noise=(5, 9)),
    # Sensor noise on a textured background at full tag contrast.
    "noise": dict(count=(1, 3), side=(40, 160), bg="texture", noise=(15, 22)),
    # Deep underexposure, as from a short exposure in a dim room.
    "underexposed": dict(count=(1, 3), side=(45, 190), bg="texture",
                         gain=(0.22, 0.34), noise=(3, 6)),
    # Optical blur wide enough to smear a cell.
    "blur": dict(count=(1, 3), side=(18, 60), bg=250, blur=(1.1, 2.2), noise=2.0),
    # Linear motion blur at a random angle.
    "motion": dict(count=(1, 3), side=(24, 70), bg="texture",
                   motion=(6, 15), noise=3.0),
    # Low JPEG quality: ringing right next to the border.
    "jpeg": dict(count=(1, 3), side=(30, 110), bg="texture",
                 jpeg=(8, 20), noise=2.0),
    # A shadow edge crossing the tag.
    "shadow": dict(count=(1, 3), side=(40, 150), bg="texture",
                   shadow=(0.30, 0.55), noise=3.0),
    # Saturated specular highlight over part of a tag.
    "glare": dict(count=(1, 3), side=(40, 160), bg="texture",
                  glare=(0.35, 0.7), noise=3.0),
    # Quiet zone clipped by the image border (black square stays inside).
    "edge": dict(count=(1, 3), side=(36, 90), bg=246, edge_tag=True, noise=2.0),
    # Many small tags: candidate crowding and cap pressure.
    "dense": dict(count=(12, 20), side=(22, 44), bg="texture", noise=4.0),
    # Everything at once: dim, noisy, slightly blurred, small.
    "mixed": dict(count=(2, 4), side=(14, 34), bg="texture", gain=(0.35, 0.5),
                  blur=(0.5, 1.0), noise=(8, 12)),
}

DICTIONARY = cv2.aruco.getPredefinedDictionary(FAMILY)
MARKER_CACHE: dict[tuple[int, int], np.ndarray] = {}


def marker_bitmap(tag_id: int, pixels: int) -> np.ndarray:
    """Marker (with its black border) at least `pixels` across, cached."""
    bucket = max(32, int(math.ceil(pixels / 16.0)) * 16)
    key = (tag_id, bucket)
    cached = MARKER_CACHE.get(key)
    if cached is None:
        if len(MARKER_CACHE) > 96:
            MARKER_CACHE.clear()
        cached = cv2.aruco.generateImageMarker(DICTIONARY, tag_id, bucket)
        MARKER_CACHE[key] = cached
    return cached


def tag_patch(tag_id: int, side: float, black: float, white: float):
    """Marker plus a one-cell white quiet zone, as an opaque float patch.

    Returns (patch, marker_px) so the caller can build the exact homography:
    the marker occupies `marker_px` pixels in the middle of the patch and the
    quiet zone is one cell (marker_px / 8) around it.
    """
    marker_px = max(TAG_CELLS * 4, int(round(side * SUPERSAMPLE)))
    marker = marker_bitmap(tag_id, marker_px)
    if marker.shape[0] != marker_px:
        marker = cv2.resize(marker, (marker_px, marker_px), interpolation=cv2.INTER_AREA)
    marker = black + marker.astype(np.float32) * (white - black) / 255.0
    quiet = marker_px // TAG_CELLS
    patch = cv2.copyMakeBorder(marker, quiet, quiet, quiet, quiet,
                               cv2.BORDER_CONSTANT, value=float(white))
    return patch.astype(np.float32), marker_px


def unit_to_quad(quad: np.ndarray) -> np.ndarray:
    src = np.float32([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    return cv2.getPerspectiveTransform(src, quad.astype(np.float32))


def render_tag(canvas: np.ndarray, patch: np.ndarray, marker_px: int,
               quad4: np.ndarray) -> None:
    """Warp `patch` onto the 4x canvas so its black square lands on `quad4`."""
    height, width = canvas.shape
    quiet = marker_px // TAG_CELLS
    to_marker = np.array([[1.0 / marker_px, 0.0, -quiet / marker_px],
                          [0.0, 1.0 / marker_px, -quiet / marker_px],
                          [0.0, 0.0, 1.0]])
    H = unit_to_quad(quad4) @ to_marker
    warped = cv2.warpPerspective(patch, H, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    mask = cv2.warpPerspective(np.full(patch.shape, 255.0, np.float32), H,
                               (width, height), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    alpha = (mask > 127.0)
    canvas[alpha] = warped[alpha]


def textured_background(rng: np.random.Generator, width: int, height: int,
                        base: float) -> np.ndarray:
    """Low-frequency blotches plus a few dark strokes, like NAPA's `texture`."""
    small = rng.normal(0.0, 1.0, (max(2, height // 48), max(2, width // 48)))
    field = cv2.resize(small.astype(np.float32), (width, height),
                       interpolation=cv2.INTER_CUBIC)
    field = cv2.GaussianBlur(field, (0, 0), 3.0)
    field *= 16.0 / (field.std() + 1e-6)
    image = np.clip(base + field, 0, 255).astype(np.float32)
    for _ in range(int(rng.integers(3, 8))):
        x0, y0 = (int(v) for v in rng.integers(0, max(1, min(width, height)), 2))
        x1 = int(np.clip(x0 + rng.normal(0, width * 0.12), 0, width - 1))
        y1 = int(np.clip(y0 + rng.normal(0, height * 0.12), 0, height - 1))
        value = float(np.clip(base - rng.uniform(40, 130), 0, 255))
        cv2.line(image, (x0, y0), (x1, y1), value, int(rng.integers(1, 4)),
                 cv2.LINE_AA)
    return image


def text_background(rng: np.random.Generator, width: int, height: int,
                    base: float) -> np.ndarray:
    image = np.full((height, width), base, np.float32)
    for _ in range(int(rng.integers(6, 16))):
        text = "".join(chr(int(c)) for c in rng.integers(65, 90, int(rng.integers(3, 9))))
        x = int(rng.integers(0, max(1, width - 40)))
        y = int(rng.integers(10, max(11, height - 10)))
        value = int(np.clip(base - rng.uniform(60, 160), 0, 255))
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    float(rng.uniform(0.3, 1.1)), value, int(rng.integers(1, 3)),
                    cv2.LINE_AA)
    return image


def make_quad(rng: np.random.Generator, cx: float, cy: float, side: float) -> np.ndarray:
    """Rotated, slightly squashed and perspective-skewed square."""
    angle = float(rng.uniform(0.0, 2.0 * math.pi))
    squash = float(rng.uniform(0.88, 1.0))
    tilt = float(rng.uniform(0.0, 0.22))
    ca, sa = math.cos(angle), math.sin(angle)
    half = side * 0.5
    corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    corners[:, 0] *= squash
    corners[:, 0] += tilt * corners[:, 1]
    corners[:, 1] += tilt * corners[:, 0]
    return corners @ np.array([[ca, sa], [-sa, ca]]).T + [cx, cy]


def place_tags(rng: np.random.Generator, width: int, height: int, count: int,
               side_range: tuple[float, float], edge_tag: bool):
    """Non-overlapping placements; black squares always stay inside the frame.

    `edge_tag` pushes the first tag against an edge so its quiet zone is
    clipped, which is the one case NAPA's corpus documents as partially outside.
    """
    quads: list[np.ndarray] = []
    sides: list[float] = []
    attempts = 0
    while len(quads) < count and attempts < 6000:
        attempts += 1
        side = float(rng.uniform(*side_range))
        if edge_tag and not quads:
            # Put the black square's centre half a side from an edge, so the
            # square itself is inside but the quiet zone runs off the image.
            vertical = rng.random() < 0.5
            if vertical:
                cx = float(rng.uniform(side * 0.5 + 1, width - side * 0.5 - 1))
                cy = 1.0 + side * 0.5 if rng.random() < 0.5 else height - 1.0 - side * 0.5
            else:
                cy = float(rng.uniform(side * 0.5 + 1, height - side * 0.5 - 1))
                cx = 1.0 + side * 0.5 if rng.random() < 0.5 else width - 1.0 - side * 0.5
        else:
            reach = side * 0.75
            if width * height < count * (2.3 * side) ** 2:
                break
            cx = float(rng.uniform(reach + 4, width - reach - 4))
            cy = float(rng.uniform(reach + 4, height - reach - 4))

        quad = make_quad(rng, cx, cy, side)
        # The black square must stay fully inside, or ground truth is clipped.
        if (quad[:, 0].min() < 1 or quad[:, 0].max() > width - 2
                or quad[:, 1].min() < 1 or quad[:, 1].max() > height - 2):
            continue
        centre = quad.mean(axis=0)
        if any(np.abs(other.mean(axis=0) - centre).max() < 0.5 * (side + other_side) * 1.3
               for other, other_side in zip(quads, sides)):
            continue
        quads.append(quad)
        sides.append(float(np.mean(np.linalg.norm(quad - np.roll(quad, -1, 0), axis=1))))
    return quads, sides


def render_frame(rng: np.random.Generator, width: int, height: int, spec: dict):
    scale = width / 640.0
    side_range = (spec["side"][0] * scale, spec["side"][1] * scale)
    count = int(rng.integers(spec["count"][0], spec["count"][1] + 1))
    quads, sides = place_tags(rng, width, height, count, side_range,
                              spec.get("edge_tag", False))
    if not quads:
        quads, sides = place_tags(rng, width, height, 1, side_range, False)

    ids = rng.choice(587, size=len(quads), replace=False).tolist()

    bg = spec.get("bg", 248)
    if bg == "texture":
        background = textured_background(rng, width, height, 226.0)
    elif bg == "text":
        background = text_background(rng, width, height, 228.0)
    else:
        level = float(rng.uniform(*bg)) if isinstance(bg, tuple) else float(bg)
        background = np.full((height, width), level, np.float32)

    canvas = cv2.resize(background, (width * SUPERSAMPLE, height * SUPERSAMPLE),
                        interpolation=cv2.INTER_LINEAR)
    for tag_id, quad, side in zip(ids, quads, sides):
        black = spec.get("black", 0)
        white = spec.get("white", 255)
        if isinstance(black, tuple):
            black = float(rng.uniform(*black))
        if isinstance(white, tuple):
            white = float(rng.uniform(*white))
        patch, marker_px = tag_patch(int(tag_id), float(side), float(black), float(white))
        # 1x pixel-index coord p is 1x continuous p + 0.5, and 4x is 4x that.
        render_tag(canvas, patch, marker_px, (quad + 0.5) * SUPERSAMPLE)

    image = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)

    if "gain" in spec:
        image = image * float(rng.uniform(*spec["gain"]))
    if "shadow" in spec:
        strength = float(rng.uniform(*spec["shadow"]))
        ramp = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        image = image * (1.0 - strength * ramp)
    if "glare" in spec:
        strength = float(rng.uniform(*spec["glare"]))
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        cx, cy = float(rng.uniform(0.2, 0.8)) * width, float(rng.uniform(0.2, 0.8)) * height
        radius = float(rng.uniform(0.05, 0.16)) * width
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * radius * radius))
        image = image + blob * 255.0 * strength
    if "blur" in spec:
        image = cv2.GaussianBlur(image, (0, 0), float(rng.uniform(*spec["blur"])))
    if "motion" in spec:
        length = int(rng.uniform(*spec["motion"]))
        kernel = np.zeros((length, length), np.float32)
        kernel[length // 2, :] = 1.0
        rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5),
                                      float(rng.uniform(0, 180)), 1.0)
        kernel = cv2.warpAffine(kernel, rot, (length, length))
        kernel /= max(float(kernel.sum()), 1e-6)
        image = cv2.filter2D(image, -1, kernel)
    if "noise" in spec:
        sigma = spec["noise"]
        if isinstance(sigma, tuple):
            sigma = rng.uniform(*sigma)
        image = image + rng.normal(0.0, float(sigma), image.shape).astype(np.float32)

    image = np.clip(image, 0, 255).astype(np.uint8)
    if "jpeg" in spec:
        quality = int(rng.integers(spec["jpeg"][0], spec["jpeg"][1] + 1))
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    return image, ids, quads, sides


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(here / "datasets" / "synthetic-hard"))
    parser.add_argument("--res", nargs="*", default=["640x480", "1280x720"])
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args()

    unknown = [s for s in args.scenarios if s not in SCENARIOS]
    if unknown:
        print(f"unknown scenarios: {', '.join(unknown)}")
        return 2

    directory = Path(args.out) / "data" / "hard"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for res in args.res:
        width, height = (int(v) for v in res.lower().split("x"))
        for scenario in args.scenarios:
            spec = SCENARIOS[scenario]
            for index in range(args.frames):
                rng = np.random.default_rng(
                    [args.seed, width, height, sum(map(ord, scenario)), index])
                image, ids, quads, sides = render_frame(rng, width, height, spec)
                name = f"{scenario}-{width}x{height}-{index:03d}.png"
                cv2.imwrite(str(directory / name), image)
                records.append({
                    "file_name": name,
                    "frame_id": f"{scenario}-{width}x{height}-{index:03d}",
                    "scenario": scenario,
                    "frame_index": index,
                    "width": width,
                    "height": height,
                    "family": "tag36h11",
                    "seed": args.seed,
                    # Corners are pixel-index coordinates, matching NAPA.
                    "tag_ids": [int(i) for i in ids],
                    "corners": [q.astype(np.float64).tolist() for q in quads],
                    "mean_side_px": sides,
                    "num_tags": len(quads),
                    "details_json": json.dumps(
                        {k: v for k, v in spec.items() if isinstance(v, (int, float, str))}),
                })

    records.sort(key=lambda r: (r["width"], r["height"], r["scenario"], r["frame_index"]))
    with (directory / "metadata.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    frames = len(records)
    tags = sum(r["num_tags"] for r in records)
    print(f"wrote {frames} frames ({tags} tags) to {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
