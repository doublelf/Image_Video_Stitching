"""Command line interface.

    python -m hstitch split  --input test.png --parts 3 --overlap 0.15 --out-dir parts
    python -m hstitch stitch --input parts/part0.png parts/part1.png --out out.png
    python -m hstitch stitch --input parts --out out.png --mode blend --manifest parts/manifest.json
    python -m hstitch demo   --input test.png --out out.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .blend import seam_energy
from .io import imread, imwrite
from .metrics import psnr
from .pipeline import StitchOptions, manifest_overlaps, resolve_inputs, stitch
from .split import pair_overlaps, slice_bounds, split_image, split_to_dir


def _background(value: str) -> tuple:
    text = value.strip()
    if text.startswith("#"):
        text = text[1:]
    if len(text) == 6:
        blue = int(text[4:6], 16)
        green = int(text[2:4], 16)
        red = int(text[0:2], 16)
        return (blue, green, red)
    parts = [int(item) for item in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("background must be 'RRGGBB' or 'b,g,r'")
    return tuple(parts)


def _overlaps(values: Optional[Sequence[int]], count: int) -> List[int]:
    if not values:
        return []
    if len(values) == 1:
        return [int(values[0])] * count
    if len(values) != count:
        raise SystemExit(
            f"expected {count} overlap values, got {len(values)}"
        )
    return [int(value) for value in values]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hstitch",
        description="Horizontal image concatenation and seam blending.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    split = sub.add_parser("split", help="cut an image into vertical slices")
    split.add_argument("--input", required=True, help="source image")
    split.add_argument("--parts", type=int, default=3, help="number of slices")
    split.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="overlap ratio relative to the slice width, for example 0.15",
    )
    split.add_argument("--out-dir", required=True, help="directory for the slices")
    split.add_argument("--prefix", default="part", help="file name prefix")
    split.add_argument("--ext", default=".png", help="file extension")

    run = sub.add_parser("stitch", help="stitch images side by side")
    run.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="images, directories or glob patterns, ordered from left to right",
    )
    run.add_argument("--out", required=True, help="output image")
    run.add_argument(
        "--mode",
        choices=("concat", "blend"),
        default="concat",
        help="concat places the tiles next to each other, blend fuses the overlaps",
    )
    run.add_argument("--height", type=int, default=None, help="target height in pixels")
    run.add_argument(
        "--height-strategy",
        choices=("min", "max", "first"),
        default="min",
        help="how the target height is chosen when --height is not given",
    )
    run.add_argument("--gap", type=int, default=0, help="gap between the tiles")
    run.add_argument(
        "--align",
        choices=("top", "center", "bottom"),
        default="center",
        help="vertical alignment of the shorter tiles",
    )
    run.add_argument(
        "--bg",
        type=_background,
        default=(0, 0, 0),
        help="background colour as RRGGBB or b,g,r",
    )
    run.add_argument(
        "--blend",
        choices=("multiband", "feather"),
        default="multiband",
        help="blender used by the blend mode",
    )
    run.add_argument(
        "--overlap-px",
        nargs="+",
        type=int,
        default=None,
        help="horizontal overlap between the neighbours, one value or one per pair",
    )
    run.add_argument(
        "--manifest",
        default=None,
        help="split manifest used to read the overlaps of the tiles",
    )
    run.add_argument(
        "--recursive",
        action="store_true",
        help="search the input directories recursively",
    )

    demo = sub.add_parser(
        "demo", help="split an image into three slices and stitch them back"
    )
    demo.add_argument("--input", required=True, help="source image")
    demo.add_argument("--out", required=True, help="output image")
    demo.add_argument("--parts", type=int, default=3, help="number of slices")
    demo.add_argument(
        "--overlap",
        type=float,
        default=0.15,
        help="overlap ratio relative to the slice width",
    )
    demo.add_argument(
        "--blend",
        choices=("multiband", "feather"),
        default="multiband",
        help="blender used to fuse the overlaps",
    )

    video = sub.add_parser("video", help="stitch several videos into one video")
    video.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="videos ordered from left to right",
    )
    video.add_argument("--out", required=True, help="output mp4")
    video.add_argument(
        "--mode",
        choices=("concat", "blend"),
        default="blend",
        help="concat places the views next to each other, blend fuses the overlaps",
    )
    video.add_argument(
        "--blend",
        choices=("multiband", "feather"),
        default="multiband",
        help="blender used by the blend mode",
    )
    video.add_argument(
        "--sync",
        choices=("timestamp", "index"),
        default="timestamp",
        help="how the frames of the sources are aligned",
    )
    video.add_argument("--height", type=int, default=None, help="common height")
    video.add_argument(
        "--height-strategy",
        choices=("min", "max", "first"),
        default="min",
        help="how the common height is chosen",
    )
    video.add_argument("--fps", type=float, default=None, help="output frame rate")
    video.add_argument(
        "--overlap-px",
        nargs="+",
        type=int,
        default=None,
        help="manual overlap in pixels, one value or one per pair",
    )
    video.add_argument(
        "--no-auto-overlap",
        action="store_true",
        help="do not estimate the overlap automatically",
    )
    video.add_argument(
        "--estimate-frames",
        type=int,
        default=3,
        help="number of frame pairs used for the overlap estimate",
    )
    video.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="stop after this many output frames",
    )
    video.add_argument(
        "--fourcc",
        default="mp4v",
        help="fourcc of the output codec",
    )

    serve = sub.add_parser("serve", help="start the upload service")
    serve.add_argument("--host", default="127.0.0.1", help="bind address")
    serve.add_argument("--port", type=int, default=8000, help="bind port")
    serve.add_argument("--reload", action="store_true", help="reload on changes")
    serve.add_argument(
        "--job-root",
        default=None,
        help="directory for the uploads and the results",
    )
    return parser


def command_split(args: argparse.Namespace) -> int:
    manifest = split_to_dir(
        source=args.input,
        out_dir=args.out_dir,
        parts=args.parts,
        overlap=args.overlap,
        prefix=args.prefix,
        ext=args.ext,
    )
    print(
        f"wrote {len(manifest['files'])} slices to {args.out_dir} "
        f"(overlap {manifest['overlap_px']} px, manifest {manifest['manifest_path']})"
    )
    return 0


def command_stitch(args: argparse.Namespace) -> int:
    inputs = resolve_inputs(args.input, recursive=args.recursive)
    if not inputs:
        raise SystemExit("no input images found")
    overlaps: Optional[List[int]] = None
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        overlaps = manifest_overlaps(manifest)
    if args.overlap_px:
        overlaps = _overlaps(args.overlap_px, max(0, len(inputs) - 1))

    options = StitchOptions(
        mode=args.mode,
        height=args.height,
        height_strategy=args.height_strategy,
        gap=args.gap,
        align=args.align,
        background=tuple(args.bg),
        blend=args.blend,
        overlaps=overlaps,
    )
    images = [imread(path) for path in inputs]
    result = stitch(images, options)
    imwrite(args.out, result)
    print(
        f"stitched {len(inputs)} tiles -> {args.out} ({result.shape[1]}x{result.shape[0]})"
    )
    return 0


def command_demo(args: argparse.Namespace) -> int:
    source = imread(args.input)
    height, width = source.shape[:2]
    slices = split_image(source, parts=args.parts, overlap=args.overlap)
    bounds = slice_bounds(
        width, args.parts, int(round(args.overlap * max(1, width // args.parts)))
    )
    overlaps = pair_overlaps(bounds)

    concat_options = StitchOptions(mode="concat")
    blend_options = StitchOptions(mode="blend", blend=args.blend, overlaps=overlaps)
    concat_result = stitch(slices, concat_options)
    blend_result = stitch(slices, blend_options)

    out_path = Path(args.out)
    imwrite(out_path, blend_result)
    concat_path = out_path.with_name(f"{out_path.stem}_concat{out_path.suffix}")
    imwrite(concat_path, concat_result)

    seams = []
    if len(slices) > 1:
        offset = 0
        for index, tile in enumerate(slices[:-1]):
            offset += tile.shape[1]
            seams.append(offset - overlaps[index] // 2)
    seam_scores = seam_energy(blend_result, seams)

    print(f"source     : {width}x{height}")
    print(f"slices     : {len(slices)} (overlap {overlaps} px)")
    print(f"concat     : {concat_path} ({concat_result.shape[1]}x{concat_result.shape[0]})")
    print(f"blend      : {out_path} ({blend_result.shape[1]}x{blend_result.shape[0]})")
    if blend_result.shape == source.shape:
        print(f"psnr       : {psnr(source, blend_result):.2f} dB")
    print(f"seam score : {[round(value, 2) for value in seam_scores]}")
    return 0


def command_video(args: argparse.Namespace) -> int:
    from .video.pipeline import VideoStitchOptions, stitch_videos

    overlaps = _overlaps(args.overlap_px, max(0, len(args.input) - 1)) if args.overlap_px else None
    options = VideoStitchOptions(
        mode=args.mode,
        sync=args.sync,
        fps=args.fps,
        height=args.height,
        height_strategy=args.height_strategy,
        blend=args.blend,
        auto_overlap=not args.no_auto_overlap,
        overlap_px=overlaps,
        estimate_frames=args.estimate_frames,
        max_frames=args.max_frames,
        fourcc=args.fourcc,
    )
    report = stitch_videos(args.input, args.out, options)
    print(
        f"stitched {len(args.input)} videos -> {report['output']} "
        f"({report['size'][0]}x{report['size'][1]}, {report['frames']} frames, "
        f"{report['fps']:.2f} fps)"
    )
    for index, alignment in enumerate(report["alignments"]):
        if alignment:
            print(
                f"  pair {index}: dx={alignment['dx']} dy={alignment['dy']} "
                f"overlap={alignment['overlap']} "
                f"({alignment['method']}, score {alignment['score']})"
            )
        else:
            print(f"  pair {index}: no overlap found, the views are placed side by side")
    return 0


def command_serve(args: argparse.Namespace) -> int:
    import uvicorn

    if args.job_root:
        os.environ["HSTITCH_JOB_ROOT"] = str(Path(args.job_root).resolve())
    uvicorn.run(
        "hstitch.server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "split":
        return command_split(args)
    if args.command == "stitch":
        return command_stitch(args)
    if args.command == "demo":
        return command_demo(args)
    if args.command == "video":
        return command_video(args)
    if args.command == "serve":
        return command_serve(args)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
