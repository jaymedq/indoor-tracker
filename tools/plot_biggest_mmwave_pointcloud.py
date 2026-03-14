#!/usr/bin/env python3

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


try:
    from tools.constants import (
        RADAR_PLACEMENT,
        RADAR_FACING_ANGLE,
        RADAR_FOV,
        FOV_RADIUS,
    )
except ImportError:  # allows running from within tools/
    from constants import (  # type: ignore
        RADAR_PLACEMENT,
        RADAR_FACING_ANGLE,
        RADAR_FOV,
        FOV_RADIUS,
    )


@dataclass(frozen=True)
class BestFrame:
    file_path: Path
    row_index: int
    point_count: int
    timestamp_raw: str
    x: list
    y: list
    z: list


def _detect_sep(csv_path: Path) -> str:
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
    return ";" if ";" in header else ","


def _parse_list(value) -> list:
    """Parse a list stored as a string like "[1, 2, 3]".

    Uses ast.literal_eval for safety; falls back to a constrained eval to handle
    some datasets that may contain 'nan'.
    """
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if not isinstance(value, str):
        return [value]

    s = value.strip()
    if s == "" or s.lower() == "nan":
        return []

    try:
        parsed = ast.literal_eval(s)
        return parsed if isinstance(parsed, list) else [parsed]
    except Exception:
        try:
            parsed = eval(s, {"nan": float("nan")}, {})
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            return []


def _transform_to_room_coords(x: list, y: list, z: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.array([x, y, z], dtype=float)
    if pts.size == 0:
        return np.array([]), np.array([]), np.array([])

    xs = RADAR_PLACEMENT[0] + pts[0]
    ys = RADAR_PLACEMENT[1] - pts[1]
    zs = RADAR_PLACEMENT[2] + pts[2]
    return xs, ys, zs


def _iter_mmwave_files(data_dir: Path, include_root: bool) -> Iterable[Path]:
    files = list(data_dir.rglob("*_mmwave_data.csv")) if data_dir.exists() else []
    if include_root:
        files.extend(Path(".").glob("*_mmwave_data.csv"))
    # de-dup while preserving stable ordering
    unique = sorted({p.resolve() for p in files})
    return unique


def _best_frame_in_file(csv_path: Path) -> Optional[BestFrame]:
    sep = _detect_sep(csv_path)

    # Prefer numObj for speed (no need to parse every x list).
    wanted_cols = {"numObj", "timestamp", "x", "y", "z"}
    df = pd.read_csv(
        csv_path,
        sep=sep,
        usecols=lambda c: c in wanted_cols,
        low_memory=False,
    )
    if df.empty or "x" not in df.columns or "y" not in df.columns or "z" not in df.columns:
        return None

    best_idx: Optional[int] = None
    best_count: int = -1

    if "numObj" in df.columns and df["numObj"].notna().any():
        counts = pd.to_numeric(df["numObj"], errors="coerce")
        if counts.notna().any():
            best_idx = int(counts.idxmax())
            best_count = int(counts.loc[best_idx])

    if best_idx is None:
        # Fallback: compute from parsed x list lengths
        max_len = -1
        max_i = None
        for i, v in df["x"].items():
            lst = _parse_list(v)
            if len(lst) > max_len:
                max_len = len(lst)
                max_i = int(i)
        if max_i is None:
            return None
        best_idx = max_i
        best_count = max_len

    row = df.loc[best_idx]
    x = _parse_list(row["x"])
    y = _parse_list(row["y"])
    z = _parse_list(row["z"])

    # Some rows can be inconsistent; use the minimum length to avoid crashes.
    n = min(len(x), len(y), len(z))
    if n <= 0:
        return None
    x, y, z = x[:n], y[:n], z[:n]

    ts = str(row["timestamp"]) if "timestamp" in df.columns else ""

    return BestFrame(
        file_path=csv_path,
        row_index=best_idx,
        point_count=int(best_count if best_count >= 0 else n),
        timestamp_raw=ts,
        x=x,
        y=y,
        z=z,
    )


def _plot_radar_fov(ax) -> None:
    """2D FOV helper (kept for reference)."""
    from matplotlib.patches import Wedge

    theta1 = RADAR_FACING_ANGLE - RADAR_FOV / 2
    theta2 = RADAR_FACING_ANGLE + RADAR_FOV / 2

    # EPS does not support transparency; draw the FOV as an unfilled outline.
    wedge = Wedge(
        center=(float(RADAR_PLACEMENT[0]), float(RADAR_PLACEMENT[1])),
        r=float(FOV_RADIUS),
        theta1=float(theta1),
        theta2=float(theta2),
        fill=False,
        edgecolor="cyan",
        linestyle="--",
        linewidth=1.0,
    )
    ax.add_patch(wedge)
    ax.scatter(
        [float(RADAR_PLACEMENT[0])],
        [float(RADAR_PLACEMENT[1])],
        c="k",
        marker="x",
        s=80,
        label="Radar",
    )


def _plot_radar_fov_3d(ax, z0: float) -> None:
    """Plot radar position and FOV outline on the XY plane at height z0."""
    radar_x = float(RADAR_PLACEMENT[0])
    radar_y = float(RADAR_PLACEMENT[1])

    ax.scatter([radar_x], [radar_y], [float(z0)], c="k", marker="x", s=80, label="Radar")

    theta1 = np.deg2rad(float(RADAR_FACING_ANGLE - RADAR_FOV / 2))
    theta2 = np.deg2rad(float(RADAR_FACING_ANGLE + RADAR_FOV / 2))
    r = float(FOV_RADIUS)

    for t in (theta1, theta2):
        x1 = radar_x + r * np.cos(t)
        y1 = radar_y + r * np.sin(t)
        ax.plot([radar_x, x1], [radar_y, y1], [z0, z0], color="cyan", linestyle="--", linewidth=1.0)

    ts = np.linspace(theta1, theta2, 60)
    xs = radar_x + r * np.cos(ts)
    ys = radar_y + r * np.sin(ts)
    zs = np.full_like(xs, float(z0))
    ax.plot(xs, ys, zs, color="cyan", linestyle="--", linewidth=1.0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Find the biggest mmWave pointcloud (max numObj in a single frame) "
            "across *_mmwave_data.csv files and save that frame plot to EPS."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Results"),
        help="Directory to scan recursively for *_mmwave_data.csv (default: Results)",
    )
    parser.add_argument(
        "--include-root",
        action="store_true",
        help="Also scan repo root for *_mmwave_data.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("biggest_mmwave_pointcloud.eps"),
        help="Output EPS path (default: biggest_mmwave_pointcloud.eps)",
    )
    parser.add_argument(
        "--no-transform",
        action="store_true",
        help="Do not transform points to room coordinates (no RADAR_PLACEMENT offset/inversion)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively after saving",
    )
    parser.add_argument("--xlim", nargs=2, type=float, default=[0.0, 10.0])
    parser.add_argument("--ylim", nargs=2, type=float, default=[-10.0, 0.0])
    parser.add_argument("--zlim", nargs=2, type=float, default=[-1.0, 6.0])

    args = parser.parse_args()

    if not args.show:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    files = list(_iter_mmwave_files(args.data_dir, args.include_root))
    if not files:
        raise SystemExit(f"No *_mmwave_data.csv files found under {args.data_dir} (include_root={args.include_root}).")

    global_best: Optional[BestFrame] = None
    for f in files:
        best = _best_frame_in_file(f)
        if best is None:
            continue
        if global_best is None or best.point_count > global_best.point_count:
            global_best = best

    if global_best is None:
        raise SystemExit("Could not find any non-empty pointcloud frames.")

    print("Biggest pointcloud found:")
    print(f"  file: {global_best.file_path}")
    print(f"  row_index: {global_best.row_index}")
    print(f"  timestamp: {global_best.timestamp_raw}")
    print(f"  point_count: {global_best.point_count}")

    x, y, z = global_best.x, global_best.y, global_best.z
    if args.no_transform:
        xs = np.array(x, dtype=float)
        ys = np.array(y, dtype=float)
        zs = np.array(z, dtype=float)
    else:
        xs, ys, zs = _transform_to_room_coords(x, y, z)

    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_subplot(111, projection="3d")

    # Make 3D panes fully opaque for EPS output.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor((0.0, 0.0, 0.0, 1.0))

    _plot_radar_fov_3d(ax, z0=float(RADAR_PLACEMENT[2]))

    ax.scatter(xs, ys, zs, c="b", marker="o", s=12, depthshade=False, label="Point Cloud")

    # if len(xs) > 0:
    #     ax.scatter(
    #         [float(np.mean(xs))],
    #         [float(np.mean(ys))],
    #         [float(np.mean(zs))],
    #         c="r",
    #         marker="^",
    #         s=60,
    #         depthshade=False,
    #         label="Centroid",
    #     )

    ax.set_xlim(args.xlim)
    ax.set_ylim(args.ylim)
    ax.set_zlim(args.zlim)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    ax.view_init(elev=22, azim=-55)

    ax.legend(loc="upper right", framealpha=1.0)

    fig.tight_layout()
    fig.savefig(args.output, format="eps")
    print(f"Saved EPS: {args.output}")

    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
