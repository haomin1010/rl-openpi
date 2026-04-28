#!/usr/bin/env python3

"""Local delta-window exploration on a real action sequence from the LeRobot corpus.

Run this file with the pi05 virtualenv python because it depends on pyarrow:
    /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05/.venv/bin/python local_delta_real_dataset_demo.py

Exploration flow:
1. Load one real action trajectory from the dataset. The action itself is already delta-style.
2. Sample one smooth shared index-offset curve inside a local window around each timestep.
3. Re-sample a nearby reference delta sequence from the original delta trajectory with linear interpolation.
4. Estimate per-timestep, per-dimension local delta scale from a sliding window.
5. Add local Gaussian noise around that nearby delta reference.
6. Apply a light temporal smoothing filter to reduce high-frequency jitter.
7. Keep binary gripper dimensions exactly unchanged.
8. Integrate both original and sampled delta actions into pseudo-continuous trajectories only for visualization.

This design is intentionally simpler than the OU variants:
- no OU state
- no explicit drift control
- no endpoint anchoring
- local exploration comes from nearby delta re-sampling plus local Gaussian noise
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq


Trajectory = list[list[float]]


@dataclass
class LocalDeltaSamplerConfig:
    offset_radius: int = 6
    offset_scale_range: tuple[float, float] = (0.5, 1.0)
    sample_amplitude_range: tuple[float, float] = (0.65, 1.35)
    local_scale_window: int = 9
    local_scale_floor_ratio: float = 0.15
    local_noise_scale_range: tuple[float, float] = (0.15, 0.40)
    final_smooth_kernel: tuple[float, ...] = (0.2, 0.6, 0.2)
    num_samples: int = 6
    seed: int = 17


ACTION_NAMES = [
    "left_dx",
    "left_dy",
    "left_dz",
    "left_drotvec_x",
    "left_drotvec_y",
    "left_drotvec_z",
    "left_gripper_target",
    "right_dx",
    "right_dy",
    "right_dz",
    "right_drotvec_x",
    "right_drotvec_y",
    "right_drotvec_z",
    "right_gripper_target",
]

GRIPPER_DIMS = {6, 13}
DATASET_ROOT = Path("/mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus")
DEFAULT_EPISODE = 103


def value_range(values: list[float]) -> tuple[float, float]:
    return min(values), max(values)


def map_points(
    xs: list[float],
    ys: list[float],
    box: tuple[float, float, float, float],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> list[tuple[float, float]]:
    left, top, width, height = box
    xmin, xmax = xlim
    ymin, ymax = ylim
    xr = max(1e-9, xmax - xmin)
    yr = max(1e-9, ymax - ymin)
    points = []
    for x, y in zip(xs, ys):
        px = left + (x - xmin) / xr * width
        py = top + height - (y - ymin) / yr * height
        points.append((px, py))
    return points


def svg_polyline(points: list[tuple[float, float]], stroke: str, width: float, opacity: float) -> str:
    data = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (
        f'<polyline fill="none" stroke="{stroke}" stroke-width="{width:.2f}" '
        f'stroke-linecap="round" stroke-linejoin="round" opacity="{opacity:.3f}" points="{data}" />'
    )


def draw_axes(left: float, top: float, width: float, height: float, title: str, xlabel: str, ylabel: str) -> str:
    return f"""
    <rect x="{left:.1f}" y="{top:.1f}" width="{width:.1f}" height="{height:.1f}" rx="16" fill="#fffdfa" stroke="#d9d0c7" stroke-width="1.4"/>
    <text x="{left + 16:.1f}" y="{top + 28:.1f}" font-size="18" font-weight="700" fill="#1f2937">{title}</text>
    <text x="{left + width / 2:.1f}" y="{top + height + 34:.1f}" font-size="13" text-anchor="middle" fill="#4b5563">{xlabel}</text>
    <text x="{left - 44:.1f}" y="{top + height / 2:.1f}" font-size="13" text-anchor="middle" transform="rotate(-90 {left - 44:.1f} {top + height / 2:.1f})" fill="#4b5563">{ylabel}</text>
    """


def load_episode_actions(dataset_root: Path, episode_index: int) -> tuple[Trajectory, str]:
    episode_path = dataset_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    table = pq.read_table(episode_path, columns=["action"])
    actions = [list(row) for row in table.column("action").to_pylist()]

    task = ""
    with (dataset_root / "meta" / "episodes.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if int(row["episode_index"]) == episode_index:
                task = str(row["tasks"][0]) if row.get("tasks") else ""
                break
    return actions, task


def dimension_stats(actions: Trajectory) -> list[dict[str, float]]:
    dims = len(actions[0])
    stats = []
    for d in range(dims):
        vals = [step[d] for step in actions]
        vmin, vmax = value_range(vals)
        stats.append(
            {
                "min": vmin,
                "max": vmax,
                "range": vmax - vmin,
                "mean_abs": sum(abs(v) for v in vals) / len(vals),
            }
        )
    return stats


def pick_plot_dims(actions: Trajectory) -> tuple[int, int]:
    stats = dimension_stats(actions)
    candidates = [d for d in range(len(stats)) if d not in GRIPPER_DIMS]
    ordered = sorted(candidates, key=lambda d: (stats[d]["mean_abs"], stats[d]["range"]), reverse=True)
    return ordered[0], ordered[1]


def smooth_scalar_series(values: list[float], kernel: tuple[float, ...]) -> list[float]:
    if not values or len(kernel) % 2 == 0:
        return values[:]
    radius = len(kernel) // 2
    weight_sum = sum(kernel)
    if weight_sum <= 1e-9:
        return values[:]
    out = []
    length = len(values)
    for t in range(length):
        accum = 0.0
        for k, weight in enumerate(kernel):
            idx = min(length - 1, max(0, t + k - radius))
            accum += weight * values[idx]
        out.append(accum / weight_sum)
    return out


def smooth_trajectory(traj: Trajectory, kernel: tuple[float, ...], locked_dims: set[int]) -> Trajectory:
    if not traj or len(kernel) % 2 == 0:
        return [step[:] for step in traj]
    radius = len(kernel) // 2
    weight_sum = sum(kernel)
    if weight_sum <= 1e-9:
        return [step[:] for step in traj]
    length = len(traj)
    dims = len(traj[0])
    out = []
    for t in range(length):
        step = []
        for d in range(dims):
            if d in locked_dims:
                step.append(traj[t][d])
                continue
            accum = 0.0
            for k, weight in enumerate(kernel):
                idx = min(length - 1, max(0, t + k - radius))
                accum += weight * traj[idx][d]
            step.append(accum / weight_sum)
        out.append(step)
    return out


def delta_to_continuous(actions: Trajectory, locked_dims: set[int]) -> Trajectory:
    """Integrate delta actions into a pseudo-continuous trajectory."""
    if not actions:
        return []
    dims = len(actions[0])
    running = [0.0] * dims
    continuous = []
    for step in actions:
        current = []
        for d, value in enumerate(step):
            if d in locked_dims:
                current.append(value)
                continue
            running[d] += value
            current.append(running[d])
        continuous.append(current)
    return continuous


def continuous_to_delta(traj: Trajectory, original_actions: Trajectory, locked_dims: set[int]) -> Trajectory:
    """Differentiate a pseudo-continuous trajectory back into delta actions."""
    if not traj:
        return []
    dims = len(traj[0])
    prev = [0.0] * dims
    deltas = []
    for t, step in enumerate(traj):
        current = []
        for d, value in enumerate(step):
            if d in locked_dims:
                current.append(original_actions[t][d])
                continue
            current.append(value - prev[d])
            prev[d] = value
        deltas.append(current)
    return deltas


def compute_local_delta_scale(actions: Trajectory, window: int, floor_ratio: float, locked_dims: set[int]) -> list[list[float]]:
    length = len(actions)
    dims = len(actions[0])
    radius = max(1, window // 2)
    floors = []
    for d in range(dims):
        if d in locked_dims:
            floors.append(0.0)
            continue
        mean_abs_delta = sum(abs(actions[t][d]) for t in range(length)) / max(length, 1)
        floors.append(floor_ratio * max(mean_abs_delta, 1e-4))

    local_scale = []
    for t in range(length):
        lo = max(0, t - radius)
        hi = min(length - 1, t + radius)
        span = hi - lo + 1
        step = []
        for d in range(dims):
            if d in locked_dims:
                step.append(0.0)
                continue
            mean_abs = sum(abs(actions[i][d]) for i in range(lo, hi + 1)) / span
            step.append(max(mean_abs, floors[d]))
        local_scale.append(step)
    return local_scale


def sample_offset_curve(length: int, config: LocalDeltaSamplerConfig, rng: random.Random) -> list[float]:
    raw = [rng.gauss(0.0, 1.0) for _ in range(length)]
    offsets = smooth_scalar_series(raw, (1.0, 2.0, 3.0, 2.0, 1.0))
    peak = max(max(abs(v) for v in offsets), 1e-9)
    scale = rng.uniform(*config.offset_scale_range) * config.offset_radius
    return [max(-config.offset_radius, min(config.offset_radius, v / peak * scale)) for v in offsets]


def interpolate_trajectory(traj: Trajectory, index: float, locked_dims: set[int], fallback_index: int) -> list[float]:
    length = len(traj)
    dims = len(traj[0])
    src = max(0.0, min(length - 1.0, index))
    lo = int(math.floor(src))
    hi = min(length - 1, lo + 1)
    alpha = src - lo
    step = []
    for d in range(dims):
        if d in locked_dims:
            step.append(traj[fallback_index][d])
            continue
        lo_v = traj[lo][d]
        hi_v = traj[hi][d]
        step.append((1.0 - alpha) * lo_v + alpha * hi_v)
    return step


def generate_local_delta_samples(
    actions: Trajectory,
    config: LocalDeltaSamplerConfig,
) -> tuple[list[Trajectory], list[Trajectory]]:
    rng = random.Random(config.seed)
    dims = len(actions[0])
    local_scale = compute_local_delta_scale(
        actions,
        config.local_scale_window,
        config.local_scale_floor_ratio,
        GRIPPER_DIMS,
    )

    deltas = []
    samples = []
    for _ in range(config.num_samples):
        sample_amplitude = rng.uniform(*config.sample_amplitude_range)
        noise_scale = rng.uniform(*config.local_noise_scale_range)
        offsets = sample_offset_curve(len(actions), config, rng)

        candidate = []
        for t in range(len(actions)):
            ref_step = interpolate_trajectory(actions, t + offsets[t], GRIPPER_DIMS, t)
            step = []
            for d in range(dims):
                if d in GRIPPER_DIMS:
                    step.append(actions[t][d])
                    continue
                sigma = noise_scale * local_scale[t][d] * sample_amplitude
                step.append(ref_step[d] + rng.gauss(0.0, sigma))
            candidate.append(step)

        candidate = smooth_trajectory(candidate, config.final_smooth_kernel, GRIPPER_DIMS)
        deltas.append([[c - a for c, a in zip(cstep, astep)] for cstep, astep in zip(candidate, actions)])
        samples.append(candidate)
    return deltas, samples


def render_svg(
    actions: Trajectory,
    deltas: list[Trajectory],
    samples: list[Trajectory],
    episode_index: int,
    task: str,
    plot_dims: tuple[int, int],
    output_path: Path,
) -> None:
    actions_cont = delta_to_continuous(actions, GRIPPER_DIMS)
    samples_cont = [delta_to_continuous(sample, GRIPPER_DIMS) for sample in samples]
    dim_x, dim_y = plot_dims
    width, height = 1560, 1060
    palette = ["#2563eb", "#0f766e", "#d97706", "#dc2626", "#7c3aed", "#059669"]

    xs_ref = [step[dim_x] for step in actions_cont]
    ys_ref = [step[dim_y] for step in actions_cont]
    time_axis = list(range(len(actions_cont)))

    all_x = xs_ref[:]
    all_y = ys_ref[:]
    all_dim_x = xs_ref[:]
    all_dim_y = ys_ref[:]
    for traj in samples_cont:
        all_x.extend(step[dim_x] for step in traj)
        all_y.extend(step[dim_y] for step in traj)
        all_dim_x.extend(step[dim_x] for step in traj)
        all_dim_y.extend(step[dim_y] for step in traj)

    xmin, xmax = value_range(all_x)
    ymin, ymax = value_range(all_y)
    dim_x_min, dim_x_max = value_range(all_dim_x)
    dim_y_min, dim_y_max = value_range(all_dim_y)
    xpad = 0.08 * (xmax - xmin + 1e-6)
    ypad = 0.08 * (ymax - ymin + 1e-6)
    dim_x_pad = 0.08 * (dim_x_max - dim_x_min + 1e-6)
    dim_y_pad = 0.08 * (dim_y_max - dim_y_min + 1e-6)

    traj_box = (70, 120, 620, 380)
    dx_box = (70, 560, 620, 300)
    dy_box = (730, 560, 620, 300)
    big_box = (730, 120, 760, 380)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" fill="none">',
        '<rect width="100%" height="100%" fill="#f6f0e8" />',
        '<text x="70" y="58" font-size="28" font-weight="800" fill="#111827">Local delta-window exploration on a real action sequence</text>',
        f'<text x="70" y="88" font-size="15" fill="#4b5563">episode {episode_index} | task: {task or "unknown"} | flow: smooth local re-sampling of delta -> local Gaussian noise -> light smoothing -> integrate for visualization</text>',
        draw_axes(*traj_box, "Integrated Real Episode vs Local Delta Neighbors", ACTION_NAMES[dim_x], ACTION_NAMES[dim_y]),
        draw_axes(*dx_box, f"Integrated {ACTION_NAMES[dim_x]} over Time", "time step", ACTION_NAMES[dim_x]),
        draw_axes(*dy_box, f"Integrated {ACTION_NAMES[dim_y]} over Time", "time step", ACTION_NAMES[dim_y]),
        draw_axes(*big_box, "Projected Integrated Neighborhood", ACTION_NAMES[dim_x], ACTION_NAMES[dim_y]),
    ]

    for i, traj in enumerate(samples_cont):
        color = palette[i % len(palette)]
        traj_xy = map_points(
            [step[dim_x] for step in traj],
            [step[dim_y] for step in traj],
            traj_box,
            (xmin - xpad, xmax + xpad),
            (ymin - ypad, ymax + ypad),
        )
        big_xy = map_points(
            [step[dim_x] for step in traj],
            [step[dim_y] for step in traj],
            big_box,
            (xmin - xpad, xmax + xpad),
            (ymin - ypad, ymax + ypad),
        )
        dim_x_curve = map_points(
            time_axis,
            [step[dim_x] for step in traj],
            dx_box,
            (0, len(actions_cont) - 1),
            (dim_x_min - dim_x_pad, dim_x_max + dim_x_pad),
        )
        dim_y_curve = map_points(
            time_axis,
            [step[dim_y] for step in traj],
            dy_box,
            (0, len(actions_cont) - 1),
            (dim_y_min - dim_y_pad, dim_y_max + dim_y_pad),
        )
        parts.append(svg_polyline(traj_xy, color, 2.3, 0.58))
        parts.append(svg_polyline(big_xy, color, 2.6, 0.62))
        parts.append(svg_polyline(dim_x_curve, color, 2.2, 0.72))
        parts.append(svg_polyline(dim_y_curve, color, 2.2, 0.72))

    ref_traj = map_points(xs_ref, ys_ref, traj_box, (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad))
    ref_big = map_points(xs_ref, ys_ref, big_box, (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad))
    ref_dim_x = map_points(time_axis, xs_ref, dx_box, (0, len(actions_cont) - 1), (dim_x_min - dim_x_pad, dim_x_max + dim_x_pad))
    ref_dim_y = map_points(time_axis, ys_ref, dy_box, (0, len(actions_cont) - 1), (dim_y_min - dim_y_pad, dim_y_max + dim_y_pad))
    parts.append(svg_polyline(ref_traj, "#111827", 4.0, 0.96))
    parts.append(svg_polyline(ref_big, "#111827", 4.0, 0.96))
    parts.append(svg_polyline(ref_dim_x, "#111827", 3.2, 0.96))
    parts.append(svg_polyline(ref_dim_y, "#111827", 3.2, 0.96))

    parts.extend(
        [
            '<circle cx="1120" cy="995" r="8" fill="#111827" />',
            '<text x="1138" y="1000" font-size="15" fill="#111827">real dataset action sequence</text>',
            '<circle cx="1120" cy="1025" r="8" fill="#2563eb" opacity="0.75" />',
            '<text x="1138" y="1030" font-size="15" fill="#111827">local delta-window neighboring trajectories</text>',
            "</svg>",
        ]
    )

    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    output_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    actions, task = load_episode_actions(DATASET_ROOT, DEFAULT_EPISODE)
    plot_dims = pick_plot_dims(actions)
    config = LocalDeltaSamplerConfig()
    deltas, samples = generate_local_delta_samples(actions, config)

    output_svg = output_dir / "local_delta_real_dataset_episode_103.svg"
    render_svg(actions, deltas, samples, DEFAULT_EPISODE, task, plot_dims, output_svg)

    dim_x, dim_y = plot_dims
    avg_proj_delta = []
    for delta in deltas:
        avg_proj_delta.append(
            sum(math.sqrt(step[dim_x] ** 2 + step[dim_y] ** 2) for step in delta) / len(delta)
        )

    print(f"wrote: {output_svg}")
    print(f"episode_index: {DEFAULT_EPISODE}")
    print(f"task: {task}")
    print(f"plot_dims: {plot_dims[0]}={ACTION_NAMES[plot_dims[0]]}, {plot_dims[1]}={ACTION_NAMES[plot_dims[1]]}")
    print(f"num_steps: {len(actions)}")
    print(f"avg_projected_delta_range: {min(avg_proj_delta):.6f} .. {max(avg_proj_delta):.6f}")


if __name__ == "__main__":
    main()
