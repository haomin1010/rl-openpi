#!/usr/bin/env python3

"""Simplified OU exploration on a real action sequence from the LeRobot corpus.

Run this file with the pi05 virtualenv python because it depends on pyarrow:
    /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05/.venv/bin/python ou_real_dataset_demo_simplified.py

Simplified exploration flow:
1. Load one real action trajectory from the dataset.
2. For each sampled neighboring trajectory, draw one global amplitude.
3. Sample one smooth shared random time-offset curve and use it to slightly warp timing.
4. For each non-gripper dimension, sample a zero-mean OU noise series.
5. Scale that OU series by the dimension's typical step size.
6. Clip each dimension's delta magnitude to stay inside a local neighborhood.
7. Limit how fast the delta itself can change across time, to keep perturbations smooth.
8. Add the smooth delta back to the time-warped action sequence.
9. Keep binary gripper dimensions exactly unchanged.

Compared with the fuller demo, this script intentionally removes:
- endpoint anchoring
- envelope shaping
- peak-preservation heuristics
- range/diff mixed sigma schedules
- per-dimension amplitude randomization

The goal here is to keep only the core idea:
"sample smooth OU perturbations near the action trajectory, with a small smooth timing warp."
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
class SimplifiedOUSamplerConfig:
    theta_range: tuple[float, float] = (0.045, 0.10)
    sigma_step_scale_range: tuple[float, float] = (1.3, 2.4)
    sample_amplitude_range: tuple[float, float] = (0.55, 1.45)
    time_shift_max_steps_range: tuple[float, float] = (1.0, 5.0)
    time_shift_sigma_ratio: float = 0.32
    time_shift_smoothness_limit: float = 0.45
    local_activity_window: int = 9
    local_activity_floor_ratio: float = 0.18
    max_delta_ratio: float = 0.26
    delta_smoothness_factor: float = 1.55
    delta_lowpass_kernel: tuple[float, ...] = (0.2, 0.6, 0.2)
    remove_delta_mean: bool = True
    drift_limit_ratio: float = 0.55
    warmup_steps: int = 20
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


def l2_norm(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


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
        diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        stats.append(
            {
                "min": vmin,
                "max": vmax,
                "range": vmax - vmin,
                "mean_abs": sum(abs(v) for v in vals) / len(vals),
                "mean_abs_diff": sum(abs(v) for v in diffs) / len(diffs) if diffs else 0.0,
            }
        )
    return stats


def compute_local_activity(actions: Trajectory, window: int, floor_ratio: float) -> list[list[float]]:
    """Estimate per-timestep, per-dimension local motion scale from a sliding window."""
    length = len(actions)
    dims = len(actions[0])
    radius = max(1, window // 2)
    stats = dimension_stats(actions)
    floor_scales = [floor_ratio * max(st["mean_abs_diff"], 1e-4) for st in stats]

    diffs = [[0.0] * dims for _ in range(length)]
    for t in range(1, length):
        for d in range(dims):
            diffs[t][d] = abs(actions[t][d] - actions[t - 1][d])

    local_activity = []
    for t in range(length):
        lo = max(0, t - radius)
        hi = min(length - 1, t + radius)
        span = hi - lo + 1
        step = []
        for d in range(dims):
            window_mean = sum(diffs[i][d] for i in range(lo, hi + 1)) / span
            step.append(max(window_mean, floor_scales[d]))
        local_activity.append(step)
    return local_activity


def pick_plot_dims(actions: Trajectory) -> tuple[int, int]:
    stats = dimension_stats(actions)
    candidates = [d for d in range(len(stats)) if d not in GRIPPER_DIMS]
    ordered = sorted(candidates, key=lambda d: (stats[d]["mean_abs"], stats[d]["range"]), reverse=True)
    return ordered[0], ordered[1]


def sample_ou_dimension(length: int, theta: float, sigma: float, rng: random.Random, warmup_steps: int) -> list[float]:
    x = 0.0
    out = []
    for i in range(length + warmup_steps):
        x = x + theta * (0.0 - x) + sigma * rng.gauss(0.0, 1.0)
        if i >= warmup_steps:
            out.append(x)
    return out


def enforce_per_step_delta_limit(delta: Trajectory, per_dim_limits: list[float]) -> Trajectory:
    out = []
    for step in delta:
        out.append([max(-per_dim_limits[d], min(per_dim_limits[d], v)) for d, v in enumerate(step)])
    return out


def enforce_delta_smoothness_limit(delta: Trajectory, limit: float) -> Trajectory:
    out = [delta[0][:]]
    for i in range(1, len(delta)):
        prev = out[-1]
        curr = delta[i][:]
        diff = [c - p for c, p in zip(curr, prev)]
        norm = l2_norm(diff)
        if norm > limit and norm > 1e-9:
            ratio = limit / norm
            curr = [p + d * ratio for p, d in zip(prev, diff)]
        out.append(curr)
    return out


def smooth_delta(delta: Trajectory, kernel: tuple[float, ...]) -> Trajectory:
    """Apply a small temporal low-pass filter to reduce high-frequency jitter."""
    if not delta or len(kernel) % 2 == 0:
        return [step[:] for step in delta]

    radius = len(kernel) // 2
    weight_sum = sum(kernel)
    if weight_sum <= 1e-9:
        return [step[:] for step in delta]

    length = len(delta)
    dims = len(delta[0])
    out = []
    for t in range(length):
        step = []
        for d in range(dims):
            accum = 0.0
            for k, weight in enumerate(kernel):
                idx = min(length - 1, max(0, t + k - radius))
                accum += weight * delta[idx][d]
            step.append(accum / weight_sum)
        out.append(step)
    return out


def remove_delta_dimension_mean(delta: Trajectory, locked_dims: set[int]) -> Trajectory:
    if not delta:
        return []

    length = len(delta)
    dims = len(delta[0])
    means = []
    for d in range(dims):
        if d in locked_dims:
            means.append(0.0)
        else:
            means.append(sum(delta[t][d] for t in range(length)) / length)

    out = []
    for step in delta:
        out.append([
            value if d in locked_dims else value - means[d]
            for d, value in enumerate(step)
        ])
    return out


def limit_cumulative_drift(delta: Trajectory, per_dim_drift_limits: list[float], locked_dims: set[int]) -> Trajectory:
    """Clamp prefix-sum drift so small same-sign deltas do not accumulate too far."""
    if not delta:
        return []

    dims = len(delta[0])
    running = [0.0] * dims
    out = []
    for step in delta:
        updated = []
        for d, value in enumerate(step):
            if d in locked_dims:
                updated.append(value)
                continue
            proposed = running[d] + value
            limit = per_dim_drift_limits[d]
            if abs(proposed) > limit and abs(proposed) > 1e-9:
                allowed = math.copysign(limit, proposed) - running[d]
                updated.append(allowed)
                running[d] += allowed
            else:
                updated.append(value)
                running[d] = proposed
        out.append(updated)
    return out


def enforce_scalar_smoothness_limit(values: list[float], limit: float) -> list[float]:
    out = [values[0]]
    for i in range(1, len(values)):
        prev = out[-1]
        curr = values[i]
        diff = curr - prev
        if abs(diff) > limit:
            curr = prev + math.copysign(limit, diff)
        out.append(curr)
    return out


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


def sample_time_offsets(length: int, config: SimplifiedOUSamplerConfig, rng: random.Random, sample_amplitude: float) -> list[float]:
    max_shift = rng.uniform(*config.time_shift_max_steps_range) * sample_amplitude
    raw = [rng.gauss(0.0, 1.0) for _ in range(length)]
    kernel = (1.0, 2.0, 3.0, 2.0, 1.0)
    offsets = smooth_scalar_series(raw, kernel)
    peak = max(max(abs(v) for v in offsets), 1e-9)
    offsets = [v / peak * max_shift for v in offsets]
    offsets = [max(-max_shift, min(max_shift, v)) for v in offsets]
    offsets = enforce_scalar_smoothness_limit(offsets, config.time_shift_smoothness_limit)
    return offsets


def time_warp_actions(actions: Trajectory, offsets: list[float]) -> Trajectory:
    length = len(actions)
    dims = len(actions[0])
    warped = []
    for t, offset in enumerate(offsets):
        src = max(0.0, min(length - 1.0, t + offset))
        lo = int(math.floor(src))
        hi = min(length - 1, lo + 1)
        alpha = src - lo
        step = []
        for d in range(dims):
            if d in GRIPPER_DIMS:
                step.append(actions[t][d])
                continue
            lo_v = actions[lo][d]
            hi_v = actions[hi][d]
            step.append((1.0 - alpha) * lo_v + alpha * hi_v)
        warped.append(step)
    return warped


def generate_ou_samples(actions: Trajectory, config: SimplifiedOUSamplerConfig) -> tuple[list[Trajectory], list[Trajectory]]:
    """Sample smooth neighboring trajectories by adding simplified OU deltas."""
    rng = random.Random(config.seed)
    stats = dimension_stats(actions)
    local_activity = compute_local_activity(
        actions,
        config.local_activity_window,
        config.local_activity_floor_ratio,
    )
    dims = len(stats)

    base_step_norms = []
    for i in range(1, len(actions)):
        diff = [
            actions[i][d] - actions[i - 1][d]
            for d in range(dims)
            if d not in GRIPPER_DIMS
        ]
        base_step_norms.append(l2_norm(diff))
    mean_base_step_norm = sum(base_step_norms) / len(base_step_norms)
    delta_smoothness_limit = config.delta_smoothness_factor * mean_base_step_norm

    deltas = []
    samples = []
    for _ in range(config.num_samples):
        sample_amplitude = rng.uniform(*config.sample_amplitude_range)
        warped_actions = time_warp_actions(actions, sample_time_offsets(len(actions), config, rng, sample_amplitude))
        per_dim_series = []
        per_dim_limits = []

        for d, st in enumerate(stats):
            base_range = max(st["range"], 0.002)
            if d in GRIPPER_DIMS:
                per_dim_series.append([0.0] * len(actions))
                per_dim_limits.append(0.0)
                continue

            theta = rng.uniform(*config.theta_range)
            sigma = rng.uniform(*config.sigma_step_scale_range)
            raw_series = sample_ou_dimension(len(actions), theta, sigma, rng, config.warmup_steps)
            scaled_series = [
                raw_series[t] * local_activity[t][d] * sample_amplitude
                for t in range(len(actions))
            ]
            per_dim_series.append(scaled_series)
            per_dim_limits.append(config.max_delta_ratio * base_range)

        delta = []
        for t in range(len(actions)):
            delta.append([per_dim_series[d][t] for d in range(dims)])

        delta = enforce_per_step_delta_limit(delta, per_dim_limits)
        delta = enforce_delta_smoothness_limit(delta, delta_smoothness_limit)
        delta = smooth_delta(delta, config.delta_lowpass_kernel)
        if config.remove_delta_mean:
            delta = remove_delta_dimension_mean(delta, GRIPPER_DIMS)
        per_dim_drift_limits = [
            0.0 if d in GRIPPER_DIMS else config.drift_limit_ratio * max(stats[d]["range"], 0.002)
            for d in range(dims)
        ]
        delta = limit_cumulative_drift(delta, per_dim_drift_limits, GRIPPER_DIMS)

        candidate = []
        for warped_step, dstep in zip(warped_actions, delta):
            candidate.append([
                action_value if d in GRIPPER_DIMS else action_value + dstep[d]
                for d, action_value in enumerate(warped_step)
            ])

        final_delta = [[c - a for c, a in zip(cstep, astep)] for cstep, astep in zip(candidate, actions)]
        deltas.append(final_delta)
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
    dim_x, dim_y = plot_dims
    width, height = 1560, 1060
    palette = ["#2563eb", "#0f766e", "#d97706", "#dc2626", "#7c3aed", "#059669"]

    xs_ref = [step[dim_x] for step in actions]
    ys_ref = [step[dim_y] for step in actions]
    time_axis = list(range(len(actions)))

    all_x = xs_ref[:]
    all_y = ys_ref[:]
    all_dim_x = xs_ref[:]
    all_dim_y = ys_ref[:]
    for traj in samples:
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
        '<text x="70" y="58" font-size="28" font-weight="800" fill="#111827">Simplified OU exploration on a real action sequence from lerobot_ppo_corpus</text>',
        f'<text x="70" y="88" font-size="15" fill="#4b5563">episode {episode_index} | task: {task or "unknown"} | flow: smooth time warp -> zero-mean OU delta -> per-dim clip -> delta smoothness limit</text>',
        draw_axes(*traj_box, "Real Episode vs Simplified OU Neighbors", ACTION_NAMES[dim_x], ACTION_NAMES[dim_y]),
        draw_axes(*dx_box, f"Action {ACTION_NAMES[dim_x]} over Time", "time step", ACTION_NAMES[dim_x]),
        draw_axes(*dy_box, f"Action {ACTION_NAMES[dim_y]} over Time", "time step", ACTION_NAMES[dim_y]),
        draw_axes(*big_box, "Projected Action Neighborhood", ACTION_NAMES[dim_x], ACTION_NAMES[dim_y]),
    ]

    for i, traj in enumerate(samples):
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
            (0, len(actions) - 1),
            (dim_x_min - dim_x_pad, dim_x_max + dim_x_pad),
        )
        dim_y_curve = map_points(
            time_axis,
            [step[dim_y] for step in traj],
            dy_box,
            (0, len(actions) - 1),
            (dim_y_min - dim_y_pad, dim_y_max + dim_y_pad),
        )
        parts.append(svg_polyline(traj_xy, color, 2.3, 0.58))
        parts.append(svg_polyline(big_xy, color, 2.6, 0.62))
        parts.append(svg_polyline(dim_x_curve, color, 2.2, 0.72))
        parts.append(svg_polyline(dim_y_curve, color, 2.2, 0.72))

    ref_traj = map_points(xs_ref, ys_ref, traj_box, (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad))
    ref_big = map_points(xs_ref, ys_ref, big_box, (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad))
    ref_dim_x = map_points(
        time_axis,
        xs_ref,
        dx_box,
        (0, len(actions) - 1),
        (dim_x_min - dim_x_pad, dim_x_max + dim_x_pad),
    )
    ref_dim_y = map_points(
        time_axis,
        ys_ref,
        dy_box,
        (0, len(actions) - 1),
        (dim_y_min - dim_y_pad, dim_y_max + dim_y_pad),
    )
    parts.append(svg_polyline(ref_traj, "#111827", 4.0, 0.96))
    parts.append(svg_polyline(ref_big, "#111827", 4.0, 0.96))
    parts.append(svg_polyline(ref_dim_x, "#111827", 3.2, 0.96))
    parts.append(svg_polyline(ref_dim_y, "#111827", 3.2, 0.96))

    parts.extend(
        [
            '<circle cx="1120" cy="995" r="8" fill="#111827" />',
            '<text x="1138" y="1000" font-size="15" fill="#111827">real dataset action sequence</text>',
            '<circle cx="1120" cy="1025" r="8" fill="#2563eb" opacity="0.75" />',
            '<text x="1138" y="1030" font-size="15" fill="#111827">simplified OU-sampled neighboring trajectories</text>',
            "</svg>",
        ]
    )

    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    output_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    actions, task = load_episode_actions(DATASET_ROOT, DEFAULT_EPISODE)
    plot_dims = pick_plot_dims(actions)
    config = SimplifiedOUSamplerConfig()
    deltas, samples = generate_ou_samples(actions, config)

    output_svg = output_dir / "ou_real_dataset_episode_103_simplified.svg"
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
