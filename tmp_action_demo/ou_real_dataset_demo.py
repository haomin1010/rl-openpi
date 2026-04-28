#!/usr/bin/env python3

"""Read a real action sequence from the LeRobot corpus and apply OU exploration.

Run this file with the pi05 virtualenv python because it depends on pyarrow:
    /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05/.venv/bin/python ou_real_dataset_demo.py
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
class RealOUSamplerConfig:
    theta_range: tuple[float, float] = (0.045, 0.10)
    sigma_ratio_range: tuple[float, float] = (0.018, 0.045)
    sigma_diff_scale_range: tuple[float, float] = (1.2, 2.4)
    sigma_mix_ratio: float = 0.45
    sample_amplitude_range: tuple[float, float] = (0.45, 1.45)
    per_dim_amplitude_range: tuple[float, float] = (0.70, 1.35)
    mean_reversion_ratio: float = 0.16
    max_delta_ratio: float = 0.26
    delta_smoothness_factor: float = 1.55
    peak_preserve_ratio: float = 0.86
    peak_pullback_attenuation: float = 0.18
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
        d1 = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        stats.append(
            {
                "min": vmin,
                "max": vmax,
                "range": vmax - vmin,
                "mean_abs": sum(abs(v) for v in vals) / len(vals),
                "mean_abs_diff": sum(abs(v) for v in d1) / len(d1) if d1 else 0.0,
            }
        )
    return stats


def pick_plot_dims(actions: Trajectory) -> tuple[int, int]:
    stats = dimension_stats(actions)
    candidates = [d for d in range(len(stats)) if d not in GRIPPER_DIMS]
    ordered = sorted(candidates, key=lambda d: (stats[d]["mean_abs"], stats[d]["range"]), reverse=True)
    return ordered[0], ordered[1]


def sample_ou_dimension(length: int, theta: float, sigma: float, mu: float, rng: random.Random, warmup_steps: int) -> list[float]:
    x = mu
    out = []
    for i in range(length + warmup_steps):
        x = x + theta * (mu - x) + sigma * rng.gauss(0.0, 1.0)
        if i >= warmup_steps:
            out.append(x)
    return out


def enforce_per_step_delta_limit(delta: Trajectory, per_dim_limits: list[float]) -> Trajectory:
    out = []
    for step in delta:
        clipped = []
        for d, v in enumerate(step):
            lim = per_dim_limits[d]
            clipped.append(max(-lim, min(lim, v)))
        out.append(clipped)
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


def preserve_reference_peaks(
    actions: Trajectory,
    delta: Trajectory,
    stats: list[dict[str, float]],
    threshold_ratio: float,
    attenuation: float,
    locked_dims: set[int],
) -> Trajectory:
    out = []
    for step, dstep in zip(actions, delta):
        updated = []
        for d, dv in enumerate(dstep):
            if d in locked_dims:
                updated.append(dv)
                continue
            peak_scale = max(abs(stats[d]["min"]), abs(stats[d]["max"]), 1e-9)
            near_peak = abs(step[d]) >= threshold_ratio * peak_scale
            pulls_back = step[d] * dv < 0.0
            if near_peak and pulls_back:
                updated.append(dv * attenuation)
            else:
                updated.append(dv)
        out.append(updated)
    return out


def generate_ou_samples(actions: Trajectory, config: RealOUSamplerConfig) -> tuple[list[Trajectory], list[Trajectory]]:
    rng = random.Random(config.seed)
    stats = dimension_stats(actions)
    deltas = []
    samples = []
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
    for _sample_idx in range(config.num_samples):
        delta = []
        per_dim_series = []
        per_dim_limits = []
        sample_amplitude = rng.uniform(*config.sample_amplitude_range)
        for d, st in enumerate(stats):
            base_range = max(st["range"], 0.002)
            if d in GRIPPER_DIMS:
                per_dim_series.append([0.0] * len(actions))
                per_dim_limits.append(0.0)
                continue
            theta = rng.uniform(*config.theta_range)
            sigma_from_range = rng.uniform(*config.sigma_ratio_range) * base_range
            sigma_from_diff = rng.uniform(*config.sigma_diff_scale_range) * max(st["mean_abs_diff"], 1e-4)
            sigma = (
                config.sigma_mix_ratio * sigma_from_range
                + (1.0 - config.sigma_mix_ratio) * sigma_from_diff
            )
            sigma *= sample_amplitude * rng.uniform(*config.per_dim_amplitude_range)
            mu_span = config.mean_reversion_ratio * base_range
            mu = rng.uniform(-mu_span, mu_span)
            raw_series = sample_ou_dimension(len(actions), theta, sigma, mu, rng, config.warmup_steps)
            per_dim_series.append(raw_series)
            per_dim_limits.append(config.max_delta_ratio * base_range)

        for t in range(len(actions)):
            delta.append([per_dim_series[d][t] for d in range(len(stats))])

        delta = enforce_per_step_delta_limit(delta, per_dim_limits)
        delta = enforce_delta_smoothness_limit(delta, delta_smoothness_limit)
        delta = preserve_reference_peaks(
            actions,
            delta,
            stats,
            config.peak_preserve_ratio,
            config.peak_pullback_attenuation,
            GRIPPER_DIMS,
        )
        candidate = []
        for t, (step, dstep) in enumerate(zip(actions, delta)):
            scaled_step = []
            for d, a in enumerate(step):
                if d in GRIPPER_DIMS:
                    scaled_step.append(a)
                    continue
                scaled_step.append(a + dstep[d])
            candidate.append(scaled_step)

        # Keep the binary gripper target exactly as in the dataset for this demo.
        for t in range(len(candidate)):
            for d in GRIPPER_DIMS:
                candidate[t][d] = actions[t][d]

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
    for delta, traj in zip(deltas, samples):
        all_x.extend(step[dim_x] for step in traj)
        all_y.extend(step[dim_y] for step in traj)
        all_dim_x.extend(step[dim_x] for step in traj)
        all_dim_y.extend(step[dim_y] for step in traj)

    xmin, xmax = value_range(all_x)
    ymin, ymax = value_range(all_y)
    dim_x_min, dim_x_max = value_range(all_dim_x)
    dim_y_min, dim_y_max = value_range(all_dim_y)

    xpad = 0.10 * (xmax - xmin + 1e-6)
    ypad = 0.10 * (ymax - ymin + 1e-6)
    dim_x_pad = 0.15 * (dim_x_max - dim_x_min + 1e-6)
    dim_y_pad = 0.15 * (dim_y_max - dim_y_min + 1e-6)

    traj_box = (70.0, 120.0, 650.0, 420.0)
    dx_box = (820.0, 120.0, 650.0, 180.0)
    dy_box = (820.0, 360.0, 650.0, 180.0)
    big_box = (210.0, 620.0, 1140.0, 340.0)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f6f0e8" />',
        f'<rect x="35" y="30" width="{width - 70}" height="{height - 60}" rx="28" fill="#fffdfa" stroke="#e5ddd4" stroke-width="1.6"/>',
        '<text x="70" y="58" font-size="28" font-weight="800" fill="#111827">OU exploration on a real action sequence from lerobot_ppo_corpus</text>',
        f'<text x="70" y="84" font-size="15" fill="#6b7280">episode_{episode_index:06d}: {task}</text>',
        f'<text x="70" y="104" font-size="15" fill="#6b7280">Projection dims: {dim_x} = {ACTION_NAMES[dim_x]}, {dim_y} = {ACTION_NAMES[dim_y]}. Gripper targets are kept unchanged in this demo.</text>',
        draw_axes(*traj_box, "Real Episode vs OU-Sampled Neighbors", ACTION_NAMES[dim_x], ACTION_NAMES[dim_y]),
        draw_axes(*dx_box, f"{ACTION_NAMES[dim_x]} over Time", "time step", ACTION_NAMES[dim_x]),
        draw_axes(*dy_box, f"{ACTION_NAMES[dim_y]} over Time", "time step", ACTION_NAMES[dim_y]),
        draw_axes(*big_box, "Projected Action Path", ACTION_NAMES[dim_x], ACTION_NAMES[dim_y]),
    ]

    for i, (delta, traj) in enumerate(zip(deltas, samples)):
        color = palette[i % len(palette)]
        xs = [step[dim_x] for step in traj]
        ys = [step[dim_y] for step in traj]

        traj_points = map_points(xs, ys, traj_box, (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad))
        dx_points = map_points(
            time_axis,
            xs,
            dx_box,
            (0, len(actions) - 1),
            (dim_x_min - dim_x_pad, dim_x_max + dim_x_pad),
        )
        dy_points = map_points(
            time_axis,
            ys,
            dy_box,
            (0, len(actions) - 1),
            (dim_y_min - dim_y_pad, dim_y_max + dim_y_pad),
        )
        big_points = map_points(xs, ys, big_box, (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad))

        parts.append(svg_polyline(traj_points, color, 2.2, 0.58))
        parts.append(svg_polyline(dx_points, color, 1.9, 0.78))
        parts.append(svg_polyline(dy_points, color, 1.9, 0.78))
        parts.append(svg_polyline(big_points, color, 2.5, 0.62))

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
            '<circle cx="1160" cy="998" r="8" fill="#111827" />',
            '<text x="1178" y="1003" font-size="15" fill="#111827">real dataset action sequence</text>',
            '<circle cx="1160" cy="1028" r="8" fill="#2563eb" opacity="0.75" />',
            '<text x="1178" y="1033" font-size="15" fill="#111827">OU-sampled neighboring trajectories</text>',
            "</svg>",
        ]
    )

    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    output_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    actions, task = load_episode_actions(DATASET_ROOT, DEFAULT_EPISODE)
    plot_dims = pick_plot_dims(actions)
    config = RealOUSamplerConfig()
    deltas, samples = generate_ou_samples(actions, config)

    output_svg = output_dir / "ou_real_dataset_episode_103.svg"
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
