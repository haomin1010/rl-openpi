#!/usr/bin/env python3

"""Structured neighborhood exploration directly in delta-action sequence space.

Run this file with the pi05 virtualenv python because it depends on pyarrow:
    /mnt/data/lhm/vla-rl/RoboTwin/policy/pi05/.venv/bin/python structured_delta_real_dataset_demo.py

Design goals:
1. Define the neighborhood directly in delta-action space.
2. Sample temporally smooth perturbations instead of iid step noise.
3. Penalize cumulative drift because delta actions integrate over time.
4. Keep gripper dimensions exactly unchanged.
5. Integrate the delta trajectories only for visualization.

The perturbation for each action dimension is sampled as:
    a_t^sample = a_t^ref + delta_t

where delta is drawn from a Gaussian with precision
    Q = lambda0 I + lambda1 D1^T D1 + lambda2 D2^T D2
        + lambda_c C^T C + lambda_m 11^T

This means:
- lambda0 controls raw action-space proximity.
- lambda1 and lambda2 enforce smoothness in time.
- lambda_c penalizes accumulated drift of delta perturbations.
- lambda_m discourages a non-zero mean bias over the whole sequence.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


Trajectory = list[list[float]]


@dataclass
class StructuredDeltaSamplerConfig:
    local_scale_window: int = 9
    local_scale_floor_ratio: float = 0.15
    num_samples: int = 6
    seed: int = 23

    sample_amplitude_range: tuple[float, float] = (0.55, 1.15)
    step_budget_ratio: float = 2.75
    cumulative_budget_ratio: float = 8.0

    lambda0: float = 1.0
    lambda1: float = 18.0
    lambda2: float = 42.0
    lambda_c: float = 0.08
    lambda_m: float = 0.12

    pos_dim_weight: float = 1.0
    rot_dim_weight: float = 1.8
    gripper_dim_weight: float = 10.0


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
ROTATION_DIMS = {3, 4, 5, 10, 11, 12}
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


def delta_to_continuous(actions: Trajectory, locked_dims: set[int]) -> Trajectory:
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


def compute_local_delta_scale(actions: np.ndarray, window: int, floor_ratio: float, locked_dims: set[int]) -> np.ndarray:
    length, dims = actions.shape
    radius = max(1, window // 2)
    floors = np.zeros(dims, dtype=np.float64)
    for d in range(dims):
        if d in locked_dims:
            continue
        mean_abs_delta = float(np.mean(np.abs(actions[:, d])))
        floors[d] = floor_ratio * max(mean_abs_delta, 1e-4)

    local_scale = np.zeros_like(actions, dtype=np.float64)
    for t in range(length):
        lo = max(0, t - radius)
        hi = min(length, t + radius + 1)
        mean_abs = np.mean(np.abs(actions[lo:hi]), axis=0)
        local_scale[t] = np.maximum(mean_abs, floors)
        local_scale[t, list(locked_dims)] = 0.0
    return local_scale


def build_first_difference_matrix(length: int) -> np.ndarray:
    mat = np.zeros((max(0, length - 1), length), dtype=np.float64)
    for i in range(length - 1):
        mat[i, i] = -1.0
        mat[i, i + 1] = 1.0
    return mat


def build_second_difference_matrix(length: int) -> np.ndarray:
    mat = np.zeros((max(0, length - 2), length), dtype=np.float64)
    for i in range(length - 2):
        mat[i, i] = 1.0
        mat[i, i + 1] = -2.0
        mat[i, i + 2] = 1.0
    return mat


def build_cumulative_matrix(length: int) -> np.ndarray:
    return np.tril(np.ones((length, length), dtype=np.float64))


def dimension_weight(config: StructuredDeltaSamplerConfig, dim: int) -> float:
    if dim in GRIPPER_DIMS:
        return config.gripper_dim_weight
    if dim in ROTATION_DIMS:
        return config.rot_dim_weight
    return config.pos_dim_weight


def build_precision_matrix(length: int, config: StructuredDeltaSamplerConfig, dim_weight: float) -> np.ndarray:
    identity = np.eye(length, dtype=np.float64)
    d1 = build_first_difference_matrix(length)
    d2 = build_second_difference_matrix(length)
    cumulative = build_cumulative_matrix(length)
    ones = np.ones((length, 1), dtype=np.float64)

    precision = config.lambda0 * dim_weight * identity
    if len(d1) > 0:
        precision += config.lambda1 * dim_weight * (d1.T @ d1)
    if len(d2) > 0:
        precision += config.lambda2 * dim_weight * (d2.T @ d2)
    precision += config.lambda_c * dim_weight * (cumulative.T @ cumulative)
    precision += config.lambda_m * dim_weight * (ones @ ones.T)
    precision += 1e-6 * identity
    return precision


def sample_structured_perturbation(
    rng: np.random.Generator,
    local_scale: np.ndarray,
    config: StructuredDeltaSamplerConfig,
) -> np.ndarray:
    length, dims = local_scale.shape
    perturb = np.zeros((length, dims), dtype=np.float64)
    sample_amplitude = rng.uniform(*config.sample_amplitude_range)

    for d in range(dims):
        if d in GRIPPER_DIMS:
            continue

        weight = dimension_weight(config, d)
        precision = build_precision_matrix(length, config, weight)
        covariance = np.linalg.inv(precision)
        raw = rng.multivariate_normal(np.zeros(length, dtype=np.float64), covariance, check_valid="ignore")

        raw -= np.mean(raw)
        raw_std = float(np.std(raw))
        if raw_std < 1e-9:
            continue
        raw /= raw_std

        delta = sample_amplitude * local_scale[:, d] * raw

        step_budget = config.step_budget_ratio * np.maximum(local_scale[:, d], 1e-6)
        step_ratio = float(np.max(np.abs(delta) / step_budget))
        if step_ratio > 1.0:
            delta /= step_ratio

        cumulative = np.cumsum(delta)
        cumulative_budget = config.cumulative_budget_ratio * max(float(np.mean(local_scale[:, d])), 1e-6)
        cumulative_ratio = float(np.max(np.abs(cumulative)) / cumulative_budget)
        if cumulative_ratio > 1.0:
            delta /= cumulative_ratio

        perturb[:, d] = delta

    return perturb


def generate_structured_delta_samples(
    actions: Trajectory,
    config: StructuredDeltaSamplerConfig,
) -> tuple[list[Trajectory], list[Trajectory]]:
    rng = np.random.default_rng(config.seed)
    action_array = np.asarray(actions, dtype=np.float64)
    local_scale = compute_local_delta_scale(
        action_array,
        config.local_scale_window,
        config.local_scale_floor_ratio,
        GRIPPER_DIMS,
    )

    deltas = []
    samples = []
    for _ in range(config.num_samples):
        perturb = sample_structured_perturbation(rng, local_scale, config)
        candidate = action_array.copy()
        candidate[:, :] += perturb
        candidate[:, list(GRIPPER_DIMS)] = action_array[:, list(GRIPPER_DIMS)]

        deltas.append(perturb.tolist())
        samples.append(candidate.tolist())
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
        '<text x="70" y="58" font-size="28" font-weight="800" fill="#111827">Structured delta-action neighborhood exploration</text>',
        f'<text x="70" y="88" font-size="15" fill="#4b5563">episode {episode_index} | task: {task or "unknown"} | flow: smooth delta-space Gaussian perturbation -> cumulative-drift control -> integrate for visualization</text>',
        draw_axes(*traj_box, "Integrated Real Episode vs Structured Delta Neighbors", ACTION_NAMES[dim_x], ACTION_NAMES[dim_y]),
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
            '<text x="1138" y="1030" font-size="15" fill="#111827">structured delta-space neighboring trajectories</text>',
            "</svg>",
        ]
    )

    output_path.write_text("\n".join(parts), encoding="utf-8")


def summarize_deltas(deltas: list[Trajectory], plot_dims: tuple[int, int]) -> tuple[float, float, float]:
    dim_x, dim_y = plot_dims
    avg_proj_delta = []
    avg_proj_cumulative = []
    max_abs_step = 0.0
    for delta in deltas:
        seq = np.asarray(delta, dtype=np.float64)
        proj_step = np.sqrt(seq[:, dim_x] ** 2 + seq[:, dim_y] ** 2)
        proj_cumulative = np.sqrt(
            np.cumsum(seq[:, dim_x]) ** 2 + np.cumsum(seq[:, dim_y]) ** 2
        )
        avg_proj_delta.append(float(np.mean(proj_step)))
        avg_proj_cumulative.append(float(np.mean(proj_cumulative)))
        max_abs_step = max(max_abs_step, float(np.max(np.abs(seq))))
    return min(avg_proj_delta), max(avg_proj_delta), max(avg_proj_cumulative + [0.0, max_abs_step])


def main() -> None:
    output_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    actions, task = load_episode_actions(DATASET_ROOT, DEFAULT_EPISODE)
    plot_dims = pick_plot_dims(actions)
    config = StructuredDeltaSamplerConfig()
    deltas, samples = generate_structured_delta_samples(actions, config)

    output_svg = output_dir / "structured_delta_real_dataset_episode_103.svg"
    render_svg(actions, deltas, samples, DEFAULT_EPISODE, task, plot_dims, output_svg)

    proj_delta_min, proj_delta_max, proj_cumulative_max = summarize_deltas(deltas, plot_dims)
    print(f"wrote: {output_svg}")
    print(f"episode_index: {DEFAULT_EPISODE}")
    print(f"task: {task}")
    print(f"plot_dims: {plot_dims[0]}={ACTION_NAMES[plot_dims[0]]}, {plot_dims[1]}={ACTION_NAMES[plot_dims[1]]}")
    print(f"num_steps: {len(actions)}")
    print(f"avg_projected_delta_range: {proj_delta_min:.6f} .. {proj_delta_max:.6f}")
    print(f"max_projected_cumulative_or_step: {proj_cumulative_max:.6f}")


if __name__ == "__main__":
    main()
