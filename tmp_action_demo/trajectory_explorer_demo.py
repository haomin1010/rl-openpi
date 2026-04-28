#!/usr/bin/env python3

"""Trajectory-level smooth exploration demo.

This script generates smooth action trajectories around a reference trajectory
and writes an SVG visualization without third-party dependencies.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass


Trajectory = list[list[float]]


@dataclass
class SamplerConfig:
    num_knots: int = 8
    smoothing_passes: int = 3
    max_delta_ratio: float = 0.22
    velocity_limit_ratio: float = 0.18
    global_scale_range: tuple[float, float] = (0.03, 0.18)
    local_scale_ratio: float = 0.35


def linspace(start: float, end: float, num: int) -> list[float]:
    if num == 1:
        return [start]
    step = (end - start) / (num - 1)
    return [start + i * step for i in range(num)]


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def gaussian_kernel(radius: int, sigma: float) -> list[float]:
    kernel = []
    for i in range(-radius, radius + 1):
        kernel.append(math.exp(-(i * i) / (2.0 * sigma * sigma)))
    total = sum(kernel)
    return [v / total for v in kernel]


def convolve_reflect(values: list[float], kernel: list[float]) -> list[float]:
    radius = len(kernel) // 2
    out = []
    n = len(values)
    for i in range(n):
        acc = 0.0
        for k, weight in enumerate(kernel):
            idx = i + k - radius
            if idx < 0:
                idx = -idx
            elif idx >= n:
                idx = 2 * n - idx - 2
            acc += values[idx] * weight
        out.append(acc)
    return out


def transpose(traj: Trajectory) -> Trajectory:
    dims = len(traj[0])
    return [[step[d] for step in traj] for d in range(dims)]


def untranspose(traj: Trajectory) -> Trajectory:
    steps = len(traj[0])
    dims = len(traj)
    return [[traj[d][t] for d in range(dims)] for t in range(steps)]


def trajectory_bounds(traj: Trajectory) -> tuple[list[float], list[float]]:
    dims = len(traj[0])
    mins = [min(step[d] for step in traj) for d in range(dims)]
    maxs = [max(step[d] for step in traj) for d in range(dims)]
    return mins, maxs


def build_reference_trajectory(length: int = 120) -> Trajectory:
    traj = []
    for t in range(length):
        s = t / (length - 1)
        x = 0.9 * math.sin(2.2 * math.pi * s) + 0.25 * math.sin(5.4 * math.pi * s)
        y = 0.75 * math.cos(1.7 * math.pi * s + 0.35) + 0.2 * math.sin(4.5 * math.pi * s)
        traj.append([x, y])
    return traj


def knot_indices(length: int, num_knots: int) -> list[int]:
    positions = linspace(0, length - 1, num_knots)
    return [round(v) for v in positions]


def interpolate_knots(length: int, knot_ids: list[int], knot_values: list[float]) -> list[float]:
    values = [0.0] * length
    for seg in range(len(knot_ids) - 1):
        left_i = knot_ids[seg]
        right_i = knot_ids[seg + 1]
        left_v = knot_values[seg]
        right_v = knot_values[seg + 1]
        span = max(1, right_i - left_i)
        for t in range(left_i, right_i + 1):
            u = (t - left_i) / span
            w = smoothstep(u)
            values[t] = left_v * (1.0 - w) + right_v * w
    values[0] = knot_values[0]
    values[-1] = knot_values[-1]
    return values


def smooth_sequence(values: list[float], passes: int) -> list[float]:
    kernel = gaussian_kernel(radius=3, sigma=1.4)
    out = values[:]
    for _ in range(passes):
        out = convolve_reflect(out, kernel)
    return out


def scale_trajectory(delta: Trajectory, scale: float) -> Trajectory:
    return [[v * scale for v in step] for step in delta]


def add_trajectories(a: Trajectory, b: Trajectory) -> Trajectory:
    return [[va + vb for va, vb in zip(sa, sb)] for sa, sb in zip(a, b)]


def subtract_trajectories(a: Trajectory, b: Trajectory) -> Trajectory:
    return [[va - vb for va, vb in zip(sa, sb)] for sa, sb in zip(a, b)]


def l2_norm(step: list[float]) -> float:
    return math.sqrt(sum(v * v for v in step))


def enforce_radius_limit(delta: Trajectory, max_radius: float) -> Trajectory:
    limited = []
    for step in delta:
        radius = l2_norm(step)
        if radius > max_radius and radius > 1e-9:
            ratio = max_radius / radius
            limited.append([v * ratio for v in step])
        else:
            limited.append(step[:])
    return limited


def enforce_velocity_limit(traj: Trajectory, max_step_change: float) -> Trajectory:
    out = [traj[0][:]]
    for i in range(1, len(traj)):
        prev = out[-1]
        curr = traj[i][:]
        diff = [c - p for c, p in zip(curr, prev)]
        step_norm = l2_norm(diff)
        if step_norm > max_step_change and step_norm > 1e-9:
            ratio = max_step_change / step_norm
            curr = [p + d * ratio for p, d in zip(prev, diff)]
        out.append(curr)
    return out


def sample_smooth_delta(
    ref: Trajectory,
    action_range: float,
    rng: random.Random,
    config: SamplerConfig,
) -> Trajectory:
    length = len(ref)
    dims = len(ref[0])
    knot_ids = knot_indices(length, config.num_knots)
    global_scale = rng.uniform(*config.global_scale_range) * action_range
    local_scale = global_scale * config.local_scale_ratio

    per_dim = []
    for _ in range(dims):
        global_knots = [rng.gauss(0.0, global_scale) for _ in knot_ids]
        local_knots = [rng.gauss(0.0, local_scale) for _ in knot_ids]
        global_curve = interpolate_knots(length, knot_ids, global_knots)
        local_curve = interpolate_knots(length, knot_ids, local_knots)
        curve = [g + l for g, l in zip(global_curve, local_curve)]
        per_dim.append(smooth_sequence(curve, config.smoothing_passes))

    delta = untranspose(per_dim)
    delta = enforce_radius_limit(delta, config.max_delta_ratio * action_range)
    return delta


def generate_exploration_trajectories(
    ref: Trajectory,
    num_samples: int,
    seed: int = 7,
    config: SamplerConfig | None = None,
) -> list[Trajectory]:
    config = config or SamplerConfig()
    rng = random.Random(seed)
    mins, maxs = trajectory_bounds(ref)
    action_range = max(maxs[d] - mins[d] for d in range(len(mins)))
    max_step_change = max(1e-6, config.velocity_limit_ratio * action_range)

    samples = []
    for _ in range(num_samples):
        delta = sample_smooth_delta(ref, action_range, rng, config)
        candidate = add_trajectories(ref, delta)
        candidate = enforce_velocity_limit(candidate, max_step_change)
        samples.append(candidate)
    return samples


def value_range(values: list[float]) -> tuple[float, float]:
    return min(values), max(values)


def svg_polyline(
    points: list[tuple[float, float]],
    stroke: str,
    stroke_width: float,
    opacity: float = 1.0,
) -> str:
    data = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (
        f'<polyline fill="none" stroke="{stroke}" stroke-width="{stroke_width:.2f}" '
        f'stroke-linecap="round" stroke-linejoin="round" opacity="{opacity:.3f}" '
        f'points="{data}" />'
    )


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


def draw_axes(
    left: float,
    top: float,
    width: float,
    height: float,
    title: str,
    xlabel: str,
    ylabel: str,
) -> str:
    return f"""
    <rect x="{left:.1f}" y="{top:.1f}" width="{width:.1f}" height="{height:.1f}" rx="16" fill="#fffdfa" stroke="#d9d0c7" stroke-width="1.4"/>
    <text x="{left + 16:.1f}" y="{top + 28:.1f}" font-size="18" font-weight="700" fill="#1f2937">{title}</text>
    <text x="{left + width / 2:.1f}" y="{top + height + 34:.1f}" font-size="13" text-anchor="middle" fill="#4b5563">{xlabel}</text>
    <text x="{left - 44:.1f}" y="{top + height / 2:.1f}" font-size="13" text-anchor="middle" transform="rotate(-90 {left - 44:.1f} {top + height / 2:.1f})" fill="#4b5563">{ylabel}</text>
    """


def render_svg(ref: Trajectory, samples: list[Trajectory], output_path: str) -> None:
    width, height = 1400, 920
    bg = "#f6f0e8"
    card = "#fffdfa"

    xs_ref = [p[0] for p in ref]
    ys_ref = [p[1] for p in ref]
    all_x = xs_ref[:]
    all_y = ys_ref[:]
    for traj in samples:
        all_x.extend(p[0] for p in traj)
        all_y.extend(p[1] for p in traj)

    xmin, xmax = value_range(all_x)
    ymin, ymax = value_range(all_y)
    xpad = 0.08 * (xmax - xmin + 1e-6)
    ypad = 0.08 * (ymax - ymin + 1e-6)
    xy_box = (70.0, 80.0, 620.0, 520.0)
    dim1_box = (760.0, 80.0, 560.0, 250.0)
    dim2_box = (760.0, 390.0, 560.0, 250.0)

    time_axis = list(range(len(ref)))
    d1_ref = [p[0] for p in ref]
    d2_ref = [p[1] for p in ref]
    d1_all = d1_ref[:]
    d2_all = d2_ref[:]
    for traj in samples:
        d1_all.extend(p[0] for p in traj)
        d2_all.extend(p[1] for p in traj)
    d1_min, d1_max = value_range(d1_all)
    d2_min, d2_max = value_range(d2_all)
    d1_pad = 0.12 * (d1_max - d1_min + 1e-6)
    d2_pad = 0.12 * (d2_max - d2_min + 1e-6)

    palette = ["#d97706", "#0f766e", "#2563eb", "#dc2626", "#7c3aed", "#059669", "#b45309", "#be185d"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{bg}" />',
        f'<rect x="35" y="30" width="{width - 70}" height="{height - 60}" rx="28" fill="{card}" stroke="#e5ddd4" stroke-width="1.6"/>',
        '<text x="70" y="52" font-size="28" font-weight="800" fill="#111827">Smooth trajectory-level exploration around a reference action sequence</text>',
        '<text x="70" y="76" font-size="15" fill="#6b7280">Method: multi-scale knot perturbation + smooth interpolation + trajectory velocity limit</text>',
        draw_axes(*xy_box, title="2D Action Trajectory", xlabel="action dim 1", ylabel="action dim 2"),
        draw_axes(*dim1_box, title="Dim 1 vs Time", xlabel="time step", ylabel="a[0]"),
        draw_axes(*dim2_box, title="Dim 2 vs Time", xlabel="time step", ylabel="a[1]"),
    ]

    for i, traj in enumerate(samples):
        color = palette[i % len(palette)]
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        xy_points = map_points(xs, ys, xy_box, (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad))
        parts.append(svg_polyline(xy_points, color, 2.2, opacity=0.55))

        d1_points = map_points(time_axis, xs, dim1_box, (0, len(ref) - 1), (d1_min - d1_pad, d1_max + d1_pad))
        d2_points = map_points(time_axis, ys, dim2_box, (0, len(ref) - 1), (d2_min - d2_pad, d2_max + d2_pad))
        parts.append(svg_polyline(d1_points, color, 1.9, opacity=0.55))
        parts.append(svg_polyline(d2_points, color, 1.9, opacity=0.55))

    ref_xy_points = map_points(xs_ref, ys_ref, xy_box, (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad))
    ref_d1_points = map_points(time_axis, d1_ref, dim1_box, (0, len(ref) - 1), (d1_min - d1_pad, d1_max + d1_pad))
    ref_d2_points = map_points(time_axis, d2_ref, dim2_box, (0, len(ref) - 1), (d2_min - d2_pad, d2_max + d2_pad))
    parts.append(svg_polyline(ref_xy_points, "#111827", 4.0, opacity=0.95))
    parts.append(svg_polyline(ref_d1_points, "#111827", 3.2, opacity=0.95))
    parts.append(svg_polyline(ref_d2_points, "#111827", 3.2, opacity=0.95))

    parts.extend(
        [
            '<circle cx="1110" cy="705" r="8" fill="#111827" />',
            '<text x="1128" y="710" font-size="15" fill="#111827">reference trajectory</text>',
            '<circle cx="1110" cy="740" r="8" fill="#2563eb" opacity="0.70" />',
            '<text x="1128" y="745" font-size="15" fill="#111827">sampled smooth exploration trajectories</text>',
            '<text x="70" y="850" font-size="15" fill="#374151">Why this looks smooth: perturbation is sampled in a low-dimensional knot space, then interpolated and filtered over time.</text>',
            '<text x="70" y="878" font-size="15" fill="#374151">Why this explores enough: each sample draws a different global/local scale, so some trajectories stay close while others probe farther out.</text>',
            "</svg>",
        ]
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main() -> None:
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_svg = os.path.join(output_dir, "action_trajectory_exploration.svg")

    ref = build_reference_trajectory(length=120)
    samples = generate_exploration_trajectories(ref, num_samples=8, seed=11)
    render_svg(ref, samples, output_svg)

    delta_norms = []
    for traj in samples:
        diff = subtract_trajectories(traj, ref)
        delta_norms.append(sum(l2_norm(step) for step in diff) / len(diff))

    print(f"wrote: {output_svg}")
    print(f"num_samples: {len(samples)}")
    print(f"avg_delta_norm_range: {min(delta_norms):.4f} .. {max(delta_norms):.4f}")


if __name__ == "__main__":
    main()
