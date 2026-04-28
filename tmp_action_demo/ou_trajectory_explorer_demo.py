#!/usr/bin/env python3

"""OU-noise trajectory-level exploration demo.

Generate smooth exploration trajectories around a reference 2D action
sequence using Ornstein-Uhlenbeck noise and render an SVG figure.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass


Trajectory = list[list[float]]


@dataclass
class OUSamplerConfig:
    theta_range: tuple[float, float] = (0.08, 0.22)
    sigma_range: tuple[float, float] = (0.02, 0.08)
    global_bias_scale_range: tuple[float, float] = (0.02, 0.08)
    sample_amplitude_range: tuple[float, float] = (0.45, 1.55)
    per_dim_amplitude_range: tuple[float, float] = (0.70, 1.35)
    mean_reversion_target_scale: float = 0.20
    max_delta_ratio: float = 0.28
    delta_smoothness_ratio: float = 0.10
    envelope_edge_power: float = 1.35
    envelope_floor: float = 0.10
    peak_preserve_ratio: float = 0.86
    peak_pullback_attenuation: float = 0.18
    warmup_steps: int = 15


def l2_norm(step: list[float]) -> float:
    return math.sqrt(sum(v * v for v in step))


def add_trajectories(a: Trajectory, b: Trajectory) -> Trajectory:
    return [[va + vb for va, vb in zip(sa, sb)] for sa, sb in zip(a, b)]


def subtract_trajectories(a: Trajectory, b: Trajectory) -> Trajectory:
    return [[va - vb for va, vb in zip(sa, sb)] for sa, sb in zip(a, b)]


def trajectory_bounds(traj: Trajectory) -> tuple[list[float], list[float]]:
    dims = len(traj[0])
    mins = [min(step[d] for step in traj) for d in range(dims)]
    maxs = [max(step[d] for step in traj) for d in range(dims)]
    return mins, maxs


def build_reference_trajectory(length: int = 140) -> Trajectory:
    traj = []
    for t in range(length):
        s = t / (length - 1)
        x = 0.85 * math.sin(2.4 * math.pi * s) + 0.15 * math.sin(8.0 * math.pi * s)
        y = 0.70 * math.cos(1.8 * math.pi * s + 0.25) + 0.22 * math.sin(4.8 * math.pi * s + 0.15)
        traj.append([x, y])
    return traj


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


def enforce_delta_smoothness_limit(delta: Trajectory, max_delta_change: float) -> Trajectory:
    out = [delta[0][:]]
    for i in range(1, len(delta)):
        prev = out[-1]
        curr = delta[i][:]
        diff = [c - p for c, p in zip(curr, prev)]
        diff_norm = l2_norm(diff)
        if diff_norm > max_delta_change and diff_norm > 1e-9:
            ratio = max_delta_change / diff_norm
            curr = [p + d * ratio for p, d in zip(prev, diff)]
        out.append(curr)
    return out


def sample_ou_dimension(
    length: int,
    theta: float,
    sigma: float,
    mu: float,
    rng: random.Random,
    warmup_steps: int,
) -> list[float]:
    x = mu
    values = []
    total_steps = length + warmup_steps
    for i in range(total_steps):
        noise = rng.gauss(0.0, 1.0)
        x = x + theta * (mu - x) + sigma * noise
        if i >= warmup_steps:
            values.append(x)
    return values


def apply_envelope(delta: Trajectory, floor: float, power: float) -> Trajectory:
    if len(delta) <= 1:
        return [step[:] for step in delta]

    out = []
    denom = len(delta) - 1
    for t, step in enumerate(delta):
        s = t / denom
        envelope = floor + (1.0 - floor) * math.sin(math.pi * s) ** power
        out.append([envelope * v for v in step])
    return out


def anchor_delta_endpoints(delta: Trajectory) -> Trajectory:
    if len(delta) <= 1:
        return [step[:] for step in delta]

    dims = len(delta[0])
    start = delta[0]
    end = delta[-1]
    denom = len(delta) - 1
    out = []
    for t, step in enumerate(delta):
        s = t / denom
        bridge = [(1.0 - s) * start[d] + s * end[d] for d in range(dims)]
        out.append([step[d] - bridge[d] for d in range(dims)])
    return out


def preserve_reference_peaks(ref: Trajectory, delta: Trajectory, threshold_ratio: float, attenuation: float) -> Trajectory:
    dims = len(ref[0])
    max_abs_per_dim = [max(abs(step[d]) for step in ref) for d in range(dims)]
    out = []
    for rstep, dstep in zip(ref, delta):
        updated = []
        for d, dv in enumerate(dstep):
            peak_scale = max_abs_per_dim[d]
            if peak_scale <= 1e-9:
                updated.append(dv)
                continue
            near_peak = abs(rstep[d]) >= threshold_ratio * peak_scale
            pulls_back = rstep[d] * dv < 0.0
            if near_peak and pulls_back:
                updated.append(dv * attenuation)
            else:
                updated.append(dv)
        out.append(updated)
    return out


def sample_ou_delta(
    ref: Trajectory,
    action_range: float,
    rng: random.Random,
    config: OUSamplerConfig,
) -> Trajectory:
    length = len(ref)
    dims = len(ref[0])
    mean_span = config.mean_reversion_target_scale * action_range
    global_bias_scale = rng.uniform(*config.global_bias_scale_range) * action_range
    sample_amplitude = rng.uniform(*config.sample_amplitude_range)

    per_dim = []
    for _ in range(dims):
        theta = rng.uniform(*config.theta_range)
        sigma = rng.uniform(*config.sigma_range) * action_range
        dim_amplitude = sample_amplitude * rng.uniform(*config.per_dim_amplitude_range)
        bias = rng.gauss(0.0, global_bias_scale)
        mu = rng.uniform(-mean_span, mean_span) + bias
        values = sample_ou_dimension(
            length=length,
            theta=theta,
            sigma=sigma,
            mu=mu,
            rng=rng,
            warmup_steps=config.warmup_steps,
        )
        per_dim.append([dim_amplitude * v for v in values])

    delta = [[per_dim[d][t] for d in range(dims)] for t in range(length)]
    delta = anchor_delta_endpoints(delta)
    delta = enforce_radius_limit(delta, config.max_delta_ratio * action_range)
    delta = enforce_delta_smoothness_limit(delta, config.delta_smoothness_ratio * action_range)
    delta = apply_envelope(delta, config.envelope_floor, config.envelope_edge_power)
    delta = preserve_reference_peaks(
        ref,
        delta,
        config.peak_preserve_ratio,
        config.peak_pullback_attenuation,
    )
    return delta


def generate_ou_exploration_trajectories(
    ref: Trajectory,
    num_samples: int,
    seed: int = 23,
    config: OUSamplerConfig | None = None,
) -> tuple[list[Trajectory], list[Trajectory]]:
    config = config or OUSamplerConfig()
    rng = random.Random(seed)
    mins, maxs = trajectory_bounds(ref)
    action_range = max(maxs[d] - mins[d] for d in range(len(mins)))

    deltas = []
    samples = []
    for _ in range(num_samples):
        delta = sample_ou_delta(ref, action_range, rng, config)
        candidate = add_trajectories(ref, delta)
        deltas.append(subtract_trajectories(candidate, ref))
        samples.append(candidate)
    return deltas, samples


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


def render_svg(ref: Trajectory, deltas: list[Trajectory], samples: list[Trajectory], output_path: str) -> None:
    width, height = 1520, 1040
    bg = "#f6f0e8"
    card = "#fffdfa"
    palette = ["#2563eb", "#0f766e", "#d97706", "#dc2626", "#7c3aed", "#059669", "#be185d", "#92400e"]

    xs_ref = [step[0] for step in ref]
    ys_ref = [step[1] for step in ref]
    all_x = xs_ref[:]
    all_y = ys_ref[:]
    for traj in samples:
        all_x.extend(step[0] for step in traj)
        all_y.extend(step[1] for step in traj)

    xmin, xmax = value_range(all_x)
    ymin, ymax = value_range(all_y)
    xpad = 0.08 * (xmax - xmin + 1e-6)
    ypad = 0.08 * (ymax - ymin + 1e-6)

    time_axis = list(range(len(ref)))
    delta0_all = []
    delta1_all = []
    for delta in deltas:
        delta0_all.extend(step[0] for step in delta)
        delta1_all.extend(step[1] for step in delta)
    d0min, d0max = value_range(delta0_all)
    d1min, d1max = value_range(delta1_all)
    d0pad = 0.12 * (d0max - d0min + 1e-6)
    d1pad = 0.12 * (d1max - d1min + 1e-6)

    x_box = (70.0, 92.0, 650.0, 420.0)
    d0_box = (800.0, 92.0, 640.0, 200.0)
    d1_box = (800.0, 338.0, 640.0, 200.0)
    xy_box = (260.0, 590.0, 990.0, 360.0)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{bg}" />',
        f'<rect x="35" y="30" width="{width - 70}" height="{height - 60}" rx="28" fill="{card}" stroke="#e5ddd4" stroke-width="1.6"/>',
        '<text x="70" y="54" font-size="28" font-weight="800" fill="#111827">Trajectory-level exploration with Ornstein-Uhlenbeck noise</text>',
        '<text x="70" y="80" font-size="15" fill="#6b7280">OU noise keeps perturbations temporally correlated, so each sampled rollout sustains a direction instead of jittering step by step.</text>',
        draw_axes(*x_box, title="Reference vs OU-Explored Trajectories", xlabel="action dim 1", ylabel="action dim 2"),
        draw_axes(*d0_box, title="OU Delta on Dim 1", xlabel="time step", ylabel="delta a[0]"),
        draw_axes(*d1_box, title="OU Delta on Dim 2", xlabel="time step", ylabel="delta a[1]"),
        draw_axes(*xy_box, title="Action Coordinates Over Time", xlabel="action dim 1", ylabel="action dim 2"),
    ]

    for i, delta in enumerate(deltas):
        color = palette[i % len(palette)]
        d0 = [step[0] for step in delta]
        d1 = [step[1] for step in delta]
        points0 = map_points(time_axis, d0, d0_box, (0, len(ref) - 1), (d0min - d0pad, d0max + d0pad))
        points1 = map_points(time_axis, d1, d1_box, (0, len(ref) - 1), (d1min - d1pad, d1max + d1pad))
        parts.append(svg_polyline(points0, color, 2.0, opacity=0.78))
        parts.append(svg_polyline(points1, color, 2.0, opacity=0.78))

    for i, traj in enumerate(samples):
        color = palette[i % len(palette)]
        xs = [step[0] for step in traj]
        ys = [step[1] for step in traj]
        points = map_points(xs, ys, x_box, (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad))
        parts.append(svg_polyline(points, color, 2.3, opacity=0.55))

    ref_points = map_points(xs_ref, ys_ref, x_box, (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad))
    ref_points_big = map_points(xs_ref, ys_ref, xy_box, (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad))
    parts.append(svg_polyline(ref_points, "#111827", 4.2, opacity=0.96))

    for i, traj in enumerate(samples):
        color = palette[i % len(palette)]
        xs = [step[0] for step in traj]
        ys = [step[1] for step in traj]
        points = map_points(xs, ys, xy_box, (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad))
        parts.append(svg_polyline(points, color, 2.6, opacity=0.62))
    parts.append(svg_polyline(ref_points_big, "#111827", 4.2, opacity=0.96))

    parts.extend(
        [
            '<circle cx="1120" cy="986" r="8" fill="#111827" />',
            '<text x="1138" y="991" font-size="15" fill="#111827">reference action trajectory</text>',
            '<circle cx="1120" cy="1018" r="8" fill="#2563eb" opacity="0.75" />',
            '<text x="1138" y="1023" font-size="15" fill="#111827">OU-sampled smooth rollouts</text>',
            '<text x="70" y="992" font-size="15" fill="#374151">Design: sample temporally correlated delta_t by OU process, vary per-sample and per-dim amplitude, then limit delta magnitude and delta change.</text>',
            '<text x="70" y="1018" font-size="15" fill="#374151">Interpretation: trajectories stay smooth because the perturbation itself changes gradually, while the original peak structure is preserved better.</text>',
            "</svg>",
        ]
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main() -> None:
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_svg = os.path.join(output_dir, "ou_action_trajectory_exploration.svg")

    ref = build_reference_trajectory(length=140)
    deltas, samples = generate_ou_exploration_trajectories(ref, num_samples=8, seed=29)
    render_svg(ref, deltas, samples, output_svg)

    delta_norms = []
    for delta in deltas:
        delta_norms.append(sum(l2_norm(step) for step in delta) / len(delta))

    print(f"wrote: {output_svg}")
    print(f"num_samples: {len(samples)}")
    print(f"avg_delta_norm_range: {min(delta_norms):.4f} .. {max(delta_norms):.4f}")


if __name__ == "__main__":
    main()
