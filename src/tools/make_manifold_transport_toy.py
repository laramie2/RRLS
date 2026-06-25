from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper/figures"


def source_map(z: np.ndarray) -> np.ndarray:
    """Nearly spherical source distribution with a mild local warp."""
    u, v = z[:, 0], z[:, 1]
    x = -1.45 + 0.46 * u + 0.04 * v + 0.035 * (u**2 - v**2)
    y = 0.02 + 0.03 * u + 0.46 * v + 0.035 * u * v
    return np.stack([x, y], axis=1)


def target_first_order(z: np.ndarray) -> np.ndarray:
    """Affine part of the target transport: the first-order local model."""
    u, v = z[:, 0], z[:, 1]
    x = 1.36 + 0.44 * u - 0.02 * v
    y = 0.02 + 0.03 * u + 0.44 * v
    return np.stack([x, y], axis=1)


def target_second_order(z: np.ndarray) -> np.ndarray:
    """Target distribution: a visibly elliptical cloud with mild curved deformation."""
    u, v = z[:, 0], z[:, 1]
    x = 1.45 + 0.73 * u - 0.13 * v + 0.34 * (u**2 - 0.48 * v**2)
    y = 0.02 + 0.08 * u + 0.28 * v + 0.28 * u * v - 0.08 * (u**2 - v**2)
    return np.stack([x, y], axis=1)


def bezier_path(start: np.ndarray, end: np.ndarray, bend: float, steps: int = 40) -> np.ndarray:
    """Smooth multi-step reference path used only for visualization."""
    t = np.linspace(0.0, 1.0, steps)[:, None]
    control_a = start + np.array([0.85, bend])
    control_b = end - np.array([0.85, -bend])
    return (
        (1.0 - t) ** 3 * start
        + 3.0 * (1.0 - t) ** 2 * t * control_a
        + 3.0 * (1.0 - t) * t**2 * control_b
        + t**3 * end
    )


def paired_endpoint_error(points: np.ndarray, target_points: np.ndarray) -> float:
    return float(np.linalg.norm(points - target_points, axis=1).mean())


def sample_disk(rng: np.random.Generator, count: int, radius: float = 1.0) -> np.ndarray:
    angles = rng.uniform(0.0, 2.0 * np.pi, count)
    radii = radius * np.sqrt(rng.uniform(0.0, 1.0, count))
    return np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)


def distribution_clouds(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    z_src = sample_disk(rng, 1200, radius=1.0)
    z_tgt = sample_disk(rng, 1200, radius=1.0)
    src_cloud = source_map(z_src) + rng.normal(scale=0.050, size=(len(z_src), 2))
    tgt_cloud = target_second_order(z_tgt) + rng.normal(scale=0.050, size=(len(z_tgt), 2))
    return src_cloud, tgt_cloud


def draw_distribution_background(ax: plt.Axes, src_cloud: np.ndarray, tgt_cloud: np.ndarray) -> None:
    ax.scatter(src_cloud[:, 0], src_cloud[:, 1], s=12, c="#e53e3e", alpha=0.13, linewidths=0)
    ax.scatter(tgt_cloud[:, 0], tgt_cloud[:, 1], s=12, c="#4299e1", alpha=0.18, linewidths=0)


def setup_axis(ax: plt.Axes) -> None:
    ax.set_xlim(-2.25, 2.75)
    ax.set_ylim(-1.25, 1.25)
    ax.set_xticks(np.linspace(-2.0, 2.0, 5))
    ax.set_yticks(np.linspace(-1.0, 1.0, 5))
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    ax.grid(True, color="#d9d9d9", alpha=0.45, linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def draw_particles(
    ax: plt.Axes,
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    curved: bool,
    bends: np.ndarray,
) -> None:
    for start, end, bend in zip(starts, ends, bends):
        if curved:
            path = bezier_path(start, end, bend=bend)
            ax.plot(path[:, 0], path[:, 1], color="#9f3faf", alpha=0.62, linewidth=1.15)
        else:
            ax.plot([start[0], end[0]], [start[1], end[1]], color="#9f3faf", alpha=0.62, linewidth=1.15)
    ax.scatter(starts[:, 0], starts[:, 1], s=28, c="#e53e3e", edgecolors="black", linewidths=0.65, zorder=5)
    ax.scatter(ends[:, 0], ends[:, 1], s=28, c="#3182ce", edgecolors="black", linewidths=0.65, zorder=5)


def draw_error_segments(ax: plt.Axes, first_order: np.ndarray, second_order: np.ndarray) -> None:
    for bad, good in zip(first_order[::3], second_order[::3]):
        ax.plot(
            [bad[0], good[0]],
            [bad[1], good[1]],
            color="#dd6b20",
            linestyle="--",
            linewidth=1.0,
            alpha=0.65,
        )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    src_cloud, tgt_cloud = distribution_clouds(rng)

    particle_angles = np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False)
    particle_radii = 0.20 + 0.78 * ((np.arange(len(particle_angles)) % 4) + 1) / 4.0
    z_particles = np.stack(
        [particle_radii * np.cos(particle_angles), particle_radii * np.sin(particle_angles)],
        axis=1,
    )
    starts = source_map(z_particles) + rng.normal(scale=0.025, size=(len(z_particles), 2))
    intended_ends = target_second_order(z_particles)
    naive_ends = intended_ends + np.stack(
        [
            0.24 + 0.24 * (z_particles[:, 0] ** 2 + 0.5 * z_particles[:, 1] ** 2),
            0.42 * z_particles[:, 0] - 0.30 * z_particles[:, 1] + 0.25 * z_particles[:, 0] * z_particles[:, 1],
        ],
        axis=1,
    )
    naive_ends = naive_ends + rng.normal(scale=0.045, size=(len(z_particles), 2))
    first_order_ends = target_first_order(z_particles) + rng.normal(scale=0.025, size=(len(z_particles), 2))
    second_order_ends = target_second_order(z_particles) + rng.normal(scale=0.025, size=(len(z_particles), 2))
    bends = 0.24 * np.sin(particle_angles)

    first_error = paired_endpoint_error(first_order_ends, intended_ends)
    second_error = paired_endpoint_error(second_order_ends, intended_ends)
    naive_error = paired_endpoint_error(naive_ends, intended_ends)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 11,
            "mathtext.fontset": "stix",
            "figure.dpi": 220,
            "savefig.dpi": 300,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.45))
    ax_ref, ax_first, ax_second, ax_legend = axes.ravel()

    for ax in [ax_ref, ax_first, ax_second]:
        setup_axis(ax)
        draw_distribution_background(ax, src_cloud, tgt_cloud)

    draw_particles(ax_ref, starts, naive_ends, curved=False, bends=bends)
    ax_ref.set_title("One-step simple drift transport")
    ax_ref.text(
        -2.12,
        1.05,
        "\n".join(["naive one-step", f"mean error: {naive_error:.3f}"]),
        ha="left",
        va="top",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#feb2b2", alpha=0.9),
    )

    draw_particles(ax_first, starts, first_order_ends, curved=False, bends=bends)
    draw_error_segments(ax_first, first_order_ends, intended_ends)
    ax_first.set_title("One-step first-order transport")
    ax_first.text(
        -2.12,
        1.05,
        "\n".join(["first-order miss", f"mean error: {first_error:.3f}"]),
        ha="left",
        va="top",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#f6ad55", alpha=0.9),
    )

    draw_particles(ax_second, starts, second_order_ends, curved=False, bends=bends)
    ax_second.set_title("One-step second-order transport")
    ax_second.text(
        -2.12,
        1.05,
        "\n".join(["second-order correction", f"mean error: {second_error:.3f}"]),
        ha="left",
        va="top",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#90cdf4", alpha=0.9),
    )

    ax_legend.axis("off")
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#e53e3e", alpha=0.22, markersize=9),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#4299e1", alpha=0.28, markersize=9),
        plt.Line2D([0], [0], marker="o", color="black", markerfacecolor="#e53e3e", markeredgewidth=0.75, markersize=8),
        plt.Line2D([0], [0], marker="o", color="black", markerfacecolor="#3182ce", markeredgewidth=0.75, markersize=8),
        plt.Line2D([0], [0], color="#9f3faf", linewidth=1.6, alpha=0.7),
        plt.Line2D([0], [0], color="#dd6b20", linestyle="--", linewidth=1.3),
    ]
    labels = [
        "Source distribution",
        "Target distribution",
        "Start particles",
        "End particles",
        "Transport path",
        "First-order miss",
    ]
    ax_legend.legend(handles, labels, loc="center", frameon=True, borderpad=0.9, labelspacing=0.65)

    fig.tight_layout(pad=0.9)
    for suffix in ("pdf", "png"):
        out_path = OUT_DIR / f"manifold_transport_toy.{suffix}"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
        print(out_path)


if __name__ == "__main__":
    main()
