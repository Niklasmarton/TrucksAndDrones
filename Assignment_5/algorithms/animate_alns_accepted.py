import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import ALNS_improved as alns


def _embed_2d_from_distance_matrix(distance_matrix):
    import numpy as np

    d = np.array(distance_matrix, dtype=float)
    n = d.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (d ** 2) @ j
    eigvals, eigvecs = np.linalg.eigh(b)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    vals = np.maximum(eigvals[:2], 0.0)
    vecs = eigvecs[:, :2]
    coords = vecs * np.sqrt(vals)
    if coords.shape[1] < 2:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 2 - coords.shape[1]))])
    return coords


def _coords_from_instance_name(instance_name):
    instance_path = alns.TEST_FILES_DIR / instance_name
    instance_data = alns.load_instance(instance_path=instance_path)
    emb = _embed_2d_from_distance_matrix(instance_data["truck_times"])
    return {i: (float(emb[i, 0]), float(emb[i, 1])) for i in range(emb.shape[0])}


def _circle_fallback(max_node):
    coords = {}
    total = max_node + 1
    for node in range(total):
        if node == 0:
            coords[node] = (0.0, 0.0)
            continue
        angle = 2.0 * math.pi * (node - 1) / max(1, total - 1)
        coords[node] = (math.cos(angle), math.sin(angle))
    return coords


def _draw_snapshot(ax, snap, coords, xlim=None, ylim=None):
    ax.clear()
    truck = snap["truck"]
    drone1 = [tuple(t) for t in snap["drone1"]]
    drone2 = [tuple(t) for t in snap["drone2"]]

    xs = [coords[n][0] for n in truck]
    ys = [coords[n][1] for n in truck]
    ax.plot(xs, ys, "-o", color="tab:blue", linewidth=2.0, markersize=4, label="Truck")

    for node, launch_idx, land_idx in drone1:
        lnode = truck[launch_idx]
        rnode = truck[land_idx]
        ax.plot(
            [coords[lnode][0], coords[node][0], coords[rnode][0]],
            [coords[lnode][1], coords[node][1], coords[rnode][1]],
            color="tab:orange",
            linewidth=1.6,
            alpha=0.85,
        )

    for node, launch_idx, land_idx in drone2:
        lnode = truck[launch_idx]
        rnode = truck[land_idx]
        ax.plot(
            [coords[lnode][0], coords[node][0], coords[rnode][0]],
            [coords[lnode][1], coords[node][1], coords[rnode][1]],
            color="tab:green",
            linewidth=1.6,
            alpha=0.85,
        )

    ax.scatter(
        [coords[0][0]],
        [coords[0][1]],
        color="red",
        s=50,
        zorder=5,
        label="Depot",
    )
    ax.set_aspect("equal")
    if xlim is not None and ylim is not None:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    else:
        xs_all = [coords[n][0] for n in coords]
        ys_all = [coords[n][1] for n in coords]
        xmin, xmax = min(xs_all), max(xs_all)
        ymin, ymax = min(ys_all), max(ys_all)
        xpad = max(1e-6, 0.05 * (xmax - xmin if xmax > xmin else 1.0))
        ypad = max(1e-6, 0.05 * (ymax - ymin if ymax > ymin else 1.0))
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.grid(alpha=0.2)
    ax.set_title(
        f"iter={snap['iter']} | phase={snap['phase']} | op={snap['operator']} | "
        f"inc={snap['incumbent_cost']:.1f} | best={snap['best_cost']:.1f}"
    )


def animate_from_snapshot_file(snapshot_file, output_gif):
    with open(snapshot_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    snapshots = payload.get("snapshots", [])
    if not snapshots:
        raise ValueError("No snapshots found in file.")

    max_node = 0
    for s in snapshots:
        max_node = max(max_node, max(s["truck"]))
        for t in s["drone1"]:
            max_node = max(max_node, int(t[0]))
        for t in s["drone2"]:
            max_node = max(max_node, int(t[0]))

    instance_name = payload.get("instance", alns.file_name)
    try:
        coords = _coords_from_instance_name(instance_name)
    except Exception:
        coords = _circle_fallback(max_node)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    xs_all = [coords[n][0] for n in coords]
    ys_all = [coords[n][1] for n in coords]
    xmin, xmax = min(xs_all), max(xs_all)
    ymin, ymax = min(ys_all), max(ys_all)
    xpad = max(1e-6, 0.05 * (xmax - xmin if xmax > xmin else 1.0))
    ypad = max(1e-6, 0.05 * (ymax - ymin if ymax > ymin else 1.0))
    xlim = (xmin - xpad, xmax + xpad)
    ylim = (ymin - ypad, ymax + ypad)

    def _update(frame_idx):
        _draw_snapshot(ax, snapshots[frame_idx], coords, xlim=xlim, ylim=ylim)
        return []

    anim = FuncAnimation(fig, _update, frames=len(snapshots), interval=100, blit=False)
    out = Path(output_gif)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=PillowWriter(fps=10))
    plt.close(fig)
    return str(out)


def run_one_and_animate(
    runs=1,
    warmup_iterations=500,
    iterations=9500,
    final_temperature=0.1,
    snapshot_accept_stride=1,
):
    instance_data = alns.load_instance()
    n_customers = instance_data["n_customers"]
    initial_solution = [[i for i in range(n_customers + 1)] + [0], [], []]

    out_dir = Path(alns.ASSIGNMENT_DIR) / "outputs"
    snapshot_file = out_dir / "accepted_snapshots_run1.json"
    gif_file = out_dir / "accepted_solutions_evolution.gif"

    alns.run_statistics(
        initial_solution,
        instance_data=instance_data,
        runs=runs,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        final_temperature=final_temperature,
        verbose=True,
        return_metrics=False,
        print_solution_pipe=False,
        snapshot_on_accepted=True,
        snapshot_output_file=str(snapshot_file),
        snapshot_accept_stride=snapshot_accept_stride,
    )

    output_path = animate_from_snapshot_file(str(snapshot_file), str(gif_file))
    print(f"Animation written to: {output_path}")


if __name__ == "__main__":
    run_one_and_animate()
