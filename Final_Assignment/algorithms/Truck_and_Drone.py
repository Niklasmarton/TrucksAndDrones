from collections import OrderedDict
from pathlib import Path
from datetime import datetime
import sys
import random
import time
import json
import re

ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
IO_DIR = ASSIGNMENT_DIR / "io"
CORE_DIR = ASSIGNMENT_DIR / "core"
NEW_OPS_DIR = ASSIGNMENT_DIR / "new_operators"
for p in (IO_DIR, CORE_DIR, NEW_OPS_DIR):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from read_file import read_instance
from CalCulateTotalArrivalTime import CalCulateTotalArrivalTime
from FeasibiltyCheck import SolutionFeasibility

import op1_reinsert as op1
import op2_destroy_repair as op2
import op3_or_opt as op3
import op8_related_destroy as op8
import op9_escape_related_large as op9
import op14_double_bridge_escape as op14
import op10_truck_2opt as op10
import op11_TSP_drone_rebuild as op11
import op12_truck_drone_swap as op12
import op13_drone_sync_tuner as op13
import op15_drone_relocate as op15
import op16_optimal_drone_assign as op16
import op17_full_rebuild as op17

import local_search as ls

TEST_FILES_DIR = ASSIGNMENT_DIR.parent / "Test_files"
file_name = "F_100.txt"

def clone_solution(solution):
    return [solution[0][:], solution[1][:], solution[2][:]]


def load_instance(instance_path=None):
    if instance_path is None:
        instance_path = TEST_FILES_DIR / file_name
    return read_instance(str(instance_path))


def build_initial_solution(instance_data):
    """
    Construction heuristic:
    1. Greedy NN walk assigns each customer to truck (0), drone1 (1), or drone2
       (2) in a randomly-shuffled cyclic order.
    2. ALL customers start on the truck in NN visit order (a solid starting
       truck route that op11 can TSP-improve in the first iterations).
    3. Customers designated for a drone are moved off the truck one by one:
       - find the best static-hover-feasible (launch, land) pair in the current
         (shrinking) truck, avoiding already-used endpoints
       - if no valid pair exists the customer stays on the truck
    4. An availability pass removes any drone trip where the drone can't finish
       its prior mission before the next scheduled launch would require hovering
       past the flight limit; those customers go back to the truck.
    The result is always a valid feasible solution — no final fallback needed.
    """
    n_customers = instance_data["n_customers"]
    depot = instance_data.get("depot_index", 0)
    T = instance_data["truck_times"]
    D = instance_data["drone_times"]
    flight_limit = instance_data["flight_limit"]

    # --- Step 1: NN walk with cyclic vehicle assignment ---
    unvisited = list(range(1, n_customers + 1))
    visit_order = []
    assignment = {}  # node -> 0 (truck), 1 (drone1), 2 (drone2)
    current = depot
    cycle = [0, 1, 2]
    random.shuffle(cycle)
    step = 0
    while unvisited:
        nearest = min(unvisited, key=lambda n: T[current][n])
        unvisited.remove(nearest)
        current = nearest
        visit_order.append(nearest)
        assignment[nearest] = cycle[step % 3]
        step += 1

    # --- Step 2: All customers on the truck in NN visit order ---
    truck = [depot] + visit_order + [depot]
    drone1, drone2 = [], []

    # --- Helpers ---
    def _truck_prefix():
        prefix = [0.0]
        for i in range(len(truck) - 1):
            prefix.append(prefix[-1] + T[truck[i]][truck[i + 1]])
        return prefix

    def _shift_after_remove(route, removed_idx):
        """Remap indices after removing truck[removed_idx]."""
        return [(n, l - (l > removed_idx), r - (r > removed_idx))
                for n, l, r in route]

    def _shift_after_insert(route, insert_idx):
        """Remap indices after inserting at truck position insert_idx."""
        return [(n, l + (l >= insert_idx), r + (r >= insert_idx))
                for n, l, r in route]

    def _find_pair_in(node, temp_truck, drone_route):
        """
        Find the best (launch_idx, land_idx) for node in temp_truck.
        Checks:
          - Raw drone flight <= flight_limit
          - Static hover: max(drone_flight, truck_segment) <= flight_limit
          - Timeline: the pair fits in at least one gap in the existing drone_route
            (launch >= previous trip's land, landing <= next trip's launch)
        drone_route is already index-adjusted for temp_truck.
        """
        nt = len(temp_truck)
        prefix = [0.0]
        for i in range(nt - 1):
            prefix.append(prefix[-1] + T[temp_truck[i]][temp_truck[i + 1]])
        used_l = {l for _, l, _ in drone_route}
        used_r = {r for _, _, r in drone_route}

        # Pre-build sorted (prev_land, next_launch) pairs for all insertion gaps.
        sorted_trips = sorted(drone_route, key=lambda x: (x[1], x[2], x[0]))
        gaps = []  # (prev_land, next_launch) for each insertion slot
        for i in range(len(sorted_trips) + 1):
            prev_land = 0 if i == 0 else sorted_trips[i - 1][2]
            next_launch = float("inf") if i == len(sorted_trips) else sorted_trips[i][1]
            gaps.append((prev_land, next_launch))

        def _fits_in_gap(l, r):
            """True if (l, r) fits in any timeline gap."""
            for prev_land, next_launch in gaps:
                if l >= prev_land and r <= next_launch:
                    return True
            return False

        # Use a conservative limit to leave headroom for cross-drone truck delays
        # that the static check can't see (drone2 landing delays truck at an
        # intermediate node, increasing the effective segment for drone1).
        safe_limit = flight_limit * 0.88

        best_score, best_l, best_r = float("inf"), None, None
        for l in range(1, nt - 1):
            if l in used_l:
                continue
            for r in range(l + 1, nt - 1):
                if r in used_r:
                    continue
                if not _fits_in_gap(l, r):
                    continue
                df = D[temp_truck[l]][node] + D[node][temp_truck[r]]
                if df > safe_limit:
                    continue
                ts = prefix[r] - prefix[l]
                if max(df, ts) > safe_limit:
                    continue
                hover = max(0.0, ts - df)
                score = 2.0 * hover + 0.5 * abs(df - ts) + 0.01 * df
                if score < best_score:
                    best_score = score
                    best_l, best_r = l, r
        return (best_l, best_r) if best_l is not None else None

    def _availability_pass(drone_route):
        """
        Walk trips in launch order.  For each trip, compute effective flight
        accounting for drone availability (it can't launch before it returns
        from its prior trip).  Remove trips that would require hovering past
        flight_limit.  Returns (kept_trips, fallback_customers).
        """
        prefix = _truck_prefix()
        trips = sorted(drone_route, key=lambda x: (x[1], x[2], x[0]))
        kept, fallback, drone_est_return = [], [], 0.0
        for node, l, r in trips:
            df = D[truck[l]][node] + D[node][truck[r]]
            actual_launch = max(prefix[l], drone_est_return)
            drone_return_t = actual_launch + df
            drone_hover = max(0.0, prefix[r] - drone_return_t)
            if df + drone_hover > flight_limit:
                fallback.append(node)
            else:
                kept.append((node, l, r))
                drone_est_return = drone_return_t
        return kept, fallback

    # --- Step 3: Move drone-assigned customers from truck to their drone ---
    for target, drone_route in [(1, drone1), (2, drone2)]:
        candidates = [n for n in visit_order if assignment[n] == target]
        for node in candidates:
            node_idx = truck.index(node)
            # Guard: if this truck position is already a launch or land node for
            # an existing drone trip, removing it would corrupt those indices.
            all_used_l = {l for _, l, _ in drone1} | {l for _, l, _ in drone2}
            all_used_r = {r for _, _, r in drone1} | {r for _, _, r in drone2}
            if node_idx in all_used_l or node_idx in all_used_r:
                continue
            temp_truck = truck[:node_idx] + truck[node_idx + 1:]
            # Shift both drone routes for this removal (before finding the pair)
            shifted_d1 = _shift_after_remove(drone1, node_idx)
            shifted_d2 = _shift_after_remove(drone2, node_idx)
            shifted_target = shifted_d1 if target == 1 else shifted_d2
            pair = _find_pair_in(node, temp_truck, shifted_target)
            if pair is None:
                continue  # customer stays on truck
            # Commit: update state in-place
            truck[:] = temp_truck
            drone1[:] = shifted_d1
            drone2[:] = shifted_d2
            drone_route.append((node, pair[0], pair[1]))
            drone_route.sort(key=lambda x: (x[1], x[2], x[0]))

    # --- Step 4: Availability pass — fix multi-trip drone timeline violations ---
    drone1[:], fb1 = _availability_pass(drone1)
    drone2[:], fb2 = _availability_pass(drone2)
    for node in fb1 + fb2:
        _, ins_idx = _best_truck_insert(node, truck, T)
        if ins_idx is None:
            ins_idx = len(truck) - 1
        drone1[:] = _shift_after_insert(drone1, ins_idx)
        drone2[:] = _shift_after_insert(drone2, ins_idx)
        truck[:] = truck[:ins_idx] + [node] + truck[ins_idx:]

    # --- Step 5: Final safety pass using the dynamic calculator ---
    # Cross-drone cascading effects (drone2 landing delays truck at an
    # intermediate node, increasing effective flight for a concurrent drone1 trip)
    # can slip through the static checks above.  Iteratively remove drone trips
    # until the dynamic simulation passes, then put failed customers on the truck.
    _, calc, checker = build_evaluator(instance_data)
    for _attempt in range(len(drone1) + len(drone2) + 2):
        feasible, _ = evaluate_solution([truck, drone1, drone2], calc, checker)
        if feasible:
            break
        # Remove the last trip from whichever drone has more trips.
        if not drone1 and not drone2:
            break
        target_route = drone1 if len(drone1) >= len(drone2) else drone2
        worst = target_route[-1]
        target_route.pop()
        node_to_restore = worst[0]
        _, ins_idx = _best_truck_insert(node_to_restore, truck, T)
        if ins_idx is None:
            ins_idx = len(truck) - 1
        drone1[:] = _shift_after_insert(drone1, ins_idx)
        drone2[:] = _shift_after_insert(drone2, ins_idx)
        truck[:] = truck[:ins_idx] + [node_to_restore] + truck[ins_idx:]

    return [truck, drone1, drone2]


def unpack_instance(instance_data):
    return {
        "T": instance_data["truck_times"],
        "D": instance_data["drone_times"],
        "flight_limit": instance_data["flight_limit"],
        "n_customers": instance_data["n_customers"],
        "depot": instance_data.get("depot_index", 0),
    }


def configure_operator_context(instance_data):
    T = instance_data["truck_times"]
    D = instance_data["drone_times"]
    fr = instance_data["flight_limit"]
    depot = instance_data.get("depot_index", 0)
    for op in (op1, op2, op3, op8, op9, op10, op11, op12, op13, op15, op16, op17):
        if hasattr(op, "set_operator_context"):
            op.set_operator_context(T, D, fr, depot)


def configure_operator_search_progress(progress):
    for op in (op1, op2, op3, op8, op9, op10, op11, op12, op13, op15, op16, op17):
        if hasattr(op, "set_search_progress"):
            op.set_search_progress(progress)


def build_evaluator(instance_data):
    ctx = unpack_instance(instance_data)
    calc = CalCulateTotalArrivalTime()
    calc.truck_times = ctx["T"]
    calc.drone_times = ctx["D"]
    calc.flight_range = ctx["flight_limit"]
    calc.depot_index = ctx["depot"]

    checker = SolutionFeasibility(
        n_nodes=ctx["n_customers"] + 1,
        n_drones=2,
        depot_index=ctx["depot"],
        drone_times=ctx["D"],
        flight_range=ctx["flight_limit"],
    )
    return ctx, calc, checker


def to_parts_solution(solution):
    truck, drone1, drone2 = solution
    drone_serving_1 = [node for node, _, _ in drone1]
    drone_serving_2 = [node for node, _, _ in drone2]
    launch_indices_1 = [launch_idx + 1 for _, launch_idx, _ in drone1]
    landing_indices_1 = [land_idx + 1 for _, _, land_idx in drone1]
    launch_indices_2 = [launch_idx + 1 for _, launch_idx, _ in drone2]
    landing_indices_2 = [land_idx + 1 for _, _, land_idx in drone2]
    return {
        "part1": truck,
        "part2": drone_serving_1 + [-1] + drone_serving_2,
        "part3": launch_indices_1 + [-1] + launch_indices_2,
        "part4": landing_indices_1 + [-1] + landing_indices_2,
    }


def evaluate_solution(solution, calc, checker):
    parts_solution = to_parts_solution(solution)
    if not checker.is_solution_feasible(parts_solution):
        return False, float("inf")

    total_time, _, _, calc_feasible = calc.calculate_total_waiting_time(parts_solution)
    if not calc_feasible:
        return False, float("inf")
    return True, total_time


def solution_key(solution):
    truck, drone1, drone2 = solution
    return (tuple(truck), tuple(drone1), tuple(drone2))


def fast_precheck_solution(solution, ctx):
    truck, drone1, drone2 = solution
    D = ctx["D"]
    flight_limit = ctx["flight_limit"]
    n_customers = ctx["n_customers"]
    depot = ctx["depot"]

    if len(truck) < 2 or truck[0] != depot or truck[-1] != depot:
        return False

    truck_customers = [node for node in truck if node != depot]
    if len(truck_customers) != len(set(truck_customers)):
        return False

    drone_nodes = []
    for route in (drone1, drone2):
        prev_land = 0
        used_launch = set()
        used_land = set()
        for node, launch_idx, land_idx in route:
            if node == depot:
                return False
            if not (0 <= launch_idx < land_idx < len(truck)):
                return False
            if launch_idx in used_launch or land_idx in used_land:
                return False
            if launch_idx < prev_land:
                return False

            launch_node = truck[launch_idx]
            land_node = truck[land_idx]
            if D[launch_node][node] + D[node][land_node] > flight_limit:
                return False

            used_launch.add(launch_idx)
            used_land.add(land_idx)
            prev_land = land_idx
            drone_nodes.append(node)

    if len(drone_nodes) != len(set(drone_nodes)):
        return False

    all_served = truck_customers + drone_nodes
    if len(all_served) != n_customers:
        return False
    if len(set(all_served)) != n_customers:
        return False
    return True


def format_solution_pipe(solution):
    parts = to_parts_solution(solution)
    return (
        f"{','.join(str(x) for x in parts['part1'])} | "
        f"{','.join(str(x) for x in parts['part2'])} | "
        f"{','.join(str(x) for x in parts['part3'])} | "
        f"{','.join(str(x) for x in parts['part4'])}"
    )


def _append_best_runs_log(
    log_file,
    instance_label,
    run_records,
    average_cost,
    overall_best_cost,
    average_runtime,
    best_solution_pipe,
):
    out_path = Path(log_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")

    with open(out_path, "a", encoding="utf-8") as f:
        f.write(
            f"=== {timestamp} | instance={instance_label} | runs={len(run_records)} ===\n"
        )
        for rec in run_records:
            f.write(
                f"run {rec['run_id']}: best_cost={rec['best_cost']:.1f}, "
                f"best_iter={rec['best_iter']}, runtime={rec['runtime']:.4f}s\n"
            )
        f.write(f"average_cost={average_cost:.1f}\n")
        f.write(f"average_runtime={average_runtime:.4f}s\n")
        f.write(f"overall_best_cost={overall_best_cost:.1f}\n")
        f.write(f"best_solution_pipe={best_solution_pipe}\n")
        f.write("\n")

    return str(out_path)


def _snapshot_record(iter_idx, phase, op_name, incumbent_cost, best_cost, solution):
    return {
        "iter": int(iter_idx),
        "phase": phase,
        "operator": op_name,
        "incumbent_cost": float(incumbent_cost),
        "best_cost": float(best_cost),
        "truck": solution[0][:],
        "drone1": [list(t) for t in solution[1]],
        "drone2": [list(t) for t in solution[2]],
    }


def _embed_2d_from_distance_matrix(distance_matrix):
    """Classic MDS: embed nodes into 2D using the truck-time matrix as distances."""
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


def _plot_best_solution(solution, instance_data, output_dir, instance_label, cost):
    """
    Save a visualisation of the best solution to output_dir/best_visualised/.

    Truck route: solid blue line with arrows.
    Drone 1 trips: dashed orange line (launch → customer → land).
    Drone 2 trips: dashed green line (launch → customer → land).
    Nodes are labelled with their index; depot is a red square.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return ""

    truck, drone1, drone2 = solution
    coords = _embed_2d_from_distance_matrix(instance_data["truck_times"])
    depot = instance_data.get("depot_index", 0)

    drone1_nodes = {t[0] for t in drone1}
    drone2_nodes = {t[0] for t in drone2}
    truck_only = [n for n in truck if n != depot and n not in drone1_nodes and n not in drone2_nodes]

    fig, ax = plt.subplots(figsize=(11, 9))
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(instance_label))
    ax.set_title(
        f"{instance_label}  —  best cost: {cost:.1f}\n"
        f"truck stops: {len(truck_only)}  |  drone1: {len(drone1)}  |  drone2: {len(drone2)}",
        fontsize=11,
    )
    ax.set_xlabel("x (MDS projection)")
    ax.set_ylabel("y (MDS projection)")

    # All nodes as faint grey background dots
    ax.scatter(coords[:, 0], coords[:, 1], s=18, c="lightgrey", zorder=1)

    # Truck-only customers
    if truck_only:
        tx = [coords[n, 0] for n in truck_only]
        ty = [coords[n, 1] for n in truck_only]
        ax.scatter(tx, ty, s=55, c="tab:blue", zorder=3, label="Truck customer")

    # Drone1 customers
    if drone1_nodes:
        d1x = [coords[n, 0] for n in drone1_nodes]
        d1y = [coords[n, 1] for n in drone1_nodes]
        ax.scatter(d1x, d1y, s=55, c="tab:orange", zorder=3, label="Drone 1 customer")

    # Drone2 customers
    if drone2_nodes:
        d2x = [coords[n, 0] for n in drone2_nodes]
        d2y = [coords[n, 1] for n in drone2_nodes]
        ax.scatter(d2x, d2y, s=55, c="tab:green", zorder=3, label="Drone 2 customer")

    # Depot
    ax.scatter(
        [coords[depot, 0]], [coords[depot, 1]],
        s=160, c="red", marker="s", zorder=5, label="Depot",
    )

    # Truck route arrows
    for i in range(len(truck) - 1):
        a, b = truck[i], truck[i + 1]
        dx = coords[b, 0] - coords[a, 0]
        dy = coords[b, 1] - coords[a, 1]
        ax.annotate(
            "",
            xy=(coords[b, 0], coords[b, 1]),
            xytext=(coords[a, 0], coords[a, 1]),
            arrowprops=dict(
                arrowstyle="-|>",
                color="tab:blue",
                lw=1.6,
                alpha=0.75,
                mutation_scale=12,
            ),
            zorder=2,
        )

    # Drone 1 trips
    for node, launch_idx, land_idx in drone1:
        ln = truck[launch_idx]
        rn = truck[land_idx]
        xs = [coords[ln, 0], coords[node, 0], coords[rn, 0]]
        ys = [coords[ln, 1], coords[node, 1], coords[rn, 1]]
        ax.plot(xs, ys, color="tab:orange", linestyle="--", linewidth=1.5, alpha=0.85, zorder=2)

    # Drone 2 trips
    for node, launch_idx, land_idx in drone2:
        ln = truck[launch_idx]
        rn = truck[land_idx]
        xs = [coords[ln, 0], coords[node, 0], coords[rn, 0]]
        ys = [coords[ln, 1], coords[node, 1], coords[rn, 1]]
        ax.plot(xs, ys, color="tab:green", linestyle="--", linewidth=1.5, alpha=0.85, zorder=2)

    # Node labels
    n_nodes = len(instance_data["truck_times"])
    fontsize = max(5, min(8, 80 // n_nodes))
    for i in range(n_nodes):
        ax.annotate(
            str(i),
            (coords[i, 0], coords[i, 1]),
            fontsize=fontsize,
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
            zorder=6,
        )

    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)
    plt.tight_layout()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"best_solution_{safe_label}.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return str(out_file)


def _plot_operator_deltas(delta_points_by_op, op_names, output_dir, instance_label, run_label):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    safe_instance = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(instance_label))
    safe_run = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_label))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for op_name in op_names:
        points = delta_points_by_op.get(op_name, [])
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(xs, ys, s=10, alpha=0.6, color="tab:blue")
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
        ax.set_title(f"Operator {op_name} Delta Values ({instance_label}, {run_label})")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Delta Objective (new - incumbent)")
        ax.grid(alpha=0.25)
        plt.tight_layout()

        out_file = out_dir / f"delta_scatter_{safe_instance}_{safe_run}_{op_name}.png"
        fig.savefig(out_file, dpi=150)
        plt.close(fig)
        saved.append(str(out_file))

    return saved


def _plot_operator_weights(weight_history, op_names, output_dir, instance_label, run_label):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return ""

    if not weight_history:
        return ""

    safe_instance = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(instance_label))
    safe_run = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_label))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    segments = list(range(1, len(weight_history) + 1))
    fig, ax = plt.subplots(figsize=(9, 5))
    for op_name in op_names:
        ys = [float(w.get(op_name, 0.0)) for _, w in weight_history]
        ax.plot(segments, ys, marker="o", linewidth=1.5, markersize=3, label=op_name)

    ax.set_title(f"Operator Weights by Segment ({instance_label}, {run_label})")
    ax.set_xlabel("Segment Number")
    ax.set_ylabel("Operator Weight")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()

    out_file = out_dir / f"operator_weights_{safe_instance}_{safe_run}.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return str(out_file)


def _plot_temperature_history(temperature_history, output_dir, instance_label, run_label):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return ""

    if not temperature_history:
        return ""

    safe_instance = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(instance_label))
    safe_run = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_label))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xs = [int(p[0]) for p in temperature_history]
    ys = [float(p[1]) for p in temperature_history]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, ys, linewidth=1.6, color="tab:red")
    ax.set_title(f"RRT Deviation Across Iterations ({instance_label}, {run_label})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Allowed Deviation D")
    ax.grid(alpha=0.25)
    plt.tight_layout()

    out_file = out_dir / f"temperature_plot_{safe_instance}_{safe_run}.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return str(out_file)


def _plot_acceptance_probability(acceptance_points, output_dir, instance_label, run_label):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return ""

    if not acceptance_points:
        return ""

    safe_instance = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(instance_label))
    safe_run = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_label))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xs = [int(p[0]) for p in acceptance_points]
    ys = [float(p[1]) for p in acceptance_points]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(xs, ys, s=10, alpha=0.55, color="tab:purple")
    ax.set_title(f"RRT Acceptance Indicator for Worsening Moves ({instance_label}, {run_label})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Acceptable Under RRT (0/1)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    plt.tight_layout()

    out_file = out_dir / f"acceptance_probability_{safe_instance}_{safe_run}.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return str(out_file)


def _plot_accepted_objective(accepted_objective_points, output_dir, instance_label, run_label):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return ""

    if not accepted_objective_points:
        return ""

    safe_instance = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(instance_label))
    safe_run = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_label))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xs = [int(p[0]) for p in accepted_objective_points]
    ys = [float(p[1]) for p in accepted_objective_points]

    best_so_far = []
    running = float("inf")
    for y in ys:
        if y < running:
            running = y
        best_so_far.append(running)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax_top.plot(xs, ys, linewidth=1.2, color="tab:blue", alpha=0.9)
    ax_top.scatter(xs, ys, s=8, alpha=0.55, color="tab:blue")
    ax_top.set_title(f"Accepted Solutions Objective by Iteration ({instance_label}, {run_label})")
    ax_top.set_ylabel("Accepted objective")
    ax_top.grid(alpha=0.25)

    ax_bot.plot(xs, best_so_far, linewidth=1.5, color="tab:red")
    ax_bot.set_title("Best-so-far (running min)")
    ax_bot.set_xlabel("Iteration")
    ax_bot.set_ylabel("Best objective")
    ax_bot.grid(alpha=0.25)
    if best_so_far:
        final = best_so_far[-1]
        ax_bot.axhline(final, color="tab:gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax_bot.annotate(f"{final:.1f}", xy=(xs[-1], final), xytext=(-60, 8),
                        textcoords="offset points", fontsize=9, color="tab:gray")
    plt.tight_layout()

    out_file = out_dir / f"accepted_objective_{safe_instance}_{safe_run}.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return str(out_file)


def _init_stats(op_names):
    return {
        op_name: {
            "used": 0,
            "feasible": 0,
            "accepted": 0,
            "improved": 0,
            "uphill_accepted": 0,
            "uphill_rejected": 0,
            "worse_feasible": 0,
            "delta_sum": 0.0,
            "improve_delta_sum": 0.0,
            "uphill_accepted_delta_sum": 0.0,
            "p_accept_sum": 0.0,
            "p_accept_count": 0,
        }
        for op_name in op_names
    }


def _normalize_weight_dict(weights, min_weight=0.03):
    fixed = {k: max(min_weight, float(v)) for k, v in weights.items()}
    s = sum(fixed.values())
    return {k: v / s for k, v in fixed.items()}


def _normalize_weight_dict_with_caps(
    weights,
    min_weight=0.03,
    lower_caps=None,
    upper_caps=None,
):
    lower_caps = lower_caps or {}
    upper_caps = upper_caps or {}
    keys = list(weights.keys())
    if not keys:
        return {}

    base = {k: max(min_weight, float(weights[k])) for k in keys}
    lower = {k: max(min_weight, float(lower_caps.get(k, min_weight))) for k in keys}
    upper = {k: float(upper_caps.get(k, 1.0)) for k in keys}

    for k in keys:
        if upper[k] < lower[k]:
            upper[k] = lower[k]

    x = {k: min(max(base[k], lower[k]), upper[k]) for k in keys}

    # Bounded re-scaling so sum(x)=1 while preserving per-operator bounds.
    for _ in range(24):
        s = sum(x.values())
        if abs(s - 1.0) <= 1e-12:
            break

        if s > 1.0:
            free = [k for k in keys if x[k] > lower[k] + 1e-12]
            if not free:
                break
            excess = s - 1.0
            room = sum(x[k] - lower[k] for k in free)
            if room <= 1e-12:
                break
            for k in free:
                take = excess * ((x[k] - lower[k]) / room)
                x[k] = max(lower[k], x[k] - take)
        else:
            free = [k for k in keys if x[k] < upper[k] - 1e-12]
            if not free:
                break
            deficit = 1.0 - s
            room = sum(upper[k] - x[k] for k in free)
            if room <= 1e-12:
                break
            for k in free:
                add = deficit * ((upper[k] - x[k]) / room)
                x[k] = min(upper[k], x[k] + add)

    # Final normalize for numerical cleanliness.
    total = sum(x.values())
    if total <= 0:
        return _normalize_weight_dict(weights, min_weight=min_weight)
    return {k: x[k] / total for k in keys}


def _phase_weight_summary(weight_history, op_names, iterations, fallback_weights):
    buckets = {
        "early": {k: [] for k in op_names},
        "mid": {k: [] for k in op_names},
        "late": {k: [] for k in op_names},
    }
    for iter_idx, w in weight_history:
        progress = (iter_idx / iterations) if iterations > 0 else 1.0
        if progress <= (1.0 / 3.0):
            phase = "early"
        elif progress <= (2.0 / 3.0):
            phase = "mid"
        else:
            phase = "late"
        for k in op_names:
            buckets[phase][k].append(float(w.get(k, 0.0)))

    summary = {}
    for phase in ("early", "mid", "late"):
        summary[phase] = {}
        for k in op_names:
            values = buckets[phase][k]
            if values:
                summary[phase][k] = sum(values) / len(values)
            else:
                summary[phase][k] = float(fallback_weights.get(k, 0.0))
    return summary


def roulette_pick(weights, op_names):
    r = random.random()
    acc = 0.0
    for op_name in op_names:
        acc += weights[op_name]
        if r <= acc:
            return op_name
    return op_names[-1]


def apply_main_operator(solution, op_name):
    if op_name == "op1":
        return op1.operator(solution)
    if op_name == "op2":
        return op2.operator(solution)
    if op_name == "op3":
        return op3.operator(solution)
    if op_name == "op8":
        return op8.operator(solution)
    if op_name == "op10":
        return op10.operator(solution)
    if op_name == "op11":
        return op11.operator(solution)
    if op_name == "op12":
        return op12.operator(solution)
    if op_name == "op13":
        return op13.operator(solution)
    if op_name == "op15":
        return op15.operator(solution)
    if op_name == "op16":
        return op16.operator(solution)
    if op_name == "op17":
        return op17.operator(solution)


def _truck_removal_gain(truck, idx, truck_times):
    if idx <= 0 or idx >= len(truck) - 1:
        return -1e18
    a = truck[idx - 1]
    b = truck[idx]
    c = truck[idx + 1]
    return truck_times[a][b] + truck_times[b][c] - truck_times[a][c]


def _best_truck_insert(node, truck, truck_times):
    best_delta = None
    best_idx = None
    for ins_idx in range(1, len(truck)):
        a = truck[ins_idx - 1]
        b = truck[ins_idx]
        delta = truck_times[a][node] + truck_times[node][b] - truck_times[a][b]
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_idx = ins_idx
    return best_delta, best_idx


def _remove_conflicting_and_shift_after_removal(route, removed_idx, removed_node):
    kept = []
    orphan_customers = []
    for cust, launch_idx, land_idx in route:
        if cust == removed_node or launch_idx == removed_idx or land_idx == removed_idx:
            orphan_customers.append(cust)
            continue
        if launch_idx > removed_idx:
            launch_idx -= 1
        if land_idx > removed_idx:
            land_idx -= 1
        kept.append((cust, launch_idx, land_idx))
    kept.sort(key=lambda x: (x[1], x[2], x[0]))
    return kept, orphan_customers


def _remove_conflicts_and_shift_after_insert(route, insert_idx):
    kept = []
    orphan_customers = []
    for cust, launch_idx, land_idx in route:
        if launch_idx == insert_idx or land_idx == insert_idx:
            orphan_customers.append(cust)
            continue
        if launch_idx >= insert_idx:
            launch_idx += 1
        if land_idx >= insert_idx:
            land_idx += 1
        kept.append((cust, launch_idx, land_idx))
    kept.sort(key=lambda x: (x[1], x[2], x[0]))
    return kept, orphan_customers


def _aggressive_escape_destroy_repair(solution, ctx):
    truck_times = ctx["T"]
    truck, drone1, drone2 = clone_solution(solution)

    if len(truck) <= 7:
        return solution

    scored = []
    for idx in range(1, len(truck) - 1):
        gain = _truck_removal_gain(truck, idx, truck_times)
        scored.append((gain, idx))
    if not scored:
        return solution

    scored.sort(key=lambda x: x[0], reverse=True)
    top10 = scored[: min(10, len(scored))]
    if len(top10) <= 5:
        picked = [idx for _, idx in top10]
    else:
        picked = [idx for _, idx in random.sample(top10, 5)]
    picked = sorted(set(picked), reverse=True)

    pending = []
    for rem_idx in picked:
        rem_node = truck[rem_idx]
        pending.append(rem_node)
        truck = truck[:rem_idx] + truck[rem_idx + 1 :]
        drone1, orphan1 = _remove_conflicting_and_shift_after_removal(drone1, rem_idx, rem_node)
        drone2, orphan2 = _remove_conflicting_and_shift_after_removal(drone2, rem_idx, rem_node)
        pending.extend(orphan1)
        pending.extend(orphan2)

    pending_unique = []
    seen = set()
    for node in pending:
        if node in seen:
            continue
        seen.add(node)
        pending_unique.append(node)

    for node in pending_unique:
        _, ins_idx = _best_truck_insert(node, truck, truck_times)
        if ins_idx is None:
            continue
        drone1, orphan1 = _remove_conflicts_and_shift_after_insert(drone1, ins_idx)
        drone2, orphan2 = _remove_conflicts_and_shift_after_insert(drone2, ins_idx)
        truck = truck[:ins_idx] + [node] + truck[ins_idx:]
        for orphan in orphan1 + orphan2:
            if orphan not in seen:
                seen.add(orphan)
                pending_unique.append(orphan)

    return [truck, drone1, drone2]


def _escape_with_related_large(
    incumbent,
    incumbent_cost,
    best_solution,
    best_cost,
    ctx,
    cached_evaluate,
):
    # Always restart from best known solution, not the drifted incumbent.
    # This ensures escape explores a new basin relative to the global best,
    # not from a degraded working point accumulated by uphill RRT moves.
    current = clone_solution(best_solution)
    current_cost = best_cost
    improved_best = False
    feasible_steps = 0

    # Primary escape: double-bridge (op14).
    # A single double-bridge is a complete topological perturbation — try up
    # to 3 times in case of edge cases (very short route, remap failure).
    # Expected quality impact: 1-5% above best, recovering in 50-100 iters.
    for _ in range(3):
        cand = op14.operator(current)
        if cand is current or cand == current:
            continue
        if not fast_precheck_solution(cand, ctx):
            continue
        feasible, cand_cost = cached_evaluate(cand)
        if not feasible:
            continue
        current = cand
        current_cost = cand_cost
        feasible_steps += 1
        if current_cost < best_cost:
            improved_best = True
        break  # one successful double-bridge is the full escape

    # Fallback to op9 if double-bridge could not fire (e.g. too few nodes).
    # Suggestion 1: reduced step count (was n_customers // 5 = 20 for n=100)
    # to avoid compounding random degradation to 40%+ above best.
    if feasible_steps == 0:
        n_customers = ctx.get("n_customers", 50)
        n_escape_steps = max(3, min(6, n_customers // 15))
        for _ in range(n_escape_steps):
            cand = op9.operator(current)
            if cand is current or cand == current:
                continue
            if not fast_precheck_solution(cand, ctx):
                continue
            feasible, cand_cost = cached_evaluate(cand)
            if not feasible:
                continue
            current = cand
            current_cost = cand_cost
            feasible_steps += 1
            if current_cost < best_cost:
                improved_best = True

    # Final fallback: aggressive destroy/repair.
    if feasible_steps == 0:
        fallback = _aggressive_escape_destroy_repair(clone_solution(best_solution), ctx)
        if fallback != best_solution and fast_precheck_solution(fallback, ctx):
            feasible, fb_cost = cached_evaluate(fallback)
            if feasible:
                current = fallback
                current_cost = fb_cost
                feasible_steps += 1
                if current_cost < best_cost:
                    improved_best = True

    return current, current_cost, improved_best, feasible_steps


def alns_improved(
    initial_solution,
    instance_data=None,
    warmup_iterations=500,
    iterations=9500,
    final_temperature=0.1,
    cache_limit=200000,
    reaction_factor=0.08,
    segment_length=200,
    escape_stall_limit=650,
    ctx=None,
    calc=None,
    checker=None,
    shared_eval_cache=None,
    snapshot_on_accepted=False,
    snapshot_accept_stride=1,
    snapshot_every_iteration=False,
    snapshot_iteration_stride=25,
    reward_improve_threshold=50.0,
    uphill_reward_cap=30.0,
    sigma_global_best=8.0,
    sigma_incumbent_improve=4.0,
    sigma_small_improve=1.0,
    sigma_uphill_accepted=0.6,
    warmup_delta_trim_quantile=0.9,
    collect_delta_points=False,
    collect_temperature_history=False,
    collect_acceptance_points=False,
    collect_accepted_objective_points=False,
    rrt_deviation_factor=0.13,
    rrt_decay_exponent=1.0,
    time_limit_seconds=None,
    op_names_override=None,
    enable_local_search=False,
    ls_time_budget_seconds=5.0,
    ls_max_cycles=20,
    ls_at_new_best=True,
    ls_at_end=True,
    ls_end_time_budget_seconds=10.0,
    ls_ils_passes=1,
):
    if instance_data is None:
        instance_data = load_instance()
    if ctx is None or calc is None or checker is None:
        ctx, calc, checker = build_evaluator(instance_data)
        configure_operator_context(instance_data)

    # Scale operator set to instance size.
    # On F_20 (16-30 customers): op3/op8/op11 contribute very little
    # (improve_per_1000 < 2.5k, op11 acceptance <2%) and waste ~40% of budget.
    # Freeing that budget for op1/op2/op10 improved F_20 best from 3305→3295.
    # On F_10 (≤15 customers): op3/op8 provide essential diversification —
    # removing them stops the algorithm finding 1412 entirely. Keep all ops.
    n_cust = ctx.get("n_customers", 100)
    if op_names_override is not None:
        op_names = list(op_names_override)
        if n_cust > 30:
            rrt_deviation_factor = 0.05
    elif n_cust <= 15:
        # F_10/R_10 size. Full set so ALNS can adapt per instance type:
        # - F_10: op3/op8/op11 diversify; op12 earns ~0.22 weight for truck↔drone swap
        # - R_10: op13 earns high weight for timing tuning; op12 also helps
        # op13 at floor weight (~0.03) on instances where it doesn't help costs very little.
        # op16 disabled: it deterministically converges to the ~807 basin on R_10
        # (verified across runs), pulling other operators' improvements back into it.
        op_names = ["op1", "op2", "op3", "op8", "op10", "op11", "op12", "op13", "op17"]
    elif 16 <= n_cust <= 30:
        # F_20-size: op12 is very strong (found 3274); op13 helps timing.
        # op3 added: or-opt segment moves help on R instances where single-node
        # relocate (op1) misses improvements requiring two adjacent nodes to move.
        op_names = ["op1", "op2", "op3", "op10", "op12", "op13", "op17"]
    elif 31 <= n_cust <= 60:
        # n=50 (R_50/F_50): same op set as n=100 but linear RRT decay.
        # Convex decay (p=1.5) regresses R_50 by +136 avg (95% CI [+12,+260], sig)
        # because basin lottery is not the limiter at n=50 — the algorithm finds
        # good basins reliably and the wider Q1 window just wastes budget.
        op_names = ["op1", "op2", "op3", "op12", "op13", "op15"]
        rrt_deviation_factor = 0.05
    else:
        # F_100/R_100 and contest blind n=100 instances: op8 and op10 both earn
        # <8k improve/1k with the tight RRT window and decay to near min_weight
        # floor — remove both. op15 added: cross-drone relocate explores the
        # drone partition dimension that no other operator covers.
        # op11/op17 removed: 8-run R_100 ablation -260 avg / -223 best, 4-run
        # F_100 ablation -1041 avg / -1160 best (new low 28898).
        op_names = ["op1", "op2", "op3", "op12", "op13", "op15"]
        # Tighten RRT acceptance: 0.05 (was 0.13 default) gives tighter mid-run
        # window for faster convergence at n=100.
        rrt_deviation_factor = 0.05
        # Convex RRT decay (frac = ((G-g)/G)^1.5) holds the acceptance window
        # wider through Q1-Q2 to reduce basin-lottery variance. 10x4 head-to-head
        # showed -avg on F_100/Contest/Contest_new and new all-time lows on F_100
        # (29005) and Contest_new (22399); std reduced ~15-20% across the four
        # n=100 datasets. NOT applied at n=50 (regresses R_50 significantly).
        rrt_decay_exponent = 1.5

    eval_cache = shared_eval_cache if shared_eval_cache is not None else OrderedDict()

    def cached_evaluate(sol):
        key = solution_key(sol)
        cached = eval_cache.get(key)
        if cached is not None:
            eval_cache.move_to_end(key)
            return cached

        result = evaluate_solution(sol, calc, checker)
        eval_cache[key] = result
        if cache_limit is not None and cache_limit > 0 and len(eval_cache) > cache_limit:
            eval_cache.popitem(last=False)
        return result

    stats = _init_stats(op_names)

    incumbent = clone_solution(initial_solution)
    incumbent_feasible, incumbent_cost = cached_evaluate(incumbent)
    if not incumbent_feasible:
        raise ValueError("Initial solution is not feasible.")

    best_solution = clone_solution(incumbent)
    best_cost = incumbent_cost
    best_found_iteration = 0

    total_steps = max(1, warmup_iterations + iterations)
    accepted_snapshots = []
    periodic_snapshots = []
    snapshot_accept_stride = max(1, int(snapshot_accept_stride))
    snapshot_iteration_stride = max(1, int(snapshot_iteration_stride))
    accepted_counter = 0

    deltas = []
    delta_points = {k: [] for k in op_names}
    acceptance_points = []
    accepted_objective_points = []

    def _rrt_deviation(best_obj, g, G):
        G = max(1, int(G))
        g = max(0, min(int(g), G))
        frac = ((G - g) / G) ** float(rrt_decay_exponent)
        return float(rrt_deviation_factor) * frac * max(1.0, float(best_obj))

    for w in range(warmup_iterations):
        configure_operator_search_progress(w / total_steps)
        op_name = random.choice(op_names)
        stats[op_name]["used"] += 1

        candidate = apply_main_operator(incumbent, op_name)
        if candidate is incumbent or candidate == incumbent:
            continue
        if not fast_precheck_solution(candidate, ctx):
            continue

        feasible, cand_cost = cached_evaluate(candidate)
        if not feasible:
            continue

        stats[op_name]["feasible"] += 1
        delta_e = cand_cost - incumbent_cost
        if collect_delta_points:
            delta_points[op_name].append((w + 1, float(delta_e)))
        if snapshot_every_iteration and ((w + 1) % snapshot_iteration_stride == 0):
            periodic_snapshots.append(
                _snapshot_record(
                    w + 1,
                    "warmup",
                    op_name,
                    cand_cost,
                    best_cost,
                    candidate,
                )
            )
        if delta_e >= 0:
            deltas.append(delta_e)

        if delta_e < 0:
            incumbent = candidate
            incumbent_cost = cand_cost
            stats[op_name]["accepted"] += 1
            stats[op_name]["improved"] += 1
            stats[op_name]["delta_sum"] += delta_e
            stats[op_name]["improve_delta_sum"] += -delta_e
            if collect_accepted_objective_points:
                accepted_objective_points.append((w + 1, float(incumbent_cost)))
            if snapshot_on_accepted:
                accepted_counter += 1
                if accepted_counter % snapshot_accept_stride == 0:
                    accepted_snapshots.append(
                        _snapshot_record(
                            w + 1,
                            "warmup",
                            op_name,
                            incumbent_cost,
                            best_cost,
                            incumbent,
                        )
                    )
            if incumbent_cost < best_cost:
                best_solution = clone_solution(incumbent)
                best_cost = incumbent_cost
                best_found_iteration = w + 1
        else:
            stats[op_name]["worse_feasible"] += 1
            rrt_dev_w = _rrt_deviation(best_cost, w + 1, warmup_iterations)
            can_accept_worse = cand_cost <= (best_cost + rrt_dev_w)
            if collect_acceptance_points:
                acceptance_points.append((w + 1, 1.0 if can_accept_worse else 0.0))
            if can_accept_worse:
                incumbent = candidate
                incumbent_cost = cand_cost
                stats[op_name]["accepted"] += 1
                stats[op_name]["uphill_accepted"] += 1
                stats[op_name]["delta_sum"] += delta_e
                stats[op_name]["uphill_accepted_delta_sum"] += delta_e
                stats[op_name]["p_accept_sum"] += 1.0
                stats[op_name]["p_accept_count"] += 1
                if collect_accepted_objective_points:
                    accepted_objective_points.append((w + 1, float(incumbent_cost)))
                if snapshot_on_accepted:
                    accepted_counter += 1
                    if accepted_counter % snapshot_accept_stride == 0:
                        accepted_snapshots.append(
                            _snapshot_record(
                                w + 1,
                                "warmup",
                                op_name,
                                incumbent_cost,
                                best_cost,
                                incumbent,
                            )
                        )
            else:
                stats[op_name]["uphill_rejected"] += 1
                stats[op_name]["p_accept_sum"] += 0.0
                stats[op_name]["p_accept_count"] += 1

    if deltas:
        sorted_deltas = sorted(deltas)
        q = max(0.0, min(1.0, float(warmup_delta_trim_quantile)))
        cut_idx = int(len(sorted_deltas) * q)
        cut_idx = max(1, min(len(sorted_deltas), cut_idx))
        trimmed = sorted_deltas[:cut_idx]
        delta_avg = sum(trimmed) / len(trimmed)
    else:
        delta_avg = 0.0
    t0 = _rrt_deviation(best_cost, 0, iterations)
    t0_used_fallback = False

    weights = {k: 1.0 / len(op_names) for k in op_names}
    segment_scores = {k: 0.0 for k in op_names}
    segment_uses = {k: 0 for k in op_names}
    segment_feasible_uses = {k: 0 for k in op_names}
    weight_history = []
    temperature_history = []
    if collect_temperature_history:
        temperature_history.append((warmup_iterations, float(t0)))

    escape_calls = 0
    escape_log = []  # (iter, incumbent_cost_before, incumbent_cost_after, esc_improved_best, best_cost_after)
    escape_feasible_steps = 0
    no_best_improve_steps = 0
    # Warm RRT reset: after each escape, give the new basin a full d₀ acceptance
    # window for this many iterations before resuming normal cooling.
    _WARM_RESET_WINDOW = 500
    escape_warm_reset_until = -1
    # op11 dynamic suppression: after 20% of iterations, if op11's cumulative
    # feasibility rate is below 8% cap its weight at 0.05 for the rest of the run.
    # op11 earns large but infrequent rewards that inflate its ALNS weight far
    # above what its per-call throughput justifies — suppression corrects this
    # without imposing any floors or ceilings on other operators.
    _op11_suppressed = False
    _op11_suppress_check_iter = max(1, int(iterations * 0.20))

    _time_limited = time_limit_seconds is not None and time_limit_seconds > 0
    _t_loop_start = time.perf_counter() if _time_limited else 0.0

    for it in range(iterations):
        if _time_limited:
            _elapsed = time.perf_counter() - _t_loop_start
            if _elapsed >= time_limit_seconds:
                break
            # Drive cooling from elapsed-vs-budget instead of iteration index
            _virt_total = max(1, int(time_limit_seconds))
            _virt_step = min(_virt_total, max(1, int(_elapsed)))
        if it <= escape_warm_reset_until:
            # Full initial acceptance window regardless of how far into the run we are.
            rrt_dev = float(rrt_deviation_factor) * max(1.0, float(best_cost))
        elif _time_limited:
            rrt_dev = _rrt_deviation(best_cost, _virt_step, _virt_total)
        else:
            rrt_dev = _rrt_deviation(best_cost, it + 1, iterations)
        if collect_temperature_history:
            temperature_history.append((warmup_iterations + it + 1, float(rrt_dev)))
        if _time_limited:
            configure_operator_search_progress(min(1.0, _elapsed / time_limit_seconds))
        else:
            configure_operator_search_progress((warmup_iterations + it) / total_steps)
        op_name = roulette_pick(weights, op_names)
        stats[op_name]["used"] += 1
        segment_uses[op_name] += 1

        candidate = apply_main_operator(incumbent, op_name)
        if candidate is incumbent or candidate == incumbent:
            continue
        if not fast_precheck_solution(candidate, ctx):
            continue

        feasible, cand_cost = cached_evaluate(candidate)
        if not feasible:
            continue

        stats[op_name]["feasible"] += 1
        segment_feasible_uses[op_name] += 1

        delta_e = cand_cost - incumbent_cost
        if collect_delta_points:
            delta_points[op_name].append((warmup_iterations + it + 1, float(delta_e)))
        if snapshot_every_iteration and ((warmup_iterations + it + 1) % snapshot_iteration_stride == 0):
            periodic_snapshots.append(
                _snapshot_record(
                    warmup_iterations + it + 1,
                    "main",
                    op_name,
                    cand_cost,
                    best_cost,
                    candidate,
                )
            )
        accepted = False
        improved_best = False

        if delta_e < 0:
            incumbent = candidate
            incumbent_cost = cand_cost
            accepted = True
            stats[op_name]["accepted"] += 1
            stats[op_name]["improved"] += 1
            stats[op_name]["delta_sum"] += delta_e
            stats[op_name]["improve_delta_sum"] += -delta_e

            if incumbent_cost < best_cost:
                best_solution = clone_solution(incumbent)
                best_cost = incumbent_cost
                improved_best = True
                best_found_iteration = warmup_iterations + it + 1
                if enable_local_search and ls_at_new_best:
                    ls_sol, ls_cost, _ = ls.local_search(
                        best_solution,
                        best_cost,
                        cached_evaluate,
                        ctx,
                        max_cycles=ls_max_cycles,
                        time_budget_seconds=ls_time_budget_seconds,
                    )
                    if ls_cost < best_cost - 1e-9:
                        best_solution = clone_solution(ls_sol)
                        best_cost = ls_cost
                        incumbent = clone_solution(ls_sol)
                        incumbent_cost = ls_cost
        else:
            stats[op_name]["worse_feasible"] += 1
            can_accept_worse = cand_cost <= (best_cost + rrt_dev)
            p_accept = 1.0 if can_accept_worse else 0.0
            stats[op_name]["p_accept_sum"] += p_accept
            stats[op_name]["p_accept_count"] += 1
            if collect_acceptance_points:
                acceptance_points.append((warmup_iterations + it + 1, float(p_accept)))
            if can_accept_worse:
                incumbent = candidate
                incumbent_cost = cand_cost
                accepted = True
                stats[op_name]["accepted"] += 1
                stats[op_name]["uphill_accepted"] += 1
                stats[op_name]["delta_sum"] += delta_e
                stats[op_name]["uphill_accepted_delta_sum"] += delta_e
            else:
                stats[op_name]["uphill_rejected"] += 1

        if accepted and snapshot_on_accepted:
            accepted_counter += 1
            if accepted_counter % snapshot_accept_stride == 0:
                accepted_snapshots.append(
                    _snapshot_record(
                        warmup_iterations + it + 1,
                        "main",
                        op_name,
                        incumbent_cost,
                        best_cost,
                        incumbent,
                    )
                )
        if accepted and collect_accepted_objective_points:
            accepted_objective_points.append((warmup_iterations + it + 1, float(incumbent_cost)))

        improve_mag = max(0.0, -delta_e)
        improve_threshold = max(1.0, float(reward_improve_threshold))
        meaningful_improve = improve_mag >= improve_threshold
        magnitude_bonus = min(3.0, improve_mag / (2.0 * improve_threshold))
        segment_reward = 0.0

        if improved_best:
            segment_reward = float(sigma_global_best) + magnitude_bonus
            no_best_improve_steps = 0
        elif delta_e < 0:
            if meaningful_improve:
                segment_reward = float(sigma_incumbent_improve) + 0.5 * magnitude_bonus
            else:
                # Keep some credit for small-but-consistent improving moves.
                small_gain = improve_mag / improve_threshold
                segment_reward = float(sigma_small_improve) * max(0.0, min(1.0, small_gain))
            no_best_improve_steps += 1
        elif accepted:
            # Controlled credit for accepted RRT diversification moves.
            rrt_window = max(1.0, float(rrt_dev))
            if 0.0 < delta_e <= rrt_window:
                uphill_quality = max(0.0, 1.0 - (delta_e / rrt_window))
                segment_reward = float(sigma_uphill_accepted) * uphill_quality
            no_best_improve_steps += 1
        else:
            no_best_improve_steps += 1

        if segment_reward > 0.0:
            segment_scores[op_name] += segment_reward

        if (it + 1) % segment_length == 0:
            updated = {}
            progress = (warmup_iterations + it + 1) / total_steps
            for k in op_names:
                theta = segment_uses[k]
                if theta > 0:
                    # Use total operator calls in the denominator so sparse-feasible operators
                    # are not over-rewarded by a few large successful moves.
                    effective_r = reaction_factor * (theta / (theta + 8.0))
                    updated[k] = weights[k] * (1.0 - effective_r) + effective_r * (
                        segment_scores[k] / theta
                    )
                else:
                    # Late-run scarcity safeguard: avoid killing an operator purely due no feasible hits.
                    if progress >= 0.7:
                        updated[k] = weights[k] * (1.0 - 0.35 * reaction_factor)
                    else:
                        updated[k] = weights[k] * (1.0 - reaction_factor)
            # No floors or ceilings — let ALNS weight operators freely.
            # Exception 1: op11 is suppressed if its feasibility is too low,
            # because rare large rewards inflate its ALNS weight well above
            # what its per-call throughput justifies.
            # Exception 2: op16 is capped at 0.20 to prevent it from dominating
            # on small instances.  Op16 finds the local optimum of drone
            # assignment in the very first few calls (100% improvement rate),
            # which gives it runaway weight and starves other operators that
            # need to explore different truck routes to escape the basin.
            _upper = {}
            _lower = {}
            _suppress_check_due = (
                _elapsed >= 0.20 * time_limit_seconds
                if _time_limited
                else (it + 1) >= _op11_suppress_check_iter
            )
            if (
                "op11" in op_names
                and not _op11_suppressed
                and _suppress_check_due
            ):
                op11_used = stats["op11"]["used"]
                op11_feasible = stats["op11"]["feasible"]
                if op11_used >= 50 and (op11_feasible / op11_used) < 0.08:
                    _op11_suppressed = True
            if _op11_suppressed:
                _upper["op11"] = 0.05
            if "op16" in op_names:
                # Cap at 0.25: op16 fires first at 25% of iterations, and after
                # that its 100% improvement rate would otherwise dominate the budget.
                _upper["op16"] = 0.25
            weights = _normalize_weight_dict_with_caps(
                updated,
                min_weight=0.03,
                lower_caps=_lower,
                upper_caps=_upper,
            )
            weight_history.append((it + 1, dict(weights)))
            segment_scores = {k: 0.0 for k in op_names}
            segment_uses = {k: 0 for k in op_names}
            segment_feasible_uses = {k: 0 for k in op_names}

        if no_best_improve_steps >= escape_stall_limit:
            escape_calls += 1
            _esc_inc_before = incumbent_cost
            incumbent, incumbent_cost, esc_improved_best, esc_steps = _escape_with_related_large(
                incumbent,
                incumbent_cost,
                best_solution,
                best_cost,
                ctx,
                cached_evaluate,
            )
            escape_feasible_steps += esc_steps
            if esc_improved_best and incumbent_cost < best_cost:
                best_solution = clone_solution(incumbent)
                best_cost = incumbent_cost
                best_found_iteration = warmup_iterations + it + 1
            escape_log.append((
                warmup_iterations + it + 1,
                float(_esc_inc_before),
                float(incumbent_cost),
                bool(esc_improved_best),
                float(best_cost),
            ))
            # Always reset stall counter so ALNS has a full window to explore
            # from the escaped position before triggering again.
            no_best_improve_steps = 0
            # Warm RRT reset: restore full d₀ acceptance for the next window
            # so the new basin is explored properly regardless of run progress.
            escape_warm_reset_until = it + _WARM_RESET_WINDOW

    best_parts = to_parts_solution(best_solution)
    assert checker.is_solution_feasible(best_parts)
    assert calc.calculate_total_waiting_time(best_parts)[3]

    phase_weights = _phase_weight_summary(weight_history, op_names, iterations, weights)

    stats["_meta"] = {
        "warmup_delta_avg": delta_avg,
        "warmup_delta_samples": len(deltas),
        "t0": t0,
        "t0_used_fallback": t0_used_fallback,
        "final_weights": dict(weights),
        "phase_weights": phase_weights,
        "weight_history": weight_history,
        "escape_calls": escape_calls,
        "escape_feasible_steps": escape_feasible_steps,
        "escape_log": escape_log,
        "accepted_snapshots": accepted_snapshots,
        "periodic_snapshots": periodic_snapshots,
        "best_found_iteration": best_found_iteration,
        "delta_points": delta_points,
        "temperature_history": temperature_history,
        "acceptance_points": acceptance_points,
        "accepted_objective_points": accepted_objective_points,
    }

    if enable_local_search and ls_at_end:
        ils_deadline = time.perf_counter() + ls_end_time_budget_seconds
        per_pass_budget = ls_end_time_budget_seconds / max(1, ls_ils_passes)

        ls_sol, ls_cost, _ = ls.local_search(
            best_solution,
            best_cost,
            cached_evaluate,
            ctx,
            max_cycles=ls_max_cycles,
            time_budget_seconds=per_pass_budget,
        )
        if ls_cost < best_cost - 1e-9:
            best_solution = clone_solution(ls_sol)
            best_cost = ls_cost

        for _pass in range(max(0, ls_ils_passes - 1)):
            if time.perf_counter() >= ils_deadline:
                break
            perturbed = op14.operator(best_solution)
            feasible_p, p_cost = cached_evaluate(perturbed)
            if not feasible_p:
                continue
            remaining = ils_deadline - time.perf_counter()
            if remaining <= 0.5:
                break
            ls_sol, ls_cost, _ = ls.local_search(
                perturbed,
                p_cost,
                cached_evaluate,
                ctx,
                max_cycles=ls_max_cycles,
                time_budget_seconds=min(per_pass_budget, remaining),
            )
            if ls_cost < best_cost - 1e-9:
                best_solution = clone_solution(ls_sol)
                best_cost = ls_cost

    return best_solution, best_cost, stats


def run_statistics(
    initial_solution,
    instance_data=None,
    runs=10,
    warmup_iterations=500,
    iterations=9500,
    final_temperature=0.1,
    cache_limit=200000,
    reaction_factor=0.15,
    segment_length=100,
    escape_stall_limit=650,
    verbose=True,
    return_metrics=False,
    print_solution_pipe=False,
    snapshot_on_accepted=False,
    snapshot_output_file=None,
    snapshot_accept_stride=1,
    snapshot_every_iteration=False,
    snapshot_iteration_stride=25,
    reward_improve_threshold=50.0,
    uphill_reward_cap=30.0,
    sigma_global_best=8.0,
    sigma_incumbent_improve=4.0,
    sigma_small_improve=1.0,
    sigma_uphill_accepted=0.6,
    warmup_delta_trim_quantile=0.9,
    plot_delta_scatter_best_run=False,
    delta_plot_output_dir=None,
    plot_weights_best_run=False,
    weight_plot_output_dir=None,
    plot_temperature_best_run=False,
    temperature_plot_output_dir=None,
    plot_acceptance_probability_best_run=False,
    acceptance_probability_output_dir=None,
    plot_accepted_objective_best_run=True,
    accepted_objective_output_dir=None,
    temperature_threshold=1.0,
    save_best_runs_log=False,
    best_runs_log_file=None,
    solution_factory=None,
    save_best_solution_plot=False,
    best_solution_plot_output_dir=None,
    time_limit_seconds=None,
    op_names_override=None,
    rrt_decay_exponent=1.0,
    enable_local_search=False,
    ls_time_budget_seconds=5.0,
    ls_max_cycles=20,
    ls_at_new_best=True,
    ls_at_end=True,
    ls_end_time_budget_seconds=10.0,
    ls_ils_passes=1,
):
    if instance_data is None:
        instance_data = load_instance()

    ctx, calc, checker = build_evaluator(instance_data)
    configure_operator_context(instance_data)

    n_cust = ctx.get("n_customers", 100)
    if op_names_override is not None:
        op_names = list(op_names_override)
    elif n_cust <= 15:
        op_names = ["op1", "op2", "op3", "op8", "op10", "op11", "op12", "op13", "op17"]
    elif 16 <= n_cust <= 30:
        op_names = ["op1", "op2", "op3", "op10", "op12", "op13", "op17"]
    else:
        op_names = ["op1", "op2", "op3", "op12", "op13", "op15"]

    if not enable_local_search:
        enable_local_search = True
        ls_at_new_best = False
        ls_at_end = True
        if n_cust > 30:
            if ls_end_time_budget_seconds < 40.0:
                ls_end_time_budget_seconds = 40.0
            if ls_max_cycles < 30:
                ls_max_cycles = 30
        else:
            if ls_end_time_budget_seconds < 10.0:
                ls_end_time_budget_seconds = 10.0
            if ls_max_cycles < 20:
                ls_max_cycles = 20

    _check_sol = solution_factory() if solution_factory is not None else initial_solution
    init_feasible, init_cost = evaluate_solution(_check_sol, calc, checker)
    if not init_feasible:
        raise ValueError("Initial solution is not feasible; cannot compute improvement statistics.")

    run_costs = []
    run_times = []
    global_best_solution = None
    global_best_cost = float("inf")
    shared_eval_cache = OrderedDict()

    aggregate_stats = _init_stats(op_names)

    warmup_delta_avgs = []
    warmup_delta_samples = []
    t0_values = []
    t0_fallback_count = 0
    final_weights_acc = {k: 0.0 for k in op_names}
    phase_weights_acc = {
        "early": {k: 0.0 for k in op_names},
        "mid": {k: 0.0 for k in op_names},
        "late": {k: 0.0 for k in op_names},
    }
    total_escape_calls = 0
    total_escape_steps = 0
    best_found_iterations = []
    best_run_delta_points = None
    best_run_weight_history = None
    best_run_temperature_history = None
    best_run_acceptance_points = None
    best_run_accepted_objective_points = None
    best_run_escape_log = None
    best_run_index = None
    threshold_cross_iterations = []
    run_best_records = []

    if "op11" in op_names:
        op11.reset_diagnostics()

    for run_id in range(runs):
        start = time.perf_counter()
        run_start_solution = solution_factory() if solution_factory is not None else clone_solution(initial_solution)
        best_solution, best_cost, op_stats = alns_improved(
            run_start_solution,
            instance_data=instance_data,
            warmup_iterations=warmup_iterations,
            iterations=iterations,
            final_temperature=final_temperature,
            cache_limit=cache_limit,
            reaction_factor=reaction_factor,
            segment_length=segment_length,
            escape_stall_limit=escape_stall_limit,
            ctx=ctx,
            calc=calc,
            checker=checker,
            shared_eval_cache=shared_eval_cache,
            snapshot_on_accepted=snapshot_on_accepted,
            snapshot_accept_stride=snapshot_accept_stride,
            snapshot_every_iteration=snapshot_every_iteration,
            snapshot_iteration_stride=snapshot_iteration_stride,
            reward_improve_threshold=reward_improve_threshold,
            uphill_reward_cap=uphill_reward_cap,
            sigma_global_best=sigma_global_best,
            sigma_incumbent_improve=sigma_incumbent_improve,
            sigma_small_improve=sigma_small_improve,
            sigma_uphill_accepted=sigma_uphill_accepted,
            warmup_delta_trim_quantile=warmup_delta_trim_quantile,
            collect_delta_points=plot_delta_scatter_best_run,
            collect_temperature_history=plot_temperature_best_run,
            collect_acceptance_points=plot_acceptance_probability_best_run,
            collect_accepted_objective_points=plot_accepted_objective_best_run,
            time_limit_seconds=time_limit_seconds,
            op_names_override=op_names_override,
            rrt_decay_exponent=rrt_decay_exponent,
            enable_local_search=enable_local_search,
            ls_time_budget_seconds=ls_time_budget_seconds,
            ls_max_cycles=ls_max_cycles,
            ls_at_new_best=ls_at_new_best,
            ls_at_end=ls_at_end,
            ls_end_time_budget_seconds=ls_end_time_budget_seconds,
            ls_ils_passes=ls_ils_passes,
        )
        elapsed = time.perf_counter() - start

        for op_name in op_names:
            for k in aggregate_stats[op_name]:
                aggregate_stats[op_name][k] += op_stats[op_name][k]

        run_meta = op_stats.get("_meta")
        if run_meta is not None:
            warmup_delta_avgs.append(run_meta.get("warmup_delta_avg", 0.0))
            warmup_delta_samples.append(run_meta.get("warmup_delta_samples", 0))
            t0_values.append(run_meta.get("t0", 0.0))
            if run_meta.get("t0_used_fallback", False):
                t0_fallback_count += 1
            fw = run_meta.get("final_weights", {})
            for k in op_names:
                final_weights_acc[k] += float(fw.get(k, 0.0))
            pw = run_meta.get("phase_weights", {})
            for phase in ("early", "mid", "late"):
                phase_map = pw.get(phase, {})
                for k in op_names:
                    phase_weights_acc[phase][k] += float(phase_map.get(k, 0.0))
            total_escape_calls += int(run_meta.get("escape_calls", 0))
            total_escape_steps += int(run_meta.get("escape_feasible_steps", 0))
            best_found_iterations.append(int(run_meta.get("best_found_iteration", 0)))
            temp_hist = run_meta.get("temperature_history", [])
            cross_iter = 0
            thr = float(temperature_threshold)
            for iter_idx, temp_val in temp_hist:
                if float(temp_val) <= thr:
                    cross_iter = int(iter_idx)
                    break
            threshold_cross_iterations.append(cross_iter)
            if (snapshot_on_accepted or snapshot_every_iteration) and snapshot_output_file and run_id == 0:
                chosen_snapshots = (
                    run_meta.get("periodic_snapshots", [])
                    if snapshot_every_iteration
                    else run_meta.get("accepted_snapshots", [])
                )
                payload = {
                    "instance": file_name,
                    "run_id": 1,
                    "warmup_iterations": warmup_iterations,
                    "iterations": iterations,
                    "initial_score": init_cost,
                    "snapshot_accept_stride": int(snapshot_accept_stride),
                    "snapshot_iteration_stride": int(snapshot_iteration_stride),
                    "snapshots": chosen_snapshots,
                }
                out_path = Path(snapshot_output_file)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f)

        run_costs.append(best_cost)
        run_times.append(elapsed)
        best_iter = best_found_iterations[-1] if best_found_iterations else 0
        run_best_records.append(
            {
                "run_id": int(run_id + 1),
                "best_cost": float(best_cost),
                "best_iter": int(best_iter),
                "runtime": float(elapsed),
            }
        )

        if verbose:
            print(
                f"Best score in run {run_id + 1}/{runs}: {best_cost} "
                f"(best found at iteration i*={best_iter})"
            )

        if best_cost < global_best_cost:
            global_best_cost = best_cost
            global_best_solution = clone_solution(best_solution)
            best_run_meta = op_stats.get("_meta") or {}
            best_run_delta_points = best_run_meta.get("delta_points", None)
            best_run_weight_history = best_run_meta.get("weight_history", None)
            best_run_temperature_history = best_run_meta.get("temperature_history", None)
            best_run_acceptance_points = best_run_meta.get("acceptance_points", None)
            best_run_accepted_objective_points = best_run_meta.get("accepted_objective_points", None)
            best_run_escape_log = best_run_meta.get("escape_log", None)
            best_run_index = run_id + 1

    avg_obj = sum(run_costs) / len(run_costs)
    best_obj = global_best_cost
    improvement_abs = init_cost - best_obj
    improvement_pct = (improvement_abs / init_cost * 100.0) if init_cost > 0 else 0.0
    avg_runtime_per_run = sum(run_times) / len(run_times)

    avg_final_weights = {k: (final_weights_acc[k] / runs if runs > 0 else 0.0) for k in op_names}
    avg_phase_weights = {
        phase: {k: (phase_weights_acc[phase][k] / runs if runs > 0 else 0.0) for k in op_names}
        for phase in ("early", "mid", "late")
    }

    metrics = {
        "initial_score": init_cost,
        "average_score": avg_obj,
        "best_score": best_obj,
        "improvement_abs": improvement_abs,
        "improvement_pct": improvement_pct,
        "average_runtime": avg_runtime_per_run,
        "warmup_delta_avg_mean": (
            sum(warmup_delta_avgs) / len(warmup_delta_avgs) if warmup_delta_avgs else 0.0
        ),
        "warmup_delta_samples_mean": (
            sum(warmup_delta_samples) / len(warmup_delta_samples) if warmup_delta_samples else 0.0
        ),
        "t0_mean": (sum(t0_values) / len(t0_values) if t0_values else 0.0),
        "t0_fallback_runs": t0_fallback_count,
        "avg_final_weights": avg_final_weights,
        "avg_phase_weights": avg_phase_weights,
        "escape_calls": total_escape_calls,
        "escape_steps": total_escape_steps,
        "best_found_iterations": best_found_iterations,
        "temperature_threshold": float(temperature_threshold),
        "temperature_threshold_cross_iterations": threshold_cross_iterations,
        "temperature_threshold_cross_mean": (
            sum(x for x in threshold_cross_iterations if x > 0)
            / max(1, len([x for x in threshold_cross_iterations if x > 0]))
        ) if threshold_cross_iterations else 0.0,
        "accepted_objective_points": best_run_accepted_objective_points,
        "aggregate_stats": aggregate_stats,
        "escape_log": best_run_escape_log,
    }

    delta_plot_files = []
    if plot_delta_scatter_best_run and best_run_delta_points is not None:
        plot_dir = (
            delta_plot_output_dir
            if delta_plot_output_dir is not None
            else str(ASSIGNMENT_DIR / "outputs" / "delta_plots")
        )
        delta_plot_files = _plot_operator_deltas(
            best_run_delta_points,
            op_names,
            plot_dir,
            file_name,
            f"run{best_run_index}",
        )
    metrics["delta_plot_files"] = delta_plot_files

    weight_plot_file = ""
    if plot_weights_best_run and best_run_weight_history is not None:
        plot_dir = (
            weight_plot_output_dir
            if weight_plot_output_dir is not None
            else str(ASSIGNMENT_DIR / "outputs" / "weight_plots")
        )
        weight_plot_file = _plot_operator_weights(
            best_run_weight_history,
            op_names,
            plot_dir,
            file_name,
            f"run{best_run_index}",
        )
    metrics["weight_plot_file"] = weight_plot_file

    temperature_plot_file = ""
    if plot_temperature_best_run and best_run_temperature_history is not None:
        plot_dir = (
            temperature_plot_output_dir
            if temperature_plot_output_dir is not None
            else str(ASSIGNMENT_DIR / "outputs" / "temperature_plots")
        )
        temperature_plot_file = _plot_temperature_history(
            best_run_temperature_history,
            plot_dir,
            file_name,
            f"run{best_run_index}",
        )
    metrics["temperature_plot_file"] = temperature_plot_file

    acceptance_probability_plot_file = ""
    if plot_acceptance_probability_best_run and best_run_acceptance_points is not None:
        plot_dir = (
            acceptance_probability_output_dir
            if acceptance_probability_output_dir is not None
            else str(ASSIGNMENT_DIR / "outputs" / "acceptance_probability_plots")
        )
        acceptance_probability_plot_file = _plot_acceptance_probability(
            best_run_acceptance_points,
            plot_dir,
            file_name,
            f"run{best_run_index}",
        )
    metrics["acceptance_probability_plot_file"] = acceptance_probability_plot_file

    accepted_objective_plot_file = ""
    if plot_accepted_objective_best_run and best_run_accepted_objective_points is not None:
        plot_dir = (
            accepted_objective_output_dir
            if accepted_objective_output_dir is not None
            else str(ASSIGNMENT_DIR / "outputs" / "accepted_objective_plots")
        )
        accepted_objective_plot_file = _plot_accepted_objective(
            best_run_accepted_objective_points,
            plot_dir,
            file_name,
            f"run{best_run_index}",
        )
    metrics["accepted_objective_plot_file"] = accepted_objective_plot_file

    best_runs_log_path = ""
    if save_best_runs_log and global_best_solution is not None and run_best_records:
        out_file = (
            best_runs_log_file
            if best_runs_log_file is not None
            else str(ASSIGNMENT_DIR / "outputs" / "best_runs_history.txt")
        )
        best_runs_log_path = _append_best_runs_log(
            out_file,
            file_name,
            run_best_records,
            avg_obj,
            best_obj,
            avg_runtime_per_run,
            format_solution_pipe(global_best_solution),
        )
    metrics["best_runs_log_file"] = best_runs_log_path

    best_solution_plot_file = ""
    if save_best_solution_plot and global_best_solution is not None:
        plot_dir = (
            best_solution_plot_output_dir
            if best_solution_plot_output_dir is not None
            else str(ASSIGNMENT_DIR / "outputs" / "best_visualised")
        )
        best_solution_plot_file = _plot_best_solution(
            global_best_solution,
            instance_data,
            plot_dir,
            file_name,
            global_best_cost,
        )
    metrics["best_solution_plot_file"] = best_solution_plot_file

    if verbose:
        print(f"Average score: {avg_obj}")
        print(f"Best score: {best_obj}")
        print(f"Average runtime: {avg_runtime_per_run:.4f} seconds")
        print(
            f"Improvement over initial solution: {improvement_abs} "
            f"({improvement_pct:.2f}% reduction)"
        )
        if t0_values:
            print(
                "RRT acceptance diagnostics: "
                f"warmup_delta_avg_mean={sum(warmup_delta_avgs)/len(warmup_delta_avgs):.4f}, "
                f"warmup_samples_mean={sum(warmup_delta_samples)/len(warmup_delta_samples):.1f}, "
                f"d0_mean={sum(t0_values)/len(t0_values):.4f}, "
                f"d0_min={min(t0_values):.4f}, d0_max={max(t0_values):.4f}, "
                f"d0_fallback_runs={t0_fallback_count}/{runs}"
            )
            crossed = [x for x in threshold_cross_iterations if x > 0]
            if crossed:
                print(
                    f"Deviation threshold diagnostics (D<={float(temperature_threshold):.4f}): "
                    f"cross_mean={sum(crossed)/len(crossed):.1f}, "
                    "cross_iters=" + ", ".join(str(x) for x in threshold_cross_iterations)
                )
            else:
                print(
                    f"Deviation threshold diagnostics (D<={float(temperature_threshold):.4f}): "
                    f"not crossed in {runs}/{runs} runs"
                )

        final_w_str = ", ".join([f"{k}={avg_final_weights[k]:.3f}" for k in op_names])
        early_w_str = ", ".join([f"{k}={avg_phase_weights['early'][k]:.3f}" for k in op_names])
        mid_w_str = ", ".join([f"{k}={avg_phase_weights['mid'][k]:.3f}" for k in op_names])
        late_w_str = ", ".join([f"{k}={avg_phase_weights['late'][k]:.3f}" for k in op_names])
        print(
            "ALNS adaptation diagnostics: "
            f"avg_final_weights=({final_w_str}), "
            f"escape_calls={total_escape_calls}, escape_feasible_steps={total_escape_steps}"
        )
        if best_found_iterations:
            print(
                "Best solution iterations (i* per run): "
                + ", ".join(str(x) for x in best_found_iterations)
            )
        if delta_plot_files:
            print("Delta plots saved:")
            for fp in delta_plot_files:
                print(f"  {fp}")
        if weight_plot_file:
            print("Operator weights plot saved:")
            print(f"  {weight_plot_file}")
        if temperature_plot_file:
            print("RRT deviation plot saved:")
            print(f"  {temperature_plot_file}")
        if acceptance_probability_plot_file:
            print("RRT acceptance indicator plot saved:")
            print(f"  {acceptance_probability_plot_file}")
        if accepted_objective_plot_file:
            print("Accepted objective plot saved:")
            print(f"  {accepted_objective_plot_file}")
        if best_runs_log_path:
            print("Best-run history appended to:")
            print(f"  {best_runs_log_path}")
        if best_solution_plot_file:
            print("Best solution visualisation saved:")
            print(f"  {best_solution_plot_file}")
        print(
            "ALNS phase weights: "
            f"early({early_w_str}), mid({mid_w_str}), late({late_w_str})"
        )

        if print_solution_pipe and global_best_solution is not None:
            print("Solution on pipe format:")
            print(format_solution_pipe(global_best_solution))

        print("Operator contribution stats (aggregated):")
        for op_name in op_names:
            s = aggregate_stats[op_name]
            used = s["used"]
            feasible_moves = s["feasible"]
            accepted = s["accepted"]
            improved = s["improved"]
            uphill_accepted = s["uphill_accepted"]
            uphill_rejected = s["uphill_rejected"]
            worse_feasible = s["worse_feasible"]
            avg_delta = s["delta_sum"] / accepted if accepted > 0 else float("nan")
            feasible_rate = (feasible_moves / used * 100.0) if used > 0 else 0.0
            accept_rate = (accepted / feasible_moves * 100.0) if feasible_moves > 0 else 0.0
            improve_rate = (improved / accepted * 100.0) if accepted > 0 else 0.0
            uphill_rate = (uphill_accepted / accepted * 100.0) if accepted > 0 else 0.0
            uphill_of_worse_rate = (
                uphill_accepted / worse_feasible * 100.0 if worse_feasible > 0 else 0.0
            )
            avg_uphill_accepted_delta = (
                s["uphill_accepted_delta_sum"] / uphill_accepted if uphill_accepted > 0 else float("nan")
            )
            avg_improve_accepted_delta = (
                s["improve_delta_sum"] / improved if improved > 0 else float("nan")
            )
            mean_p_accept = s["p_accept_sum"] / s["p_accept_count"] if s["p_accept_count"] > 0 else float("nan")
            improve_per_1k_uses = (s["improve_delta_sum"] * 1000.0 / used) if used > 0 else 0.0

            print(
                f"  {op_name}: used={used}, feasible={feasible_moves} ({feasible_rate:.1f}%), "
                f"accepted={accepted} ({accept_rate:.1f}%), improved={improved} ({improve_rate:.1f}%), "
                f"uphill_accepted={uphill_accepted} ({uphill_rate:.1f}%), "
                f"worse_feasible={worse_feasible}, uphill_accept_of_worse={uphill_of_worse_rate:.1f}%, "
                f"uphill_rejected={uphill_rejected}, mean_p_accept={mean_p_accept:.4f}, "
                f"avg_uphill_accepted_delta={avg_uphill_accepted_delta:.4f}, "
                f"avg_improve_accepted_delta={avg_improve_accepted_delta:.4f}, "
                f"avg_accepted_delta={avg_delta:.4f}, improve_per_1000_uses={improve_per_1k_uses:.2f}"
            )

        if "op11" in op_names:
            d = op11.get_diagnostics()
            calls = d["calls"]
            noop_pct = 100.0 * d["returned_same"] / calls if calls > 0 else 0.0
            opt_pct  = 100.0 * d["ortools_optimal"] / calls if calls > 0 else 0.0
            tmo_pct  = 100.0 * d["ortools_timeout"] / calls if calls > 0 else 0.0
            fb_pct   = 100.0 * d["ortools_fallback"] / calls if calls > 0 else 0.0
            print(
                f"op11 TSP diagnostics: calls={calls}, "
                f"different_route={d['tsp_improved']} ({100-noop_pct:.1f}%), "
                f"same_route_noop={d['returned_same']} ({noop_pct:.1f}%)"
            )
            print(
                f"  OR-Tools proven_optimal={d['ortools_optimal']} ({opt_pct:.1f}%), "
                f"time_limit_hit={d['ortools_timeout']} ({tmo_pct:.1f}%), "
                f"fell_back_to_2opt={d['ortools_fallback']} ({fb_pct:.1f}%)"
            )
            if d["avg_truck_cost_improvement"] > 0:
                print(
                    f"  truck cost improvement when different route found: "
                    f"avg={d['avg_truck_cost_improvement']:.2f}, "
                    f"max={d['max_truck_cost_improvement']:.2f}"
                )

    if return_metrics:
        return global_best_solution, global_best_cost, metrics
    return global_best_solution, global_best_cost


def _run_params(n_customers):
    """Return (warmup, iters, escape_stall_limit, segment_length) for a given instance size."""
    if n_customers <= 15:
        return 500, 19500, max(150, 4 * n_customers), max(15, n_customers)
    elif n_customers <= 30:
        return 500, 25000, max(200, 6 * n_customers), max(20, n_customers)
    elif n_customers <= 60:
        return 500, 50000, 1000, max(20, n_customers)
    else:
        return 500, 50000, 1000, max(20, n_customers)


def main():
    """
    Run the algorithm once on every dataset in TEST_FILES_DIR and save all
    outputs (plots, weight history, best-solution visualisation) to
    Final_Assignment/outputs/.
    """
    global file_name

    all_txt = sorted(TEST_FILES_DIR.glob("*.txt"))
    # Keep only recognisable instance files; skip saved-solution or other stray files.
    dataset_files = [
        f for f in all_txt
        if f.name.startswith(("R_", "F_", "Truck_Drone_"))
    ]
    if not dataset_files:
        print(f"No dataset files found in {TEST_FILES_DIR}")
        return

    print(f"Found {len(dataset_files)} dataset(s): {[f.name for f in dataset_files]}")

    for ds_path in dataset_files:
        file_name = ds_path.name
        print(f"\n{'='*60}")
        print(f"Dataset: {file_name}")
        print(f"{'='*60}")

        instance_data = load_instance(ds_path)
        n_customers = instance_data["n_customers"]
        warmup, iters, escape_stall_limit, segment_length = _run_params(n_customers)

        run_statistics(
            None,
            instance_data=instance_data,
            solution_factory=lambda id=instance_data: build_initial_solution(id),
            runs=1,
            warmup_iterations=warmup,
            iterations=iters,
            final_temperature=0.1,
            cache_limit=200000,
            reaction_factor=0.15,
            segment_length=segment_length,
            escape_stall_limit=escape_stall_limit,
            verbose=True,
            return_metrics=False,
            print_solution_pipe=False,
            plot_delta_scatter_best_run=True,
            plot_weights_best_run=True,
            plot_temperature_best_run=True,
            plot_acceptance_probability_best_run=True,
            plot_accepted_objective_best_run=True,
            save_best_runs_log=True,
            save_best_solution_plot=True,
        )


if __name__ == "__main__":
    main()
