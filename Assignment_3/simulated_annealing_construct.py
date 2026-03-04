from read_file import read_instance
from pathlib import Path
import copy
import sys
import numpy as np
import random


GENERAL_ALGORITHM_PATH = Path(__file__).resolve().parents[1] / "General_algorithm"
if str(GENERAL_ALGORITHM_PATH) not in sys.path:
    sys.path.append(str(GENERAL_ALGORITHM_PATH))

import simulated_annealing as sa
import op_construct as op_construct
from CalCulateTotalArrivalTime import CalCulateTotalArrivalTime
from FeasibiltyCheck import SolutionFeasibility

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


absolute_path = "/Users/niklasmarton/Library/CloudStorage/OneDrive-Personlig/ITØK/Metaheuristics/TrucksAndDrones/Test_files/"
file_name = "R_100.txt"


def _configure_sa_instance(instance):
    # Override SA's operator module for this runner.
    sa.operator = op_construct.operator
    sa.set_operator_context = op_construct.set_operator_context

    sa.T = instance["truck_times"]
    sa.D = instance["drone_times"]
    sa.flight_limit = instance["flight_limit"]
    sa.n_customers = instance["n_customers"]
    sa.depot = instance.get("depot_index", 0)
    sa.set_operator_context(sa.T, sa.D, sa.flight_limit, sa.depot)


def _build_tuple_solution_from_construction(instance):
    """
    Construct initial solution with:
    1) greedy nearest-neighbor truck route
    2) 50% chance to assign candidate customer to a random drone (1 or 2),
       using launch/landing logic based on adjacent truck nodes.
    Returns tuple format: [truck, drone1_tuples, drone2_tuples]
    """
    T = instance["truck_times"]
    D = instance["drone_times"]
    flight_limit = instance["flight_limit"]
    n_nodes = instance["n_customers"] + 1
    depot_idx = instance.get("depot_index", 0)

    # Greedy truck route construction.
    unvisited = set(range(1, n_nodes))
    truck = [depot_idx]
    current = depot_idx
    while unvisited:
        nxt = min(unvisited, key=lambda node: T[current][node])
        truck.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    truck.append(depot_idx)

    # Drone assignment records as node-values: (customer, launch_node, land_node)
    d1_records = []
    d2_records = []
    busy_until = {1: None, 2: None}
    protected_nodes = set()

    i = 1
    while i < len(truck) - 1:
        prev_node = truck[i - 1]
        curr_node = truck[i]
        next_node = truck[i + 1]

        # Drone becomes available once truck reaches its landing node.
        if busy_until[1] == curr_node:
            busy_until[1] = None
        if busy_until[2] == curr_node:
            busy_until[2] = None

        # Do not remove protected landing nodes from truck route.
        if curr_node in protected_nodes:
            protected_nodes.remove(curr_node)
            i += 1
            continue

        # Try assigning this customer to a random drone with 50% probability.
        if random.random() < 0.5:
            chosen_drone = random.choice([1, 2])
            if busy_until[chosen_drone] is None:
                trip_time = D[prev_node][curr_node] + D[curr_node][next_node]
                if trip_time <= flight_limit:
                    if chosen_drone == 1:
                        d1_records.append((curr_node, prev_node, next_node))
                    else:
                        d2_records.append((curr_node, prev_node, next_node))
                    busy_until[chosen_drone] = next_node
                    protected_nodes.add(next_node)
                    truck.pop(i)
                    continue
        i += 1

    # Convert launch/landing nodes to indices in final truck route.
    def node_to_index(node, as_landing=False):
        if node == depot_idx:
            return len(truck) - 1 if as_landing else 0
        return truck.index(node)

    def records_to_tuples(records):
        tuples = []
        for customer, launch_node, land_node in records:
            launch_idx = node_to_index(launch_node, as_landing=False)
            land_idx = node_to_index(land_node, as_landing=True)
            if launch_idx < land_idx:
                tuples.append((customer, launch_idx, land_idx))
        tuples.sort(key=lambda x: (x[1], x[2], x[0]))
        return tuples

    drone1 = records_to_tuples(d1_records)
    drone2 = records_to_tuples(d2_records)
    return [truck, drone1, drone2]


def _is_feasible(solution, instance):
    calc = CalCulateTotalArrivalTime()
    calc.truck_times = instance["truck_times"]
    calc.drone_times = instance["drone_times"]
    calc.flight_range = instance["flight_limit"]
    calc.depot_index = instance.get("depot_index", 0)

    checker = SolutionFeasibility(
        n_nodes=instance["n_customers"] + 1,
        n_drones=2,
        depot_index=instance.get("depot_index", 0),
        drone_times=instance["drone_times"],
        flight_range=instance["flight_limit"],
    )

    feasible, _ = sa.evaluate_solution(solution, calc, checker)
    return feasible


def _embed_2d_from_distance_matrix(distance_matrix):
    """
    Classical MDS embedding from a distance matrix.
    Returns Nx2 coordinates for plotting.
    """
    d = np.array(distance_matrix, dtype=float)
    n = d.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (d ** 2) @ j
    eigvals, eigvecs = np.linalg.eigh(b)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Use top-2 non-negative components.
    vals = np.maximum(eigvals[:2], 0.0)
    vecs = eigvecs[:, :2]
    coords = vecs * np.sqrt(vals)
    if coords.shape[1] < 2:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 2 - coords.shape[1]))])
    return coords


def plot_solution(solution, instance, title="Best SA Solution"):
    """
    Plot truck route and drone sorties in a separate matplotlib window.
    """
    if plt is None:
        print("Matplotlib not available; skipping plot.")
        return

    truck, drone1, drone2 = solution
    coords = _embed_2d_from_distance_matrix(instance["truck_times"])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Plot nodes
    ax.scatter(coords[:, 0], coords[:, 1], s=30, c="black", alpha=0.7)
    depot = instance.get("depot_index", 0)
    ax.scatter([coords[depot, 0]], [coords[depot, 1]], s=120, c="red", marker="s", label="Depot")

    # Truck path
    for i in range(len(truck) - 1):
        a = truck[i]
        b = truck[i + 1]
        ax.plot(
            [coords[a, 0], coords[b, 0]],
            [coords[a, 1], coords[b, 1]],
            color="tab:blue",
            linewidth=1.8,
            alpha=0.9,
        )

    # Drone paths (launch -> customer -> land)
    for node, launch_idx, land_idx in drone1:
        launch_node = truck[launch_idx]
        land_node = truck[land_idx]
        ax.plot(
            [coords[launch_node, 0], coords[node, 0], coords[land_node, 0]],
            [coords[launch_node, 1], coords[node, 1], coords[land_node, 1]],
            color="tab:orange",
            linestyle="--",
            linewidth=1.4,
            alpha=0.8,
        )
    for node, launch_idx, land_idx in drone2:
        launch_node = truck[launch_idx]
        land_node = truck[land_idx]
        ax.plot(
            [coords[launch_node, 0], coords[node, 0], coords[land_node, 0]],
            [coords[launch_node, 1], coords[node, 1], coords[land_node, 1]],
            color="tab:green",
            linestyle="--",
            linewidth=1.4,
            alpha=0.8,
        )

    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def _solution_variation(prev_solution, new_solution):
    """
    Return variation score in [0,1] between two tuple-format solutions.
    """
    if prev_solution is None or new_solution is None:
        return 1.0

    prev_truck, prev_d1, prev_d2 = prev_solution
    new_truck, new_d1, new_d2 = new_solution

    # Truck sequence change ratio (position-wise, over overlap length).
    m = min(len(prev_truck), len(new_truck))
    if m == 0:
        truck_var = 1.0
    else:
        same_pos = sum(1 for i in range(m) if prev_truck[i] == new_truck[i])
        truck_var = 1.0 - (same_pos / m)

    # Drone-customer set change ratio (Jaccard distance).
    prev_nodes = set([x[0] for x in prev_d1] + [x[0] for x in prev_d2])
    new_nodes = set([x[0] for x in new_d1] + [x[0] for x in new_d2])
    union = prev_nodes | new_nodes
    if not union:
        drone_var = 0.0
    else:
        drone_var = 1.0 - (len(prev_nodes & new_nodes) / len(union))

    # Weighted blend: truck structure dominates objective heavily.
    return 0.7 * truck_var + 0.3 * drone_var


def run_random_restart_experiments(
    instance,
    runs=100,
    iterations=10000,
    final_temperature=0.1,
    stagnation_limit=10,
):
    """
    Run SA multiple times with random-restart behavior:
    - total `runs` SA runs
    - print only when a new global best is found
    - if no improvement for `stagnation_limit` consecutive runs, restart from
      a fresh construction solution
    """
    calc = CalCulateTotalArrivalTime()
    calc.truck_times = sa.T
    calc.drone_times = sa.D
    calc.flight_range = sa.flight_limit
    calc.depot_index = sa.depot

    checker = SolutionFeasibility(
        n_nodes=sa.n_customers + 1,
        n_drones=2,
        depot_index=sa.depot,
        drone_times=sa.D,
        flight_range=sa.flight_limit,
    )

    initial_solution = _build_tuple_solution_from_construction(instance)
    init_feasible, init_cost = sa.evaluate_solution(initial_solution, calc, checker)
    if not init_feasible:
        initial_solution = [[i for i in range(sa.n_customers + 1)] + [0], [], []]
        init_feasible, init_cost = sa.evaluate_solution(initial_solution, calc, checker)
        if not init_feasible:
            raise ValueError("Both construction and truck-only initial solutions are infeasible.")

    global_best_solution = None
    global_best_cost = float("inf")
    run_costs = []
    run_times = []
    current_start = copy.deepcopy(initial_solution)
    stagnation_counter = 0
    restart_count = 0

    for run_id in range(runs):
        start = sa.time.perf_counter()
        run_best_solution, run_best_cost = sa.simulated_annealing(
            copy.deepcopy(current_start),
            warmup_iterations=0,
            iterations=iterations,
            final_temperature=final_temperature,
        )
        elapsed = sa.time.perf_counter() - start

        run_costs.append(run_best_cost)
        run_times.append(elapsed)

        improved = run_best_cost < global_best_cost
        if run_best_cost < global_best_cost:
            global_best_cost = run_best_cost
            global_best_solution = run_best_solution
            stagnation_counter = 0
            current_start = copy.deepcopy(run_best_solution)
            print(f"New best at run {run_id + 1}/{runs}: {global_best_cost}")
            print(f"Best solution: {sa.format_solution_pipe(global_best_solution)}")
        else:
            stagnation_counter += 1
            current_start = copy.deepcopy(run_best_solution)

        if (not improved) and stagnation_counter >= stagnation_limit and run_id < runs - 1:
            restart_candidate = _build_tuple_solution_from_construction(instance)
            if _is_feasible(restart_candidate, instance):
                current_start = restart_candidate
            else:
                current_start = [[i for i in range(sa.n_customers + 1)] + [0], [], []]
            restart_count += 1
            stagnation_counter = 0
            print(f"Random restart triggered after {stagnation_limit} non-improving runs (total restarts: {restart_count})")

    avg_obj = sum(run_costs) / len(run_costs)
    improvement_abs = init_cost - global_best_cost
    improvement_pct = (improvement_abs / init_cost * 100.0) if init_cost > 0 else 0.0
    avg_runtime_per_run = sum(run_times) / len(run_times)

    print(f"Initial objective value: {init_cost}")
    print(f"Average objective value (over {runs} runs): {avg_obj}")
    print(f"Best objective: {global_best_cost}")
    print(
        f"Improvement (initial -> best): {improvement_abs} "
        f"({improvement_pct:.2f}% reduction)"
    )
    print(
        f"Average runtime per iteration (1 iteration = SA run with {iterations} loops): "
        f"{avg_runtime_per_run:.4f} seconds"
    )
    print(f"Random restarts triggered: {restart_count}")

    return global_best_solution, global_best_cost


if __name__ == "__main__":
    instance = read_instance(f"{absolute_path}{file_name}")
    _configure_sa_instance(instance)

    best_solution, best_cost = run_random_restart_experiments(
        instance,
        runs=100,
        iterations=10000,
        final_temperature=0.1,
        stagnation_limit=10,
    )
    print(f"Best solution after all runs: {best_solution}")
    print(f"Best solution (pipe format, checker 1-based): {sa.format_solution_pipe(best_solution)}")
    print(f"Best objective: {best_cost}")
    plot_solution(best_solution, instance, title=f"SA Best Solution ({file_name})")
