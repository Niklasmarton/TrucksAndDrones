import random
from pathlib import Path
import sys

CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.append(str(CORE_DIR))

from drone_route_utils import (
    drone_route_is_feasible,
    remap_drone_route_by_endpoint_nodes,
    repair_drone_route,
)
from operator_context import assert_context_is_set, get_operator_context, set_operator_context


def _clone_solution(solution):
    return [solution[0][:], solution[1][:], solution[2][:]]


def _edge_check_budget(truck_len):
    # Keep 2-opt exploration roughly linear in route size on large instances.
    return max(80, min(300, 3 * truck_len))


def _choose_2opt_indices(truck_route):
    """
    Choose (i, j) using best-improvement 2-opt within a fixed edge-check budget:
    - candidate order is pseudo-randomized (random start offsets)
    - tracks best improving move seen across entire budget
    - returns best move found (or None if no improvement exists)
    Keeps depot fixed at both ends.
    """
    T, _, _, _ = get_operator_context()
    n = len(truck_route)
    if n < 5:
        return None

    i_count = n - 3
    if i_count <= 0:
        return None

    checks = 0
    max_checks = _edge_check_budget(n)
    best_delta = 0.0
    best_move = None

    i_start = random.randint(1, n - 3)
    for i_step in range(i_count):
        i = 1 + ((i_start - 1 + i_step) % i_count)
        a = truck_route[i - 1]
        b = truck_route[i]
        j_min = i + 1
        j_count = (n - 1) - j_min
        if j_count <= 0:
            continue

        j_start = j_min + random.randint(0, j_count - 1)
        for j_step in range(j_count):
            if checks >= max_checks:
                return best_move
            j = j_min + ((j_start - j_min + j_step) % j_count)
            c = truck_route[j]
            d = truck_route[j + 1]
            delta = T[a][c] + T[b][d] - T[a][b] - T[c][d]
            checks += 1
            if delta < best_delta:
                best_delta = delta
                best_move = (i, j)

    return best_move


def truck_2opt(solution):
    assert_context_is_set()
    sol = _clone_solution(solution)
    old_truck = sol[0]

    move = _choose_2opt_indices(old_truck)
    if move is None:
        return solution

    i, j = move
    new_truck = old_truck[:i] + list(reversed(old_truck[i : j + 1])) + old_truck[j + 1 :]
    if new_truck == old_truck:
        return solution

    drone1_new = remap_drone_route_by_endpoint_nodes(old_truck, new_truck, sol[1])
    drone2_new = remap_drone_route_by_endpoint_nodes(old_truck, new_truck, sol[2])
    if drone1_new is None or drone2_new is None:
        return solution

    if not drone_route_is_feasible(drone1_new):
        drone1_new = repair_drone_route(new_truck, drone1_new, max_repairs=6, max_neighbors=6)
        if drone1_new is None:
            return solution

    if not drone_route_is_feasible(drone2_new):
        drone2_new = repair_drone_route(new_truck, drone2_new, max_repairs=6, max_neighbors=6)
        if drone2_new is None:
            return solution

    sol[0] = new_truck
    sol[1] = drone1_new
    sol[2] = drone2_new
    return sol
