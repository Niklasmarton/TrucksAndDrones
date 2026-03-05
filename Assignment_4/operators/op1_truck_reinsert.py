import random
from pathlib import Path
import sys

CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.append(str(CORE_DIR))

from drone_route_utils import (
    drone_route_is_feasible,
    repair_drone_route,
    update_drone_route_indices,
)
from operator_context import assert_context_is_set, get_operator_context, set_operator_context


def _clone_solution(solution):
    return [solution[0][:], solution[1][:], solution[2][:]]


def _biased_pick(candidates, maximize, best_bias=0.75, sample_top=3):
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda x: x[1], reverse=maximize)
    if random.random() < best_bias:
        return ranked[0][0]
    k = min(sample_top, len(ranked))
    return random.choice(ranked[:k])[0]


def calculate_worst_index(solution):
    """
    Returns (idx, node) where idx is selected among top-k worst removable nodes.
    """
    T, _, _, _ = get_operator_context()
    truck = solution[0]
    k = 6
    worst = []

    for i in range(1, len(truck) - 2):
        prev_node = truck[i - 1]
        node = truck[i]
        next_node = truck[i + 1]
        score = T[prev_node][node] + T[node][next_node] - T[prev_node][next_node]

        if len(worst) < k:
            worst.append((i, score))
        else:
            min_pos = min(range(k), key=lambda p: worst[p][1])
            if score > worst[min_pos][1]:
                worst[min_pos] = (i, score)

    if not worst:
        return None
    i = _biased_pick(worst, maximize=True, best_bias=0.8, sample_top=3)
    return i, truck[i]


def calculate_best_index(truck_after_removal, node_to_insert):
    """
    Returns insertion index selected among top-k cheapest insertion edges.
    """
    T, _, _, _ = get_operator_context()
    k = 6
    best = []

    for i in range(0, len(truck_after_removal) - 2):
        a = truck_after_removal[i]
        b = truck_after_removal[i + 1]
        j = i + 1
        delta = T[a][node_to_insert] + T[node_to_insert][b] - T[a][b]

        if len(best) < k:
            best.append((j, delta))
        else:
            worst_pos = max(range(k), key=lambda p: best[p][1])
            if delta < best[worst_pos][1]:
                best[worst_pos] = (j, delta)

    if not best:
        return None
    return _biased_pick(best, maximize=False, best_bias=0.8, sample_top=3)


def truck_reinsert(solution):
    assert_context_is_set()
    sol = _clone_solution(solution)
    truck, drone1, drone2 = sol

    if len(truck) <= 3:
        return solution

    worst_pick = calculate_worst_index(sol)
    if worst_pick is None:
        return solution
    removal_index, removed_node = worst_pick
    truck.pop(removal_index)

    insertion_index = calculate_best_index(truck, removed_node)
    if insertion_index is None:
        return solution
    truck.insert(insertion_index, removed_node)

    drone1_new = update_drone_route_indices(drone1, removal_index, insertion_index, removal_index)
    drone2_new = update_drone_route_indices(drone2, removal_index, insertion_index, removal_index)

    if not drone_route_is_feasible(drone1_new):
        drone1_new = repair_drone_route(truck, drone1_new, max_repairs=5, max_neighbors=5)
        if drone1_new is None:
            return solution

    if not drone_route_is_feasible(drone2_new):
        drone2_new = repair_drone_route(truck, drone2_new, max_repairs=5, max_neighbors=5)
        if drone2_new is None:
            return solution

    sol[1] = drone1_new
    sol[2] = drone2_new
    return sol
