import random
from pathlib import Path
import sys

ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
CORE_DIR = ASSIGNMENT_DIR / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.append(str(CORE_DIR))

from operator_context import assert_context_is_set, get_operator_context, set_operator_context
from drone_route_utils import (
    drone_route_is_feasible,
    remap_drone_route_by_endpoint_nodes,
    repair_drone_route,
)

_TOP_K_BAD_NODES = 14
_PAIR_TRIALS = 42
_EXPLORE_PROB = 0.12
_SYNC_PEN_WEIGHT = 0.10


def set_search_progress(progress):
    return None


def _clone_solution(solution):
    return [solution[0][:], solution[1][:], solution[2][:]]


def _route_endpoint_unique(route):
    used_launch = set()
    used_land = set()
    for _, launch_idx, land_idx in route:
        if launch_idx in used_launch or land_idx in used_land:
            return False
        used_launch.add(launch_idx)
        used_land.add(land_idx)
    return True


def _prefix_truck_times(truck_route, truck_times):
    prefix = [0.0]
    for i in range(len(truck_route) - 1):
        prefix.append(prefix[-1] + truck_times[truck_route[i]][truck_route[i + 1]])
    return prefix


def _trip_penalty(trip, truck_route, prefix, drone_times):
    node, launch_idx, land_idx = trip
    launch_node = truck_route[launch_idx]
    land_node = truck_route[land_idx]
    sortie_time = drone_times[launch_node][node] + drone_times[node][land_node]
    truck_time = prefix[land_idx] - prefix[launch_idx]
    wait_excess = max(0.0, sortie_time - truck_time)
    mismatch = abs(sortie_time - truck_time)
    return 3.0 * wait_excess + 0.6 * mismatch + 0.02 * sortie_time


def _solution_sync_penalty(truck, drone1, drone2, truck_times, drone_times):
    prefix = _prefix_truck_times(truck, truck_times)
    pen = 0.0
    for trip in drone1:
        pen += _trip_penalty(trip, truck, prefix, drone_times)
    for trip in drone2:
        pen += _trip_penalty(trip, truck, prefix, drone_times)
    pen += 2.0 * abs(len(drone1) - len(drone2))
    return pen


def _truck_cost(truck, truck_times):
    return sum(truck_times[truck[i]][truck[i + 1]] for i in range(len(truck) - 1))


def _edge_badness(truck, idx, truck_times):
    if idx <= 0 or idx >= len(truck) - 1:
        return -1e18
    a = truck[idx - 1]
    b = truck[idx]
    c = truck[idx + 1]
    return truck_times[a][b] + truck_times[b][c] - truck_times[a][c]


def _rank_biased_pick(scored):
    if not scored:
        return None
    k = min(_TOP_K_BAD_NODES, len(scored))
    if k <= 1:
        return scored[0]
    weights = [k - i for i in range(k)]
    return random.choices(scored[:k], weights=weights, k=1)[0]


def _sample_swap_pairs(truck, truck_times):
    internal = list(range(1, len(truck) - 1))
    if len(internal) < 2:
        return []

    scored = [(_edge_badness(truck, idx, truck_times), idx) for idx in internal]
    scored.sort(key=lambda x: x[0], reverse=True)
    pool = scored[: min(_TOP_K_BAD_NODES, len(scored))]
    if len(pool) < 2:
        return []

    pairs = []
    seen = set()
    attempts = 0
    while attempts < (_PAIR_TRIALS * 3) and len(pairs) < _PAIR_TRIALS:
        attempts += 1
        pick1 = _rank_biased_pick(pool)
        pick2 = _rank_biased_pick(pool)
        if pick1 is None or pick2 is None:
            continue
        i = pick1[1]
        j = pick2[1]
        if i == j:
            continue
        if abs(i - j) <= 1:
            continue
        a, b = (i, j) if i < j else (j, i)
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


def operator(current_solution):
    assert_context_is_set()
    truck_times, drone_times, _, _ = get_operator_context()

    truck, drone1, drone2 = _clone_solution(current_solution)
    if len(truck) <= 4:
        return current_solution

    old_truck_cost = _truck_cost(truck, truck_times)
    old_sync_pen = _solution_sync_penalty(truck, drone1, drone2, truck_times, drone_times)

    pairs = _sample_swap_pairs(truck, truck_times)
    if not pairs:
        return current_solution

    candidates = []
    for i, j in pairs:
        new_truck = truck[:]
        new_truck[i], new_truck[j] = new_truck[j], new_truck[i]

        drone1_new = remap_drone_route_by_endpoint_nodes(truck, new_truck, drone1)
        drone2_new = remap_drone_route_by_endpoint_nodes(truck, new_truck, drone2)
        if drone1_new is None or drone2_new is None:
            continue

        if not drone_route_is_feasible(drone1_new):
            drone1_new = repair_drone_route(new_truck, drone1_new, max_repairs=10, max_neighbors=8)
            if drone1_new is None:
                continue
        if not drone_route_is_feasible(drone2_new):
            drone2_new = repair_drone_route(new_truck, drone2_new, max_repairs=10, max_neighbors=8)
            if drone2_new is None:
                continue

        if not _route_endpoint_unique(drone1_new) or not _route_endpoint_unique(drone2_new):
            continue

        truck_delta = _truck_cost(new_truck, truck_times) - old_truck_cost
        sync_delta = (
            _solution_sync_penalty(new_truck, drone1_new, drone2_new, truck_times, drone_times)
            - old_sync_pen
        )
        score = truck_delta + _SYNC_PEN_WEIGHT * sync_delta
        candidates.append((score, [new_truck, drone1_new, drone2_new]))

    if not candidates:
        return current_solution

    candidates.sort(key=lambda x: x[0])
    if random.random() < _EXPLORE_PROB:
        top_k = min(4, len(candidates))
        return random.choice(candidates[:top_k])[1]
    return candidates[0][1]
