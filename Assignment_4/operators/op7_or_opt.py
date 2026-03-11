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

_EXPLORE_PROB = 0.10
_SYNC_PEN_WEIGHT = 0.10


def _clone_solution(solution):
    return [solution[0][:], solution[1][:], solution[2][:]]


def _search_budget(truck_len):
    # Bound candidate checks while keeping enough breadth on larger instances.
    return max(24, min(96, 2 * truck_len))


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


def _route_endpoint_unique(route):
    used_launch = set()
    used_land = set()
    for _, launch_idx, land_idx in route:
        if launch_idx in used_launch or land_idx in used_land:
            return False
        used_launch.add(launch_idx)
        used_land.add(land_idx)
    return True


def _solution_penalty(truck, drone1, drone2, truck_times, drone_times):
    prefix = _prefix_truck_times(truck, truck_times)
    pen = 0.0
    for trip in drone1:
        pen += _trip_penalty(trip, truck, prefix, drone_times)
    for trip in drone2:
        pen += _trip_penalty(trip, truck, prefix, drone_times)
    pen += 2.0 * abs(len(drone1) - len(drone2))
    return pen


def _removal_gain(truck, start, seg_len, truck_times):
    a = truck[start - 1]
    b = truck[start + seg_len]
    if seg_len == 1:
        x = truck[start]
        return truck_times[a][x] + truck_times[x][b] - truck_times[a][b]
    x = truck[start]
    y = truck[start + 1]
    return truck_times[a][x] + truck_times[x][y] + truck_times[y][b] - truck_times[a][b]


def _insertion_delta(reduced_truck, ins_idx, segment, removal_gain, truck_times):
    prev_node = reduced_truck[ins_idx - 1]
    next_node = reduced_truck[ins_idx]
    if len(segment) == 1:
        ins_cost = (
            truck_times[prev_node][segment[0]]
            + truck_times[segment[0]][next_node]
            - truck_times[prev_node][next_node]
        )
    else:
        ins_cost = (
            truck_times[prev_node][segment[0]]
            + truck_times[segment[0]][segment[1]]
            + truck_times[segment[1]][next_node]
            - truck_times[prev_node][next_node]
        )
    return ins_cost - removal_gain


def _apply_oropt_move(truck, start, seg_len, ins_idx):
    segment = truck[start : start + seg_len]
    reduced = truck[:start] + truck[start + seg_len :]
    if ins_idx < 0 or ins_idx > len(reduced):
        return None
    return reduced[:ins_idx] + segment + reduced[ins_idx:]


def _candidate_moves(truck, truck_times):
    n = len(truck)
    if n < 5:
        return []

    moves = []
    for seg_len in (1, 2):
        if n - 2 <= seg_len:
            continue
        for start in range(1, n - seg_len):
            removal_gain = _removal_gain(truck, start, seg_len, truck_times)
            segment = truck[start : start + seg_len]
            reduced = truck[:start] + truck[start + seg_len :]

            local = []
            for ins_idx in range(1, len(reduced)):
                # Reinserting at the original index produces the same route.
                if ins_idx == start:
                    continue
                delta = _insertion_delta(reduced, ins_idx, segment, removal_gain, truck_times)
                local.append((delta, start, seg_len, ins_idx))

            if not local:
                continue
            local.sort(key=lambda x: x[0])
            keep = min(5, len(local))
            if random.random() < _EXPLORE_PROB:
                random.shuffle(local[:keep])
            moves.extend(local[:keep])

    moves.sort(key=lambda x: x[0])
    if random.random() < _EXPLORE_PROB and len(moves) > 10:
        top = moves[:10]
        random.shuffle(top)
        return top + moves[10:]
    return moves


def truck_2opt(solution):
    """
    Compatibility entrypoint for the SA framework:
    replaces classical 2-opt with Or-opt(1,2) segment relocation.
    """
    assert_context_is_set()
    sol = _clone_solution(solution)
    old_truck = sol[0]
    truck_times, drone_times, _, _ = get_operator_context()
    current_penalty = _solution_penalty(old_truck, sol[1], sol[2], truck_times, drone_times)

    moves = _candidate_moves(old_truck, truck_times)
    if not moves:
        return solution

    budget = _search_budget(len(old_truck))
    feasible_candidates = []
    for delta, start, seg_len, ins_idx in moves[:budget]:
        new_truck = _apply_oropt_move(old_truck, start, seg_len, ins_idx)
        if new_truck is None or new_truck == old_truck:
            continue

        drone1_new = remap_drone_route_by_endpoint_nodes(old_truck, new_truck, sol[1])
        drone2_new = remap_drone_route_by_endpoint_nodes(old_truck, new_truck, sol[2])
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

        new_penalty = _solution_penalty(new_truck, drone1_new, drone2_new, truck_times, drone_times)
        score = delta + _SYNC_PEN_WEIGHT * (new_penalty - current_penalty)
        feasible_candidates.append((score, [new_truck, drone1_new, drone2_new]))
        if len(feasible_candidates) >= 20:
            break

    if not feasible_candidates:
        return solution

    feasible_candidates.sort(key=lambda x: x[0])
    if random.random() < _EXPLORE_PROB:
        top_k = min(3, len(feasible_candidates))
        return random.choice(feasible_candidates[:top_k])[1]
    return feasible_candidates[0][1]
