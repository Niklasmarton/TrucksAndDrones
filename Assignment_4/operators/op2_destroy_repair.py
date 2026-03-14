import random
from pathlib import Path
import sys

CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.append(str(CORE_DIR))

from drone_route_utils import build_drone_pair, drone_route_is_feasible
from operator_context import assert_context_is_set, get_operator_context, set_operator_context

_EXPLORE_PROB = 0.15
_SEARCH_PROGRESS = 0.5


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def set_search_progress(progress):
    global _SEARCH_PROGRESS
    _SEARCH_PROGRESS = _clamp01(progress)


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


def _used_endpoints(drone1, drone2):
    used = set()
    for route in (drone1, drone2):
        for _, launch_idx, land_idx in route:
            used.add(launch_idx)
            used.add(land_idx)
    return used


def _shift_route_after_truck_removal(route, removed_idx):
    shifted = []
    for node, launch_idx, land_idx in route:
        if launch_idx == removed_idx or land_idx == removed_idx:
            return None
        if launch_idx > removed_idx:
            launch_idx -= 1
        if land_idx > removed_idx:
            land_idx -= 1
        shifted.append((node, launch_idx, land_idx))
    shifted.sort(key=lambda x: (x[1], x[2], x[0]))
    return shifted


def _destroy_size(truck_len):
    # Destroy about 8-15% of truck-only customers.
    n_customers = max(0, truck_len - 2)
    if n_customers <= 10:
        return 1
    if n_customers <= 20:
        return 2
    return 3


def _pick_destroy_indices(truck, drone1, drone2, truck_times, depot, count):
    used_endpoints = _used_endpoints(drone1, drone2)
    scored = []
    for idx in range(1, len(truck) - 1):
        if idx in used_endpoints:
            continue
        node = truck[idx]
        if node == depot:
            continue
        a = truck[idx - 1]
        b = truck[idx + 1]
        gain = truck_times[a][node] + truck_times[node][b] - truck_times[a][b]
        scored.append((gain, idx))

    if not scored:
        return []

    scored.sort(reverse=True, key=lambda x: x[0])
    pool_size = min(len(scored), max(count, 3 * count))
    pool = scored[:pool_size]
    if random.random() < _EXPLORE_PROB:
        random.shuffle(pool)

    chosen = []
    used_idx = set()
    for _, idx in pool:
        if idx in used_idx:
            continue
        chosen.append(idx)
        used_idx.add(idx)
        if len(chosen) >= count:
            break
    return sorted(chosen)


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


def _drone_trip_penalty(node, pair, truck, truck_times, drone_times):
    launch_idx, land_idx = pair
    launch_node = truck[launch_idx]
    land_node = truck[land_idx]
    sortie_time = drone_times[launch_node][node] + drone_times[node][land_node]

    truck_leg = 0.0
    for i in range(launch_idx, land_idx):
        truck_leg += truck_times[truck[i]][truck[i + 1]]

    wait_excess = max(0.0, sortie_time - truck_leg)
    mismatch = abs(sortie_time - truck_leg)
    return 2.5 * wait_excess + 0.5 * mismatch + 0.01 * sortie_time


def _candidate_drone_inserts(node, truck, drone1, drone2, truck_times, drone_times):
    candidates = []
    route_order = [(1, drone1, drone2), (2, drone2, drone1)]
    route_order.sort(key=lambda x: len(x[1]))

    for route_id, target_route, other_route in route_order:
        positions = list(range(0, len(target_route) + 1))
        if len(positions) > 6:
            positions = [0, len(target_route) // 2, len(target_route)]
            while len(positions) < 6:
                positions.append(random.randint(0, len(target_route)))
            positions = list(dict.fromkeys(positions))

        if random.random() < _EXPLORE_PROB:
            random.shuffle(positions)

        for ins_idx in positions:
            pair = build_drone_pair(
                node,
                truck,
                target_route,
                ins_idx,
                max_neighbors=8,
                pair_explore_prob=0.10,
            )
            if pair is None:
                continue

            new_target = target_route[:]
            new_target.insert(ins_idx, (node, pair[0], pair[1]))
            new_target.sort(key=lambda x: (x[1], x[2], x[0]))
            if not drone_route_is_feasible(new_target):
                continue
            if not _route_endpoint_unique(new_target):
                continue
            if not drone_route_is_feasible(other_route):
                continue
            if not _route_endpoint_unique(other_route):
                continue

            if route_id == 1:
                d1_len = len(new_target)
                d2_len = len(other_route)
            else:
                d1_len = len(other_route)
                d2_len = len(new_target)

            timing_pen = _drone_trip_penalty(node, pair, truck, truck_times, drone_times)
            balance_pen = 2.0 * abs(d1_len - d2_len)
            score = timing_pen + balance_pen
            candidates.append((score, route_id, new_target))

    return candidates


def operator(current_solution):
    """
    op6: LNS destroy/repair
    - Destroy: remove several high-cost truck customers (excluding drone endpoints).
    - Repair: reinsert each removed node using best truck insertion or feasible drone insertion.
    """
    assert_context_is_set()
    truck_times, drone_times, _, depot = get_operator_context()

    # Early phase: keep LNS mostly quiet to let truck-focused operators
    # establish a stronger backbone route first.
    if _SEARCH_PROGRESS < 0.25 and random.random() < 0.25:
        return current_solution

    candidate = _clone_solution(current_solution)
    truck, drone1, drone2 = candidate
    if len(truck) <= 4:
        return current_solution

    remove_count = _destroy_size(len(truck))
    destroy_indices = _pick_destroy_indices(truck, drone1, drone2, truck_times, depot, remove_count)
    if not destroy_indices:
        return current_solution

    removed_nodes = []
    for rem_idx in sorted(destroy_indices, reverse=True):
        removed_nodes.append(truck[rem_idx])
        truck = truck[:rem_idx] + truck[rem_idx + 1 :]
        drone1 = _shift_route_after_truck_removal(drone1, rem_idx)
        drone2 = _shift_route_after_truck_removal(drone2, rem_idx)
        if drone1 is None or drone2 is None:
            return current_solution

    random.shuffle(removed_nodes)
    for node in removed_nodes:
        truck_delta, truck_ins_idx = _best_truck_insert(node, truck, truck_times)
        drone_candidates = _candidate_drone_inserts(node, truck, drone1, drone2, truck_times, drone_times)

        use_drone = False
        if drone_candidates:
            drone_candidates.sort(key=lambda x: x[0])
            best_drone_score, best_route_id, best_new_target = drone_candidates[0]
            threshold = truck_delta if truck_delta is not None else float("inf")
            if best_drone_score < threshold or random.random() < _EXPLORE_PROB:
                use_drone = True

        if use_drone:
            if best_route_id == 1:
                drone1 = best_new_target
            else:
                drone2 = best_new_target
        else:
            if truck_ins_idx is None:
                return current_solution
            truck = truck[:truck_ins_idx] + [node] + truck[truck_ins_idx:]

    new_solution = [truck, drone1, drone2]
    if new_solution == current_solution:
        return current_solution
    return new_solution
