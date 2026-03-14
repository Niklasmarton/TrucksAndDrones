from pathlib import Path
import sys

CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.append(str(CORE_DIR))

from drone_route_utils import build_drone_pair, drone_route_is_feasible
from operator_context import assert_context_is_set, get_operator_context, set_operator_context

_SEARCH_PROGRESS = 0.5
_SYNC_PEN_WEIGHT = 0.12
_MAX_DRONE_POSITIONS = 6
_MAX_TRUCK_POSITIONS = 8


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


def _routes_feasible(drone1, drone2):
    return (
        drone_route_is_feasible(drone1)
        and drone_route_is_feasible(drone2)
        and _route_endpoint_unique(drone1)
        and _route_endpoint_unique(drone2)
    )


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
    n_customers = max(0, truck_len - 2)
    return max(2, min(6, int(round(0.05 * n_customers))))


def _truck_total_cost(truck, truck_times):
    return sum(truck_times[truck[i]][truck[i + 1]] for i in range(len(truck) - 1))


def _prefix_truck_times(truck_route, truck_times):
    prefix = [0.0]
    for i in range(len(truck_route) - 1):
        prefix.append(prefix[-1] + truck_times[truck_route[i]][truck_route[i + 1]])
    return prefix


def _trip_penalty(trip, truck_route, prefix, drone_times):
    node, launch_idx, land_idx = trip
    if (
        launch_idx < 0
        or land_idx < 0
        or launch_idx >= len(truck_route)
        or land_idx >= len(truck_route)
        or launch_idx >= land_idx
    ):
        return 1e9
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


def _surrogate_score(
    old_truck,
    old_d1,
    old_d2,
    new_truck,
    new_d1,
    new_d2,
    truck_times,
    drone_times,
):
    old_truck_cost = _truck_total_cost(old_truck, truck_times)
    new_truck_cost = _truck_total_cost(new_truck, truck_times)
    old_sync = _solution_sync_penalty(old_truck, old_d1, old_d2, truck_times, drone_times)
    new_sync = _solution_sync_penalty(new_truck, new_d1, new_d2, truck_times, drone_times)
    return (new_truck_cost - old_truck_cost) + _SYNC_PEN_WEIGHT * (new_sync - old_sync)


def _pick_destroy_nodes(truck, drone1, drone2, truck_times, depot, k):
    used_endpoints = _used_endpoints(drone1, drone2)
    items = []
    for idx in range(1, len(truck) - 1):
        if idx in used_endpoints:
            continue
        node = truck[idx]
        if node == depot:
            continue
        a = truck[idx - 1]
        b = truck[idx + 1]
        gain = truck_times[a][node] + truck_times[node][b] - truck_times[a][b]
        items.append((gain, idx, node))
    if not items:
        return []

    items.sort(reverse=True, key=lambda x: x[0])
    mode = 0 if _SEARCH_PROGRESS < 0.33 else (1 if _SEARCH_PROGRESS < 0.66 else 2)

    if mode == 0:
        chosen = items[:k]  # worst destroy
    elif mode == 1:
        # Related destroy: choose around top-gain seed using truck distance.
        seed = items[0][2]
        scored = []
        for gain, idx, node in items:
            scored.append((truck_times[seed][node], -gain, idx, node))
        scored.sort(key=lambda x: (x[0], x[1]))
        chosen = [(items[0][0], items[0][1], items[0][2])]
        chosen.extend([(-b, idx, node) for _, b, idx, node in scored if idx != items[0][1]])
        chosen = chosen[:k]
    else:
        # Top-pool diversified destroy: deterministic spread over best pool.
        pool = items[: min(len(items), 2 * k + 2)]
        if len(pool) <= k:
            chosen = pool
        else:
            picked = []
            for j in range(k):
                pos = int(round(j * (len(pool) - 1) / float(max(1, k - 1))))
                picked.append(pool[pos])
            chosen = picked

    return sorted(chosen, key=lambda x: x[1])


def _candidate_truck_positions(node, truck, truck_times):
    scored = []
    for ins_idx in range(1, len(truck)):
        a = truck[ins_idx - 1]
        b = truck[ins_idx]
        delta = truck_times[a][node] + truck_times[node][b] - truck_times[a][b]
        scored.append((delta, ins_idx))
    scored.sort(key=lambda x: x[0])
    return [ins_idx for _, ins_idx in scored[:_MAX_TRUCK_POSITIONS]]


def _candidate_drone_positions(route_len, cap=_MAX_DRONE_POSITIONS):
    all_pos = list(range(0, route_len + 1))
    if len(all_pos) <= cap:
        return all_pos
    picks = {0, route_len}
    for k in range(1, cap - 1):
        pos = int(round((k * route_len) / float(cap - 1)))
        picks.add(pos)
    return sorted(picks)


def _insert_node_options(node, truck, drone1, drone2, truck_times, drone_times):
    options = []

    # Truck insertion options
    for ins_idx in _candidate_truck_positions(node, truck, truck_times):
        new_truck = truck[:ins_idx] + [node] + truck[ins_idx:]
        new_d1 = []
        for c, l, r in drone1:
            if l >= ins_idx:
                l += 1
            if r >= ins_idx:
                r += 1
            new_d1.append((c, l, r))
        new_d2 = []
        for c, l, r in drone2:
            if l >= ins_idx:
                l += 1
            if r >= ins_idx:
                r += 1
            new_d2.append((c, l, r))
        if not _routes_feasible(new_d1, new_d2):
            continue
        score = _surrogate_score(
            truck, drone1, drone2, new_truck, new_d1, new_d2, truck_times, drone_times
        )
        options.append((score, [new_truck, new_d1, new_d2]))

    # Drone insertion options
    for route_id, target, other in ((1, drone1, drone2), (2, drone2, drone1)):
        for ins_idx in _candidate_drone_positions(len(target)):
            pair = build_drone_pair(
                node,
                truck,
                target,
                ins_idx,
                max_neighbors=10,
                pair_explore_prob=0.0,
            )
            if pair is None:
                continue
            new_target = target[:]
            new_target.insert(ins_idx, (node, pair[0], pair[1]))
            new_target.sort(key=lambda x: (x[1], x[2], x[0]))
            if not drone_route_is_feasible(new_target) or not _route_endpoint_unique(new_target):
                continue
            if route_id == 1:
                new_d1, new_d2 = new_target, other[:]
            else:
                new_d1, new_d2 = other[:], new_target
            if not _routes_feasible(new_d1, new_d2):
                continue
            score = _surrogate_score(
                truck, drone1, drone2, truck, new_d1, new_d2, truck_times, drone_times
            )
            options.append((score, [truck[:], new_d1, new_d2]))

    options.sort(key=lambda x: x[0])
    return options


def operator(current_solution):
    """
    ALNS-style destroy/repair operator:
    - destroy mode chosen by search progress (worst / related / top-pool-diverse)
    - regret-2 repair over truck+drone insertion options
    """
    assert_context_is_set()
    truck_times, drone_times, _, depot = get_operator_context()

    truck, drone1, drone2 = _clone_solution(current_solution)
    if len(truck) <= 4:
        return current_solution

    k = _destroy_size(len(truck))
    destroy_items = _pick_destroy_nodes(truck, drone1, drone2, truck_times, depot, k)
    if not destroy_items:
        return current_solution

    removed_nodes = []
    for gain, rem_idx, node in sorted(destroy_items, key=lambda x: x[1], reverse=True):
        removed_nodes.append((gain, node))
        truck = truck[:rem_idx] + truck[rem_idx + 1 :]
        drone1 = _shift_route_after_truck_removal(drone1, rem_idx)
        drone2 = _shift_route_after_truck_removal(drone2, rem_idx)
        if drone1 is None or drone2 is None:
            return current_solution

    pending = [node for _, node in sorted(removed_nodes, reverse=True, key=lambda x: x[0])]
    while pending:
        best_node_idx = None
        best_node_options = None
        best_regret = None
        best_first = None

        for idx, node in enumerate(pending):
            options = _insert_node_options(node, truck, drone1, drone2, truck_times, drone_times)
            if not options:
                continue
            first = options[0][0]
            second = options[1][0] if len(options) > 1 else (first + 1e6)
            regret = second - first
            if (
                best_node_idx is None
                or regret > best_regret
                or (regret == best_regret and first < best_first)
            ):
                best_node_idx = idx
                best_node_options = options
                best_regret = regret
                best_first = first

        if best_node_idx is None or not best_node_options:
            return current_solution

        _, chosen = best_node_options[0]
        truck, drone1, drone2 = chosen
        pending.pop(best_node_idx)

    new_solution = [truck, drone1, drone2]
    if new_solution == current_solution:
        return current_solution
    return new_solution
