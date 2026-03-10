import random
from pathlib import Path
import sys

CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.append(str(CORE_DIR))

from drone_route_utils import build_drone_pair, drone_route_is_feasible
from operator_context import assert_context_is_set, get_operator_context, set_operator_context

_EXPLORE_PROB = 0.14


def _clone_solution(solution):
    return [solution[0][:], solution[1][:], solution[2][:]]


def _prefix_truck_times(truck_route, truck_times):
    prefix = [0.0]
    for i in range(len(truck_route) - 1):
        prefix.append(prefix[-1] + truck_times[truck_route[i]][truck_route[i + 1]])
    return prefix


def _trip_penalty(node, launch_idx, land_idx, truck_route, prefix, drone_times):
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


def _pair_endpoints_available(route, launch_idx, land_idx):
    for _, l_idx, d_idx in route:
        if l_idx == launch_idx or d_idx == land_idx:
            return False
    return True


def _used_endpoints(solution):
    _, drone1, drone2 = solution
    used = set()
    for route in (drone1, drone2):
        for _, launch_idx, land_idx in route:
            used.add(launch_idx)
            used.add(land_idx)
    return used


def _shift_route_after_truck_removal(route, removed_idx):
    shifted = []
    for node, launch_idx, land_idx in route:
        # Do not remove a truck node currently used as endpoint.
        if launch_idx == removed_idx or land_idx == removed_idx:
            return None
        if launch_idx > removed_idx:
            launch_idx -= 1
        if land_idx > removed_idx:
            land_idx -= 1
        shifted.append((node, launch_idx, land_idx))
    shifted.sort(key=lambda x: (x[1], x[2], x[0]))
    return shifted


def _candidate_positions(route_len, max_positions):
    if route_len + 1 <= max_positions:
        return list(range(route_len + 1))
    picks = [0, route_len // 2, route_len]
    while len(picks) < max_positions:
        picks.append(random.randint(0, route_len))
    # Deduplicate while preserving order.
    out = []
    seen = set()
    for x in picks:
        if 0 <= x <= route_len and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _search_budgets(truck_len):
    n_customers = max(0, truck_len - 2)
    if n_customers >= 100:
        return {"truck_nodes": 10, "positions": 3, "max_candidates": 16, "pair_neighbors": 6}
    if n_customers >= 50:
        return {"truck_nodes": 14, "positions": 4, "max_candidates": 24, "pair_neighbors": 7}
    return {"truck_nodes": 20, "positions": 6, "max_candidates": 36, "pair_neighbors": 8}


def _best_truck_nodes_for_removal(solution, top_k):
    truck_times, _, _, depot = get_operator_context()
    truck = solution[0]
    used_endpoints = _used_endpoints(solution)
    scored = []
    for idx in range(1, len(truck) - 1):
        node = truck[idx]
        if node == depot:
            continue
        if idx in used_endpoints:
            continue
        a = truck[idx - 1]
        b = truck[idx + 1]
        gain = truck_times[a][node] + truck_times[node][b] - truck_times[a][b]
        scored.append((gain, idx))
    if not scored:
        return []
    scored.sort(reverse=True, key=lambda x: x[0])
    out = [idx for _, idx in scored[: min(top_k, len(scored))]]
    if random.random() < _EXPLORE_PROB:
        random.shuffle(out)
    return out


def operator(current_solution):
    """
    op5: truck->drone insertion
    - Remove one customer node from truck route.
    - Insert that customer as a drone trip on either drone 1 or drone 2.
    - Node removal from truck is guided by highest truck gain.
    """
    assert_context_is_set()
    truck_times, drone_times, _, _ = get_operator_context()
    candidate = _clone_solution(current_solution)
    truck, drone1, drone2 = candidate
    if len(truck) <= 3:
        return current_solution

    budgets = _search_budgets(len(truck))
    truck_candidates = _best_truck_nodes_for_removal(candidate, budgets["truck_nodes"])
    if not truck_candidates:
        return current_solution

    generated = []
    for rem_idx in truck_candidates:
        node = truck[rem_idx]
        a = truck[rem_idx - 1]
        b = truck[rem_idx + 1]
        truck_gain = truck_times[a][node] + truck_times[node][b] - truck_times[a][b]

        truck_new = truck[:rem_idx] + truck[rem_idx + 1 :]
        drone1_new = _shift_route_after_truck_removal(drone1, rem_idx)
        drone2_new = _shift_route_after_truck_removal(drone2, rem_idx)
        if drone1_new is None or drone2_new is None:
            continue

        # Prefer assigning to the less loaded drone.
        route_order = [(1, drone1_new, drone2_new), (2, drone2_new, drone1_new)]
        route_order.sort(key=lambda x: len(x[1]))
        if random.random() < _EXPLORE_PROB:
            route_order.reverse()

        prefix_new = _prefix_truck_times(truck_new, truck_times)
        for route_id, target_route, other_route in route_order:
            positions = _candidate_positions(len(target_route), budgets["positions"])
            if random.random() < _EXPLORE_PROB:
                random.shuffle(positions)
            for ins_idx in positions:
                pair = build_drone_pair(
                    node,
                    truck_new,
                    target_route,
                    ins_idx,
                    max_neighbors=budgets["pair_neighbors"],
                    pair_explore_prob=0.08,
                )
                if pair is None:
                    continue
                if not _pair_endpoints_available(target_route, pair[0], pair[1]):
                    continue

                new_trip = (node, pair[0], pair[1])
                route_with_trip = target_route[:]
                route_with_trip.insert(ins_idx, new_trip)
                route_with_trip.sort(key=lambda x: (x[1], x[2], x[0]))
                if not drone_route_is_feasible(route_with_trip):
                    continue
                if not _route_endpoint_unique(route_with_trip):
                    continue
                if not drone_route_is_feasible(other_route):
                    continue
                if not _route_endpoint_unique(other_route):
                    continue

                if route_id == 1:
                    new_solution = [truck_new, route_with_trip, other_route]
                    d1_len = len(route_with_trip)
                    d2_len = len(other_route)
                else:
                    new_solution = [truck_new, other_route, route_with_trip]
                    d1_len = len(other_route)
                    d2_len = len(route_with_trip)

                timing_pen = _trip_penalty(node, pair[0], pair[1], truck_new, prefix_new, drone_times)
                balance_pen = 2.0 * abs(d1_len - d2_len)
                score = -truck_gain + timing_pen + balance_pen
                generated.append((score, new_solution))

                if len(generated) >= budgets["max_candidates"]:
                    break
            if len(generated) >= budgets["max_candidates"]:
                break
        if len(generated) >= budgets["max_candidates"]:
            break

    if not generated:
        return current_solution

    generated.sort(key=lambda x: x[0])
    if random.random() < _EXPLORE_PROB:
        return random.choice(generated[: min(3, len(generated))])[1]
    return generated[0][1]

