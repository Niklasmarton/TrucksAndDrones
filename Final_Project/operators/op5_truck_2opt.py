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

_TOP_K_ANCHORS = 18
_PAIR_TRIALS = 72
_EXPLORE_PROB = 0.10
_SYNC_PEN_WEIGHT = 0.10


def _total_arrival_time(truck, drone1, drone2, truck_times, drone_times, flight_range, depot):
    """Full objective: sum of all customer arrival times / 100. Mirrors
    CalCulateTotalArrivalTime semantics. Returns inf if infeasible."""
    n = len(truck)
    if n < 2:
        return float("inf")
    drone_return_map = {}
    for drone_id, route in enumerate([drone1, drone2]):
        for customer, launch_idx, return_idx in route:
            if not (0 <= launch_idx < n and 0 <= return_idx < n and launch_idx < return_idx):
                return float("inf")
            drone_return_map.setdefault(return_idx, []).append((drone_id, launch_idx, customer))
    t_arrival = {truck[0]: 0.0}
    t_departure = {truck[0]: 0.0}
    drone_availability = [0.0, 0.0]
    total = 0.0
    for i in range(1, n):
        prev_node = truck[i - 1]
        curr_node = truck[i]
        truck_arrival_time = t_departure[prev_node] + truck_times[prev_node][curr_node]
        t_arrival[curr_node] = truck_arrival_time
        drone_returns = []
        for drone_id, launch_idx, customer in drone_return_map.get(i, []):
            launch_node = truck[launch_idx]
            flight_out = drone_times[launch_node][customer]
            flight_back = drone_times[customer][curr_node]
            total_flight = flight_out + flight_back
            actual_launch = max(t_arrival[launch_node], drone_availability[drone_id])
            drone_cust_arrival = actual_launch + flight_out
            drone_return_time = actual_launch + total_flight
            drone_availability[drone_id] = drone_return_time
            drone_returns.append(drone_return_time)
            total += drone_cust_arrival
            if curr_node != depot:
                drone_wait = max(truck_arrival_time - drone_return_time, 0.0)
            else:
                drone_wait = 0.0
            if total_flight + drone_wait > flight_range:
                return float("inf")
        if drone_returns:
            t_departure[curr_node] = max(truck_arrival_time, max(drone_returns))
        else:
            t_departure[curr_node] = truck_arrival_time
        if curr_node != depot:
            total += truck_arrival_time
    return total / 100.0


def _reoptimize_drone_windows(truck, drone_route, truck_times, drone_times, flight_range,
                              prev_land_idx_floor=0, next_launch_idx_ceil=None):
    """For each trip in drone_route (in order), pick the (launch_idx, land_idx)
    that minimizes the per-trip sync penalty on the new truck route, while
    respecting the order constraint (each trip's land_idx < next trip's launch_idx).

    Returns the reoptimized drone_route list (or original if infeasible)."""
    n = len(truck)
    if n <= 3 or not drone_route:
        return drone_route

    if next_launch_idx_ceil is None:
        next_launch_idx_ceil = n - 1

    prefix = [0.0]
    for i in range(n - 1):
        prefix.append(prefix[-1] + truck_times[truck[i]][truck[i + 1]])

    new_route = []
    floor = prev_land_idx_floor
    remaining = list(drone_route)
    for k, (cust, _old_launch, _old_land) in enumerate(remaining):
        # Reserve room: each later trip needs at least (#remaining_after - 1)
        # ascending land slots after this one's land_idx.
        remaining_after = len(remaining) - k - 1
        ceil_for_land = next_launch_idx_ceil - remaining_after
        best_pair = None
        best_score = float("inf")
        for launch_idx in range(floor, n - 1):
            if launch_idx >= ceil_for_land:
                break
            launch_node = truck[launch_idx]
            for land_idx in range(launch_idx + 1, ceil_for_land + 1):
                land_node = truck[land_idx]
                trip = drone_times[launch_node][cust] + drone_times[cust][land_node]
                if trip > flight_range:
                    continue
                truck_time = prefix[land_idx] - prefix[launch_idx]
                wait_excess = max(0.0, trip - truck_time)
                mismatch = abs(trip - truck_time)
                score = 3.0 * wait_excess + 0.6 * mismatch + 0.02 * trip
                if score < best_score:
                    best_score = score
                    best_pair = (launch_idx, land_idx)
        if best_pair is None:
            return None
        new_route.append((cust, best_pair[0], best_pair[1]))
        floor = best_pair[1] + 1
    return new_route


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


def _node_badness(truck, idx, truck_times):
    if idx <= 0 or idx >= len(truck) - 1:
        return -1e18
    a = truck[idx - 1]
    b = truck[idx]
    c = truck[idx + 1]
    return truck_times[a][b] + truck_times[b][c] - truck_times[a][c]


def _rank_biased_pick(scored):
    if not scored:
        return None
    k = min(_TOP_K_ANCHORS, len(scored))
    if k <= 1:
        return scored[0]
    weights = [k - i for i in range(k)]
    return random.choices(scored[:k], weights=weights, k=1)[0]


def _quick_2opt_delta(truck, i, j, truck_times):
    a = truck[i - 1]
    b = truck[i]
    c = truck[j]
    d = truck[j + 1]
    old_cost = truck_times[a][b] + truck_times[c][d]
    new_cost = truck_times[a][c] + truck_times[b][d]
    return new_cost - old_cost


def _sample_2opt_pairs(truck, truck_times):
    internal = list(range(1, len(truck) - 1))
    if len(internal) < 3:
        return []

    scored = [(_node_badness(truck, idx, truck_times), idx) for idx in internal]
    scored.sort(key=lambda x: x[0], reverse=True)
    pool = scored[: min(_TOP_K_ANCHORS, len(scored))]
    if len(pool) < 2:
        return []

    pairs = []
    seen = set()
    attempts = 0
    while attempts < (_PAIR_TRIALS * 4) and len(pairs) < _PAIR_TRIALS:
        attempts += 1
        p1 = _rank_biased_pick(pool)
        p2 = _rank_biased_pick(pool)
        if p1 is None or p2 is None:
            continue
        i = p1[1]
        j = p2[1]
        if i == j:
            continue
        i, j = (i, j) if i < j else (j, i)
        if i <= 0 or j >= len(truck) - 1:
            continue
        key = (i, j)
        if key in seen:
            continue
        seen.add(key)
        qdelta = _quick_2opt_delta(truck, i, j, truck_times)
        pairs.append((qdelta, i, j))

    if not pairs:
        return []
    pairs.sort(key=lambda x: x[0])
    return pairs


def _apply_2opt(truck, i, j):
    if i <= 0 or j >= len(truck) - 1 or i >= j:
        return None
    return truck[:i] + list(reversed(truck[i : j + 1])) + truck[j + 1 :]


def operator(current_solution):
    assert_context_is_set()
    truck_times, drone_times, flight_range, depot = get_operator_context()

    truck, drone1, drone2 = _clone_solution(current_solution)
    if len(truck) <= 5:
        return current_solution

    old_obj = _total_arrival_time(truck, drone1, drone2, truck_times, drone_times,
                                  flight_range, depot)

    sampled_pairs = _sample_2opt_pairs(truck, truck_times)
    if not sampled_pairs:
        return current_solution

    candidates = []
    for _, i, j in sampled_pairs:
        new_truck = _apply_2opt(truck, i, j)
        if new_truck is None or new_truck == truck:
            continue

        # Reoptimize drone windows on the new truck route. This is the key
        # change vs the previous version, which just remapped existing
        # endpoints — that missed all wins where the reversal *enabled*
        # a better launch/return pair.
        drone1_new = _reoptimize_drone_windows(new_truck, drone1, truck_times,
                                               drone_times, flight_range)
        drone2_new = _reoptimize_drone_windows(new_truck, drone2, truck_times,
                                               drone_times, flight_range)
        if drone1_new is None or drone2_new is None:
            # Reoptimization failed — fall back to remap, then evaluate.
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

        new_obj = _total_arrival_time(new_truck, drone1_new, drone2_new,
                                      truck_times, drone_times, flight_range, depot)
        if new_obj == float("inf"):
            continue
        candidates.append((new_obj, [new_truck, drone1_new, drone2_new]))

    if not candidates:
        return current_solution

    candidates.sort(key=lambda x: x[0])
    # Only keep candidates that match-or-beat current objective; otherwise
    # this operator would actively worsen the solution.
    if candidates[0][0] >= old_obj:
        return current_solution
    if random.random() < _EXPLORE_PROB:
        # Among improving candidates only.
        improving = [c for c in candidates if c[0] < old_obj]
        top_k = min(5, len(improving))
        return random.choice(improving[:top_k])[1]
    return candidates[0][1]
