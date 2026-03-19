import random
from pathlib import Path
import sys

ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
CORE_DIR = ASSIGNMENT_DIR / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.append(str(CORE_DIR))

from operator_context import assert_context_is_set, get_operator_context, set_operator_context
from drone_route_utils import drone_route_is_feasible, pair_fits_in_drone

_TRIP_SCAN_LIMIT = 4
_LAUNCH_WINDOW = 6
_RECOVERY_WINDOW = 6
_INSERT_WINDOW = 2


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


def _trip_penalty(node, launch_idx, land_idx, truck, prefix, truck_times, drone_times):
    launch_node = truck[launch_idx]
    land_node = truck[land_idx]
    truck_launch_time = prefix[launch_idx]
    truck_land_time = prefix[land_idx]
    truck_segment_time = truck_land_time - truck_launch_time
    flight_out = drone_times[launch_node][node]
    flight_back = drone_times[node][land_node]
    sortie_time = flight_out + flight_back

    drone_arrival_time = truck_launch_time + flight_out
    drone_return_time = truck_launch_time + sortie_time

    wait_excess = max(0.0, drone_return_time - truck_land_time)
    sync_mismatch = abs(drone_return_time - truck_land_time)

    # Proxy components:
    # - waiting/synchronization mismatch (dominant)
    # - truck timing impact via absolute launch/land timing
    # - sortie and truck segment lengths
    return (
        3.0 * wait_excess
        + 0.7 * sync_mismatch
        + 0.025 * sortie_time
        + 0.01 * drone_arrival_time
        + 0.002 * truck_segment_time
    )


def _solution_sync_penalty(solution, truck_times, drone_times):
    truck, drone1, drone2 = solution
    prefix = _prefix_truck_times(truck, truck_times)
    pen = 0.0
    for node, launch_idx, land_idx in drone1:
        pen += _trip_penalty(node, launch_idx, land_idx, truck, prefix, truck_times, drone_times)
    for node, launch_idx, land_idx in drone2:
        pen += _trip_penalty(node, launch_idx, land_idx, truck, prefix, truck_times, drone_times)
    pen += 2.0 * abs(len(drone1) - len(drone2))
    return pen


def _nearby_indices(center, low, high, width):
    left = max(low, center - width)
    right = min(high, center + width)
    return list(range(left, right + 1))


def _candidate_insertion_positions(route_len, base_idx):
    low = max(0, base_idx - _INSERT_WINDOW)
    high = min(route_len, base_idx + _INSERT_WINDOW)
    positions = list(range(low, high + 1))
    if 0 not in positions:
        positions.append(0)
    if route_len not in positions:
        positions.append(route_len)
    return sorted(set(positions))


def _candidate_pairs(node, truck, route_wo_trip, insertion_idx, old_launch, old_land, drone_times, flight_limit):
    max_internal = len(truck) - 2
    if max_internal < 1:
        return []

    launch_pool = _nearby_indices(old_launch, 1, max_internal, _LAUNCH_WINDOW)
    recover_pool = _nearby_indices(old_land, 1, max_internal, _RECOVERY_WINDOW)

    pairs = []
    for launch_idx in launch_pool:
        launch_node = truck[launch_idx]
        for land_idx in recover_pool:
            if launch_idx >= land_idx:
                continue
            if not pair_fits_in_drone(route_wo_trip, insertion_idx, launch_idx, land_idx):
                continue
            land_node = truck[land_idx]
            sortie = drone_times[launch_node][node] + drone_times[node][land_node]
            if sortie > flight_limit:
                continue
            pairs.append((launch_idx, land_idx))
    return pairs


def operator(current_solution):
    assert_context_is_set()
    truck_times, drone_times, flight_limit, _ = get_operator_context()

    truck, drone1, drone2 = _clone_solution(current_solution)
    if not drone1 and not drone2:
        return current_solution

    baseline_pen = _solution_sync_penalty(current_solution, truck_times, drone_times)

    trips = []
    for route_id, route in ((1, drone1), (2, drone2)):
        for idx, (node, launch_idx, land_idx) in enumerate(route):
            trips.append((route_id, idx, node, launch_idx, land_idx))

    if not trips:
        return current_solution

    random.shuffle(trips)
    scan_list = trips[: min(_TRIP_SCAN_LIMIT, len(trips))]

    best_candidate = None
    best_pen = baseline_pen

    for route_id, idx, node, old_launch, old_land in scan_list:
        if route_id == 1:
            source_route = drone1
            other_route = drone2
        else:
            source_route = drone2
            other_route = drone1

        if idx < 0 or idx >= len(source_route):
            continue

        route_wo_trip = source_route[:idx] + source_route[idx + 1 :]
        insert_positions = _candidate_insertion_positions(len(route_wo_trip), idx)

        for ins_idx in insert_positions:
            pairs = _candidate_pairs(
                node,
                truck,
                route_wo_trip,
                ins_idx,
                old_launch,
                old_land,
                drone_times,
                flight_limit,
            )
            if not pairs:
                continue

            for launch_idx, land_idx in pairs:
                trial_route = route_wo_trip[:]
                trial_route.insert(ins_idx, (node, launch_idx, land_idx))
                trial_route.sort(key=lambda x: (x[1], x[2], x[0]))

                if not drone_route_is_feasible(trial_route):
                    continue
                if not _route_endpoint_unique(trial_route):
                    continue
                if not drone_route_is_feasible(other_route):
                    continue
                if not _route_endpoint_unique(other_route):
                    continue

                if route_id == 1:
                    candidate = [truck, trial_route, other_route[:]]
                else:
                    candidate = [truck, other_route[:], trial_route]

                new_pen = _solution_sync_penalty(candidate, truck_times, drone_times)
                if new_pen < best_pen:
                    best_pen = new_pen
                    best_candidate = candidate

    if best_candidate is None:
        return current_solution
    return best_candidate
