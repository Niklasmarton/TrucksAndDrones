import random
from pathlib import Path
import sys

ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
CORE_DIR = ASSIGNMENT_DIR / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.append(str(CORE_DIR))

from operator_context import assert_context_is_set, get_operator_context
from drone_route_utils import (
    build_drone_pair,
    drone_route_is_feasible,
)

_MAX_ATTEMPTS = 6
_EXPLORE_PROB = 0.20


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


def _truck_removal_gain(truck, idx, T):
    a, node, b = truck[idx - 1], truck[idx], truck[idx + 1]
    return T[a][node] + T[node][b] - T[a][b]


def _best_truck_insert_delta(node, truck, T):
    best_delta = None
    best_idx = None
    for ins_idx in range(1, len(truck)):
        a, b = truck[ins_idx - 1], truck[ins_idx]
        delta = T[a][node] + T[node][b] - T[a][b]
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_idx = ins_idx
    return best_delta, best_idx


# how out of sync the trip is, lower is better
def _drone_sync_penalty(node, launch_idx, land_idx, truck, T, D):
    if not (0 <= launch_idx < len(truck) and 0 <= land_idx < len(truck)):
        return float("inf")
    launch_node = truck[launch_idx]
    land_node = truck[land_idx]
    sortie = D[launch_node][node] + D[node][land_node]
    truck_seg = sum(T[truck[i]][truck[i + 1]] for i in range(launch_idx, land_idx))
    return 3.0 * max(0.0, sortie - truck_seg) + 0.6 * abs(sortie - truck_seg)


def _used_endpoints(drone1, drone2):
    used = set()
    for route in (drone1, drone2):
        for _, l, r in route:
            used.add(l)
            used.add(r)
    return used


# truck customers ranked by removal gain
def _pick_truck_candidate(truck, drone1, drone2, T):
    used_ep = _used_endpoints(drone1, drone2)
    scored = []
    for idx in range(1, len(truck) - 1):
        if idx in used_ep:
            # can't pull a node that another drone trip uses as an endpoint
            continue
        gain = _truck_removal_gain(truck, idx, T)
        scored.append((gain, idx))
    if not scored:
        return []
    scored.sort(reverse=True)
    return scored


# drone trips ranked by sync penalty, worst first
def _pick_drone_candidate(drone1, drone2, truck, T, D):
    scored = []
    for route_id, route in ((1, drone1), (2, drone2)):
        for trip_idx, (node, launch_idx, land_idx) in enumerate(route):
            pen = _drone_sync_penalty(node, launch_idx, land_idx, truck, T, D)
            scored.append((pen, node, route_id, trip_idx))
    scored.sort(reverse=True)
    return scored


def _attempt_swap(current_solution, T, D, flight_limit):
    truck, drone1, drone2 = _clone_solution(current_solution)

    if len(truck) <= 3:  # need at least one free truck customer
        return None
    if not drone1 and not drone2:
        return None

    truck_candidates = _pick_truck_candidate(truck, drone1, drone2, T)
    drone_candidates = _pick_drone_candidate(drone1, drone2, truck, T, D)
    if not truck_candidates or not drone_candidates:
        return None

    # rank-biased pick from the top truck candidates
    k_t = min(4, len(truck_candidates))
    if random.random() < _EXPLORE_PROB:
        _, truck_idx = random.choice(truck_candidates[:max(k_t, 1)])
    else:
        weights = [k_t - i for i in range(k_t)]
        _, truck_idx = random.choices(truck_candidates[:k_t], weights=weights, k=1)[0]

    truck_node = truck[truck_idx]

    # same for drone candidates
    k_d = min(4, len(drone_candidates))
    if random.random() < _EXPLORE_PROB:
        _, drone_node, route_id, trip_idx = random.choice(drone_candidates[:max(k_d, 1)])
    else:
        weights = [k_d - i for i in range(k_d)]
        _, drone_node, route_id, trip_idx = random.choices(
            drone_candidates[:k_d], weights=weights, k=1
        )[0]

    # pull truck_node out
    truck.pop(truck_idx)

    # cheapest truck spot for the drone_node we'll move up to the truck
    _, ins_idx = _best_truck_insert_delta(drone_node, truck, T)
    if ins_idx is None:
        return None
    truck.insert(ins_idx, drone_node)

    # drone routes still reference old truck indices, remap them after pop+insert
    def _remap_idx(idx, pop_idx, insert_idx):
        if idx > pop_idx:
            idx -= 1
        if idx >= insert_idx:
            idx += 1
        return idx

    def _remap_route(route):
        remapped = []
        for node, l, r in route:
            remapped.append((node, _remap_idx(l, truck_idx, ins_idx),
                             _remap_idx(r, truck_idx, ins_idx)))
        remapped.sort(key=lambda x: (x[1], x[2], x[0]))
        return remapped

    drone1 = _remap_route(drone1)
    drone2 = _remap_route(drone2)

    # drop drone_node from its route. find by node id, the sort may have shifted trip_idx
    target_route = drone1 if route_id == 1 else drone2
    trip_pos = next((i for i, t in enumerate(target_route) if t[0] == drone_node), None)
    if trip_pos is None:
        return None
    target_route.pop(trip_pos)

    # try to put truck_node onto a drone, same route first to keep balance
    other_route = drone2 if route_id == 1 else drone1
    inserted = False
    for candidate_route in (target_route, other_route):
        for ins_pos in range(len(candidate_route) + 1):
            pair = build_drone_pair(
                truck_node,
                truck,
                candidate_route,
                ins_pos,
                max_neighbors=8,
                pair_explore_prob=0.10,
            )
            if pair is None:
                continue
            launch_idx, land_idx = pair
            used_l = {l for _, l, _ in candidate_route}
            used_r = {r for _, _, r in candidate_route}
            if launch_idx in used_l or land_idx in used_r:
                continue
            candidate_route.insert(ins_pos, (truck_node, launch_idx, land_idx))
            candidate_route.sort(key=lambda x: (x[1], x[2], x[0]))
            if not drone_route_is_feasible(candidate_route):
                candidate_route.pop(
                    next(i for i, t in enumerate(candidate_route) if t[0] == truck_node)
                )
                continue
            inserted = True
            break
        if inserted:
            break

    if not inserted:
        return None

    # final sanity check on both routes
    if not drone_route_is_feasible(drone1) or not drone_route_is_feasible(drone2):
        return None
    if not _route_endpoint_unique(drone1) or not _route_endpoint_unique(drone2):
        return None

    return [truck, drone1, drone2]


def operator(current_solution):
    assert_context_is_set()
    T, D, flight_limit, _ = get_operator_context()

    for _ in range(_MAX_ATTEMPTS):
        result = _attempt_swap(current_solution, T, D, flight_limit)
        if result is not None and result != current_solution:
            return result

    return current_solution
