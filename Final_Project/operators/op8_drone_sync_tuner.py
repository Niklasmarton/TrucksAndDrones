import random
from pathlib import Path
import sys

ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
CORE_DIR = ASSIGNMENT_DIR / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.append(str(CORE_DIR))

from operator_context import assert_context_is_set, get_operator_context
from drone_route_utils import drone_route_is_feasible

_MAX_TRIPS_PER_CALL = 3
_EXPLORE_PROB = 0.15


def set_search_progress(progress):
    return None


def _clone_solution(solution):
    return [solution[0][:], solution[1][:], solution[2][:]]


def _route_endpoint_unique(route):
    used_l = set(); used_r = set()
    for _, l, r in route:
        if l in used_l or r in used_r:
            return False
        used_l.add(l); used_r.add(r)
    return True


# lower = better sync. penalises drone hovering (truck slow) and big flight/leg mismatch
def _trip_sync_score(node, launch_idx, land_idx, truck, T, D):
    if not (0 <= launch_idx < land_idx < len(truck)):
        return float("inf")
    lnode = truck[launch_idx]
    rnode = truck[land_idx]
    drone_flight = D[lnode][node] + D[node][rnode]
    truck_leg = sum(T[truck[i]][truck[i + 1]] for i in range(launch_idx, land_idx))
    # drone arrives early and waits, that wait counts against the flight range
    hover_penalty = max(0.0, truck_leg - drone_flight)
    mismatch = abs(drone_flight - truck_leg)
    return 2.0 * hover_penalty + 0.5 * mismatch + 0.01 * drone_flight


# all (launch, land) pairs where the drone trip stays within flight_limit
def _all_feasible_pairs(node, truck, T, D, flight_limit, exclude_l=None, exclude_r=None):
    exclude_l = exclude_l or set()
    exclude_r = exclude_r or set()
    nt = len(truck)
    pairs = []
    for l in range(nt - 1):
        if l in exclude_l:
            continue
        for r in range(l + 1, nt):
            if r in exclude_r:
                continue
            ft = D[truck[l]][node] + D[node][truck[r]]
            if ft <= flight_limit:
                pairs.append((l, r, ft))
    return pairs


# try every (launch, land) pair for one trip and pick the one with the best sync score
def _try_retune_trip(route_id, trip_pos, solution, T, D, flight_limit):
    truck, drone1, drone2 = solution
    route = drone1 if route_id == 1 else drone2
    other = drone2 if route_id == 1 else drone1

    node, old_l, old_r = route[trip_pos]

    # endpoints already taken by the other trips in this route
    used_l = {l for i, (_, l, _) in enumerate(route) if i != trip_pos}
    used_r = {r for i, (_, _, r) in enumerate(route) if i != trip_pos}

    pairs = _all_feasible_pairs(node, truck, T, D, flight_limit,
                                exclude_l=used_l, exclude_r=used_r)
    if not pairs:
        return None

    current_score = _trip_sync_score(node, old_l, old_r, truck, T, D)

    best_score = current_score
    best_pair = None

    if random.random() < _EXPLORE_PROB:
        random.shuffle(pairs)

    for (l, r, _) in pairs:
        if l == old_l and r == old_r:
            continue
        score = _trip_sync_score(node, l, r, truck, T, D)
        if score < best_score:
            best_score = score
            best_pair = (l, r)

    if best_pair is None:
        return None

    new_route = route[:]
    new_route[trip_pos] = (node, best_pair[0], best_pair[1])
    new_route.sort(key=lambda x: (x[1], x[2], x[0]))

    if not drone_route_is_feasible(new_route):
        return None
    if not _route_endpoint_unique(new_route):
        return None

    if route_id == 1:
        new_drone1, new_drone2 = new_route, drone2
    else:
        new_drone1, new_drone2 = drone1, new_route

    if not drone_route_is_feasible(new_drone1) or not drone_route_is_feasible(new_drone2):
        return None

    return [truck, new_drone1, new_drone2]


def operator(current_solution):
    assert_context_is_set()
    T, D, flight_limit, _ = get_operator_context()

    truck, drone1, drone2 = current_solution
    if not drone1 and not drone2:
        return current_solution

    # score every trip across both drones
    scored = []
    for route_id, route in ((1, drone1), (2, drone2)):
        for trip_pos, (node, l, r) in enumerate(route):
            score = _trip_sync_score(node, l, r, truck, T, D)
            scored.append((score, route_id, trip_pos))

    if not scored:
        return current_solution

    # worst trips first, with a bit of shuffle for exploration
    scored.sort(reverse=True)
    if random.random() < _EXPLORE_PROB:
        top_k = min(len(scored), max(_MAX_TRIPS_PER_CALL * 2, 4))
        candidates = scored[:top_k]
        random.shuffle(candidates)
    else:
        candidates = scored

    solution = _clone_solution(current_solution)
    improved = False

    for _, route_id, trip_pos in candidates[:_MAX_TRIPS_PER_CALL]:
        result = _try_retune_trip(route_id, trip_pos, solution, T, D, flight_limit)
        if result is not None:
            solution = result
            improved = True

    return solution if improved else current_solution
