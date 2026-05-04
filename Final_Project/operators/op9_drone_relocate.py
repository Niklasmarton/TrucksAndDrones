import random
from pathlib import Path
import sys

ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
CORE_DIR = ASSIGNMENT_DIR / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.append(str(CORE_DIR))

from operator_context import assert_context_is_set, get_operator_context
from drone_route_utils import build_drone_pair, drone_route_is_feasible

_MAX_ATTEMPTS = 3
_EXPLORE_PROB = 0.15


def set_search_progress(progress):
    return None


def _clone_solution(solution):
    return [solution[0][:], solution[1][:], solution[2][:]]


def _route_endpoint_unique(route):
    used_l = set()
    used_r = set()
    for _, l, r in route:
        if l in used_l or r in used_r:
            return False
        used_l.add(l)
        used_r.add(r)
    return True


# lower = better drone/truck sync
def _trip_sync_score(node, launch_idx, land_idx, truck, T, D):
    if not (0 <= launch_idx < land_idx < len(truck)):
        return float("inf")
    lnode = truck[launch_idx]
    rnode = truck[land_idx]
    drone_flight = D[lnode][node] + D[node][rnode]
    truck_leg = sum(T[truck[i]][truck[i + 1]] for i in range(launch_idx, land_idx))
    hover_penalty = max(0.0, truck_leg - drone_flight)
    mismatch = abs(drone_flight - truck_leg)
    return 2.0 * hover_penalty + 0.5 * mismatch + 0.01 * drone_flight


# try every insertion position, returns the updated route or None
def _try_insert_into_route(node, target_route, truck):
    positions = list(range(len(target_route) + 1))
    if random.random() < _EXPLORE_PROB:
        random.shuffle(positions)

    for ins_idx in positions:
        pair = build_drone_pair(
            node,
            truck,
            target_route,
            ins_idx,
            max_neighbors=10,
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

        return new_target

    return None


def operator(current_solution):
    assert_context_is_set()
    T, D, flight_limit, _ = get_operator_context()

    truck, drone1, drone2 = current_solution

    # need both routes non-empty, a move needs a source and a target
    if not drone1 or not drone2:
        return current_solution

    # rank trips on both drones, worst sync first
    scored = []
    for route_id, route in ((1, drone1), (2, drone2)):
        for node, l, r in route:
            score = _trip_sync_score(node, l, r, truck, T, D)
            scored.append((score, route_id, node))

    if not scored:
        return current_solution

    scored.sort(reverse=True)

    # occasional shuffle within the top pool to explore
    if random.random() < _EXPLORE_PROB:
        top_k = min(len(scored), max(_MAX_ATTEMPTS * 2, 4))
        pool = scored[:top_k]
        random.shuffle(pool)
        candidates = pool + scored[top_k:]
    else:
        candidates = scored

    for _, route_id, node in candidates[:_MAX_ATTEMPTS]:
        source = drone1 if route_id == 1 else drone2
        target = drone2 if route_id == 1 else drone1

        # drop the node from its source route
        new_source = [t for t in source if t[0] != node]

        if not drone_route_is_feasible(new_source):
            continue

        new_target = _try_insert_into_route(node, target, truck)
        if new_target is None:
            continue

        if route_id == 1:
            return [truck[:], new_source, new_target]
        else:
            return [truck[:], new_target, new_source]

    return current_solution
