"""
op15 — cross-drone relocate

Moves one drone trip from drone1 to drone2 (or vice versa), changing
the partition of which customers are served by each drone.

Why this fills a gap:
  Every current operator leaves the drone partition essentially fixed:
    - op12 swaps truck↔drone (partition between truck and drones)
    - op13 retunes launch/land pairs within a fixed drone assignment
    - op2/op14 can change assignments indirectly, but only as a side-effect
  No operator directly moves a customer from drone1 to drone2 or vice versa.
  op15 is the only operator that specifically explores this dimension.

Selection strategy:
  Score every drone trip by sync quality (hover penalty + mismatch, same
  formula as op13).  Pick the worst-synced trip first — it is most likely
  to benefit from the different launch/land opportunities available from
  its new position in the other drone route.

  Try up to MAX_ATTEMPTS worst trips.  Return the first successful move;
  ALNS handles whether the objective improvement justifies acceptance.

Feasibility:
  After removal, the source route is re-checked with drone_route_is_feasible.
  The target route is rebuilt with build_drone_pair at every insertion
  position and validated before returning.
"""
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


def _trip_sync_score(node, launch_idx, land_idx, truck, T, D):
    """Lower = better sync. Matches op13's scoring formula."""
    if not (0 <= launch_idx < land_idx < len(truck)):
        return float("inf")
    lnode = truck[launch_idx]
    rnode = truck[land_idx]
    drone_flight = D[lnode][node] + D[node][rnode]
    truck_leg = sum(T[truck[i]][truck[i + 1]] for i in range(launch_idx, land_idx))
    hover_penalty = max(0.0, truck_leg - drone_flight)
    mismatch = abs(drone_flight - truck_leg)
    return 2.0 * hover_penalty + 0.5 * mismatch + 0.01 * drone_flight


def _try_insert_into_route(node, target_route, truck):
    """
    Try inserting node into target_route at every position.
    Returns updated target_route on success, None on failure.
    """
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

    # Need both routes non-empty — a move requires a source and a target.
    if not drone1 or not drone2:
        return current_solution

    # Score all trips from both drones, worst-first.
    scored = []
    for route_id, route in ((1, drone1), (2, drone2)):
        for node, l, r in route:
            score = _trip_sync_score(node, l, r, truck, T, D)
            scored.append((score, route_id, node))

    if not scored:
        return current_solution

    scored.sort(reverse=True)

    # Optionally shuffle within the top pool for exploration.
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

        # Remove from source route.
        new_source = [t for t in source if t[0] != node]

        # Source must still be valid after removal.
        if not drone_route_is_feasible(new_source):
            continue

        # Try inserting into target route.
        new_target = _try_insert_into_route(node, target, truck)
        if new_target is None:
            continue

        if route_id == 1:
            return [truck[:], new_source, new_target]
        else:
            return [truck[:], new_target, new_source]

    return current_solution
