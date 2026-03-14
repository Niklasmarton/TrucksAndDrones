import random
from pathlib import Path
import sys

CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.append(str(CORE_DIR))

from drone_route_utils import build_drone_pair, drone_route_is_feasible
from operator_context import assert_context_is_set, get_operator_context, set_operator_context

_EXPLORE_PROB = 0.12


def _clone_solution(solution):
    return [solution[0][:], solution[1][:], solution[2][:]]


def _prefix_truck_times(truck_route, truck_times):
    prefix = [0.0]
    for i in range(len(truck_route) - 1):
        prefix.append(prefix[-1] + truck_times[truck_route[i]][truck_route[i + 1]])
    return prefix


def _trip_badness(trip, truck_route, prefix, drone_times):
    node, launch_idx, land_idx = trip
    launch_node = truck_route[launch_idx]
    land_node = truck_route[land_idx]
    sortie_time = drone_times[launch_node][node] + drone_times[node][land_node]
    truck_time = prefix[land_idx] - prefix[launch_idx]
    wait_excess = max(0.0, sortie_time - truck_time)
    mismatch = abs(sortie_time - truck_time)
    return wait_excess, mismatch, sortie_time


def _trip_penalty(trip, truck_route, prefix, drone_times):
    wait_excess, mismatch, sortie_time = _trip_badness(trip, truck_route, prefix, drone_times)
    return 3.0 * wait_excess + 0.6 * mismatch + 0.02 * sortie_time


def _search_budgets(truck_len):
    n_customers = max(0, truck_len - 2)
    if n_customers >= 100:
        return {
            "source": 2,
            "positions": 3,
            "max_candidates": 12,
            "pair_neighbors": 6,
        }
    if n_customers >= 50:
        return {
            "source": 3,
            "positions": 4,
            "max_candidates": 20,
            "pair_neighbors": 7,
        }
    return {
        "source": 4,
        "positions": 5,
        "max_candidates": 30,
        "pair_neighbors": 8,
    }


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


def _candidate_positions(route_len, anchor_idx, max_positions):
    raw = [anchor_idx, anchor_idx - 1, anchor_idx + 1, 0, route_len]
    out = []
    seen = set()
    for idx in raw:
        if 0 <= idx <= route_len and idx not in seen:
            out.append(idx)
            seen.add(idx)

    while len(out) < min(max_positions, route_len + 1):
        idx = random.randint(0, route_len)
        if idx not in seen:
            out.append(idx)
            seen.add(idx)
    return out[:max_positions]


def _pick_source_trips(solution, truck_times, drone_times, top_k):
    truck, drone1, drone2 = solution
    prefix = _prefix_truck_times(truck, truck_times)
    scored = []
    for route_id, route in ((1, drone1), (2, drone2)):
        for trip_idx, trip in enumerate(route):
            penalty = _trip_penalty(trip, truck, prefix, drone_times)
            scored.append((penalty, route_id, trip_idx))
    if not scored:
        return []
    scored.sort(reverse=True, key=lambda x: x[0])
    limit = min(top_k, len(scored))
    if random.random() < _EXPLORE_PROB:
        chosen = random.sample(scored[:limit], k=limit)
    else:
        chosen = scored[:limit]
    return [(route_id, trip_idx) for _, route_id, trip_idx in chosen]


def _solution_penalty(solution, truck_times, drone_times):
    truck, drone1, drone2 = solution
    prefix = _prefix_truck_times(truck, truck_times)

    pen = 0.0
    for trip in drone1:
        pen += _trip_penalty(trip, truck, prefix, drone_times)
    for trip in drone2:
        pen += _trip_penalty(trip, truck, prefix, drone_times)
    pen += 2.0 * abs(len(drone1) - len(drone2))
    return pen


def operator(current_solution):
    """
    op4: drone retiming
    - Re-optimizes launch/land indices for existing drone trips.
    - Mostly keeps trip on same drone; may migrate trip to other drone if useful.
    - Does not modify truck route.
    """
    assert_context_is_set()
    truck_times, drone_times, _, _ = get_operator_context()

    sol = _clone_solution(current_solution)
    truck, drone1, drone2 = sol
    if not drone1 and not drone2:
        return current_solution

    budgets = _search_budgets(len(truck))
    source_trips = _pick_source_trips(sol, truck_times, drone_times, budgets["source"])
    if not source_trips:
        return current_solution

    current_penalty = _solution_penalty(sol, truck_times, drone_times)
    candidates = []

    for route_id, trip_idx in source_trips:
        if route_id == 1:
            source_route = drone1
            other_route = drone2
        else:
            source_route = drone2
            other_route = drone1
        if trip_idx < 0 or trip_idx >= len(source_route):
            continue

        node, old_launch, old_land = source_route[trip_idx]
        source_base = source_route[:]
        source_base.pop(trip_idx)

        target_ids = [route_id]
        if len(source_route) > len(other_route) + 1 or random.random() < 0.25:
            target_ids.append(2 if route_id == 1 else 1)

        for target_id in target_ids:
            if target_id == 1:
                target_route = source_base if route_id == 1 else drone1[:]
                non_target_route = drone2[:] if route_id == 1 else source_base
            else:
                target_route = source_base if route_id == 2 else drone2[:]
                non_target_route = drone1[:] if route_id == 2 else source_base

            anchor = min(trip_idx, len(target_route))
            ins_positions = _candidate_positions(len(target_route), anchor, budgets["positions"])
            if random.random() < _EXPLORE_PROB:
                random.shuffle(ins_positions)

            for ins_idx in ins_positions:
                preferred_pair = (old_launch, old_land) if target_id == route_id else None
                pair = build_drone_pair(
                    node,
                    truck,
                    target_route,
                    ins_idx,
                    preferred_pair=preferred_pair,
                    max_neighbors=budgets["pair_neighbors"],
                    pair_explore_prob=0.08,
                )
                if pair is None:
                    continue
                if not _pair_endpoints_available(target_route, pair[0], pair[1]):
                    continue

                new_target = target_route[:]
                new_target.insert(ins_idx, (node, pair[0], pair[1]))
                new_target.sort(key=lambda x: (x[1], x[2], x[0]))
                if not drone_route_is_feasible(new_target):
                    continue
                if not _route_endpoint_unique(new_target):
                    continue
                if not drone_route_is_feasible(non_target_route):
                    continue
                if not _route_endpoint_unique(non_target_route):
                    continue

                if target_id == 1:
                    new_solution = [truck, new_target, non_target_route]
                else:
                    new_solution = [truck, non_target_route, new_target]

                new_penalty = _solution_penalty(new_solution, truck_times, drone_times)
                candidates.append((new_penalty - current_penalty, new_solution))
                if len(candidates) >= budgets["max_candidates"]:
                    break
            if len(candidates) >= budgets["max_candidates"]:
                break
        if len(candidates) >= budgets["max_candidates"]:
            break

    if not candidates:
        return current_solution

    candidates.sort(key=lambda x: x[0])
    if random.random() < _EXPLORE_PROB:
        top_k = min(3, len(candidates))
        return random.choice(candidates[:top_k])[1]
    return candidates[0][1]
