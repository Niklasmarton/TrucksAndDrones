import random
from pathlib import Path
import sys

CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.append(str(CORE_DIR))

from drone_route_utils import (
    build_drone_pair,
    drone_route_is_feasible,
    remap_drone_route_by_endpoint_nodes,
)
from operator_context import assert_context_is_set, get_operator_context, set_operator_context


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
    return (wait_excess, mismatch, sortie_time)


def _pick_source_trip(solution, prefix, drone_times):
    truck, drone1, drone2 = solution
    candidates = []
    for route_id, route in ((1, drone1), (2, drone2)):
        for trip_idx, trip in enumerate(route):
            bad = _trip_badness(trip, truck, prefix, drone_times)
            rank = (bad[0], bad[1], bad[2])
            candidates.append((rank, route_id, trip_idx))

    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda x: x[0])
    k = min(4, len(candidates))
    _, route_id, trip_idx = random.choice(candidates[:k])
    return route_id, trip_idx


def _best_insert_positions_for_node(truck_route, node, truck_times, top_k=8):
    deltas = []
    for ins_idx in range(1, len(truck_route)):
        a = truck_route[ins_idx - 1]
        b = truck_route[ins_idx]
        delta = truck_times[a][node] + truck_times[node][b] - truck_times[a][b]
        deltas.append((delta, ins_idx))

    deltas.sort(key=lambda x: x[0])
    chosen = [idx for _, idx in deltas[: min(top_k, len(deltas))]]
    random.shuffle(chosen)
    return chosen


def _best_removal_positions(truck_route, truck_times, banned_nodes, focus_idx=None, top_k=12):
    scored = []
    for idx in range(1, len(truck_route) - 1):
        node = truck_route[idx]
        if node in banned_nodes:
            continue
        a = truck_route[idx - 1]
        b = truck_route[idx + 1]
        saving = truck_times[a][node] + truck_times[node][b] - truck_times[a][b]
        if focus_idx is not None:
            saving += max(0.0, 2.0 - 0.2 * abs(idx - focus_idx))
        scored.append((saving, idx))

    if not scored:
        return []
    scored.sort(reverse=True, key=lambda x: x[0])
    out = [idx for _, idx in scored[: min(top_k, len(scored))]]
    random.shuffle(out)
    return out


def _candidate_insertion_indices(route_len, preferred_idx):
    raw = [preferred_idx, preferred_idx - 1, preferred_idx + 1, 0, route_len]
    valid = []
    seen = set()
    for idx in raw:
        if 0 <= idx <= route_len and idx not in seen:
            valid.append(idx)
            seen.add(idx)
    return valid


_MAX_CANDIDATES = 16


def _truck_total_cost(truck, truck_times):
    cost = 0.0
    for i in range(len(truck) - 1):
        cost += truck_times[truck[i]][truck[i + 1]]
    return cost


def operator(current_solution):
    """
    op3: drone-truck mode swap
    - Take one drone-served customer and put it into the truck.
    - Take one truck-served customer and insert it into the same drone route.
    - Collects all valid candidates, scores by truck cost delta, returns best.
    """
    assert_context_is_set()
    truck_times, drone_times, _, depot = get_operator_context()

    candidate = _clone_solution(current_solution)
    truck = candidate[0]
    if len(truck) <= 3:
        return current_solution

    old_truck_cost = _truck_total_cost(truck, truck_times)

    prefix = _prefix_truck_times(truck, truck_times)
    source = _pick_source_trip(candidate, prefix, drone_times)
    if source is None:
        return current_solution
    route_id, trip_idx = source

    source_route = candidate[route_id]
    drone_node, old_launch_idx, old_land_idx = source_route[trip_idx]
    old_launch_node = truck[old_launch_idx]
    old_land_node = truck[old_land_idx]

    # Remove chosen drone trip (customer will be inserted into truck).
    drone1_base = candidate[1][:]
    drone2_base = candidate[2][:]
    if route_id == 1:
        drone1_base.pop(trip_idx)
    else:
        drone2_base.pop(trip_idx)

    candidates = []
    truck_insert_positions = _best_insert_positions_for_node(truck, drone_node, truck_times, top_k=8)
    for ins_idx in truck_insert_positions:
        truck_after_insert = truck[:ins_idx] + [drone_node] + truck[ins_idx:]

        d1_after_insert = remap_drone_route_by_endpoint_nodes(truck, truck_after_insert, drone1_base)
        d2_after_insert = remap_drone_route_by_endpoint_nodes(truck, truck_after_insert, drone2_base)
        if d1_after_insert is None or d2_after_insert is None:
            continue
        if not drone_route_is_feasible(d1_after_insert) or not drone_route_is_feasible(d2_after_insert):
            continue

        removal_positions = _best_removal_positions(
            truck_after_insert,
            truck_times,
            banned_nodes={depot, drone_node},
            focus_idx=ins_idx,
            top_k=12,
        )
        for rem_idx in removal_positions:
            truck_node = truck_after_insert[rem_idx]
            if truck_node == depot or truck_node == drone_node:
                continue

            truck_final = truck_after_insert[:rem_idx] + truck_after_insert[rem_idx + 1 :]

            d1_final = remap_drone_route_by_endpoint_nodes(truck_after_insert, truck_final, d1_after_insert)
            d2_final = remap_drone_route_by_endpoint_nodes(truck_after_insert, truck_final, d2_after_insert)
            if d1_final is None or d2_final is None:
                continue
            if not drone_route_is_feasible(d1_final) or not drone_route_is_feasible(d2_final):
                continue

            target_route = d1_final if route_id == 1 else d2_final
            other_route = d2_final if route_id == 1 else d1_final

            ins_candidates = _candidate_insertion_indices(len(target_route), trip_idx)
            preferred_pair = None
            if old_launch_node in truck_final and old_land_node in truck_final:
                p_launch = truck_final.index(old_launch_node)
                p_land = truck_final.index(old_land_node)
                if p_launch < p_land:
                    preferred_pair = (p_launch, p_land)

            for route_insert_idx in ins_candidates:
                pair = build_drone_pair(
                    truck_node,
                    truck_final,
                    target_route,
                    route_insert_idx,
                    preferred_pair=preferred_pair,
                    max_neighbors=8,
                    pair_explore_prob=0.10,
                )
                if pair is None:
                    continue

                new_target_route = target_route[:]
                new_target_route.insert(route_insert_idx, (truck_node, pair[0], pair[1]))
                if not drone_route_is_feasible(new_target_route):
                    continue

                if route_id == 1:
                    new_solution = [truck_final, new_target_route, other_route]
                else:
                    new_solution = [truck_final, other_route, new_target_route]

                score = _truck_total_cost(truck_final, truck_times) - old_truck_cost
                candidates.append((score, new_solution))

                if len(candidates) >= _MAX_CANDIDATES:
                    break
            if len(candidates) >= _MAX_CANDIDATES:
                break
        if len(candidates) >= _MAX_CANDIDATES:
            break

    if not candidates:
        return current_solution

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]
