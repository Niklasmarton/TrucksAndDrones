"""
This operator is a destroy-and-repair operator.

It removes a small set of truck customers and then reinserts them one by one, either back into the truck route or into a drone route. 

The idea was to make a larger neighborhood move than simple reinsertion, and to move around nodes with bad edges to better spots to hopefully decrease delivery time. The search can restructure parts of the solution and escape local patterns while still keeping feasibility. The intention of the operator is thus escaping bad neighboroods. 

For removal, the operator identifies truck nodes with high removal gain (nodes that contribute a lot to travel cost in their current position), excludes nodes currently used as drone launch/landing endpoints, and samples from a top candidate pool with randomness. This keeps the move targeted but not deterministic.

After removal, the drone routes are index-shifted to stay consistent with the changed truck route. If shifting creates endpoint conflicts, the move is discarded.

If finds the best truck insertion based on greedy truck delta. As for drone insertions, it finds feasible drone routes for the drone by creating launch and landing pairs. The drone pairs get scores based on a low waiting time for the truck, if the drone is slower than the truck on that route and if the drones have about the same workload. It tries to avoid drone trips that is bad for truck-drone synchronization or tries to overload one drone. From that point, the insertion is done greedily. 

It also has an explore probability that allows it to sometimes choose random insertions among top candidates. 

The destroy size is set pretty low (only 3 for the biggest datasets). That is because I wanted it to be strong enough to change up the solution but increase feasibility. 

The operator is also phase-aware, meaning it is prioritized more later in the run. 
"""
import random
from pathlib import Path
import sys

CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.append(str(CORE_DIR))

from drone_route_utils import build_drone_pair, drone_route_is_feasible
from operator_context import assert_context_is_set, get_operator_context, set_operator_context

_EXPLORE_PROB = 0.15
_SEARCH_PROGRESS = 0.5


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def set_search_progress(progress):
    global _SEARCH_PROGRESS
    _SEARCH_PROGRESS = _clamp01(progress)


def _clone_solution(solution):
    return [solution[0][:], solution[1][:], solution[2][:]]


def _rank_biased_pick(candidates, top_k):
    if not candidates:
        return None
    k = min(top_k, len(candidates))
    if k <= 1:
        return candidates[0]
    weights = [k - i for i in range(k)]
    return random.choices(candidates[:k], weights=weights, k=1)[0]


def _route_endpoint_unique(route):
    used_launch = set()
    used_land = set()
    for _, launch_idx, land_idx in route:
        if launch_idx in used_launch or land_idx in used_land:
            return False
        used_launch.add(launch_idx)
        used_land.add(land_idx)
    return True


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
    if n_customers <= 10:
        return 1
    if n_customers <= 20:
        return 2
    return 3


def _pick_destroy_indices(truck, drone1, drone2, truck_times, depot, count):
    used_endpoints = _used_endpoints(drone1, drone2)
    scored = []
    for idx in range(1, len(truck) - 1):
        if idx in used_endpoints:
            continue
        node = truck[idx]
        if node == depot:
            continue
        a = truck[idx - 1]
        b = truck[idx + 1]
        gain = truck_times[a][node] + truck_times[node][b] - truck_times[a][b]
        scored.append((gain, idx))

    if not scored:
        return []

    scored.sort(reverse=True, key=lambda x: x[0])
    pool_size = min(len(scored), max(count, 3 * count))
    pool = scored[:pool_size]
    if random.random() < _EXPLORE_PROB:
        prefix_k = min(len(pool), max(count * 2, count + 1))
        prefix = pool[:prefix_k]
        random.shuffle(prefix)
        pool[:prefix_k] = prefix

    chosen = []
    work = pool[:]
    top_k = min(len(work), max(count * 2, 4))
    while work and len(chosen) < count:
        picked = _rank_biased_pick(work, top_k=top_k)
        if picked is None:
            break
        _, idx = picked
        chosen.append(idx)
        work.remove(picked)
        top_k = min(len(work), max(count * 2, 4))
    return sorted(chosen)


def _best_truck_insert(node, truck, truck_times):
    best_delta = None
    best_idx = None
    for ins_idx in range(1, len(truck)):
        a = truck[ins_idx - 1]
        b = truck[ins_idx]
        delta = truck_times[a][node] + truck_times[node][b] - truck_times[a][b]
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_idx = ins_idx
    return best_delta, best_idx


def _drone_trip_penalty(node, pair, truck, truck_times, drone_times):
    launch_idx, land_idx = pair
    launch_node = truck[launch_idx]
    land_node = truck[land_idx]
    sortie_time = drone_times[launch_node][node] + drone_times[node][land_node]

    truck_leg = 0.0
    for i in range(launch_idx, land_idx):
        truck_leg += truck_times[truck[i]][truck[i + 1]]

    wait_excess = max(0.0, sortie_time - truck_leg)
    mismatch = abs(sortie_time - truck_leg)
    return 2.5 * wait_excess + 0.5 * mismatch + 0.01 * sortie_time


def _candidate_drone_inserts(node, truck, drone1, drone2, truck_times, drone_times):
    candidates = []
    route_order = [(1, drone1, drone2), (2, drone2, drone1)]
    route_order.sort(key=lambda x: len(x[1]))

    for route_id, target_route, other_route in route_order:
        positions = list(range(0, len(target_route) + 1))
        if len(positions) > 6:
            positions = [0, len(target_route) // 2, len(target_route)]
            while len(positions) < 6:
                positions.append(random.randint(0, len(target_route)))
            positions = list(dict.fromkeys(positions))

        if random.random() < _EXPLORE_PROB:
            random.shuffle(positions)

        for ins_idx in positions:
            pair = build_drone_pair(
                node,
                truck,
                target_route,
                ins_idx,
                max_neighbors=8,
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
            if not drone_route_is_feasible(other_route):
                continue
            if not _route_endpoint_unique(other_route):
                continue

            if route_id == 1:
                d1_len = len(new_target)
                d2_len = len(other_route)
            else:
                d1_len = len(other_route)
                d2_len = len(new_target)

            timing_pen = _drone_trip_penalty(node, pair, truck, truck_times, drone_times)
            balance_pen = 2.0 * abs(d1_len - d2_len)
            score = timing_pen + balance_pen
            candidates.append((score, route_id, new_target))

    return candidates


def _drone_phase_penalty():
    if _SEARCH_PROGRESS < 0.15:
        return 120.0
    if _SEARCH_PROGRESS < 0.35:
        return 60.0
    if _SEARCH_PROGRESS < 0.60:
        return 20.0
    return 0.0


def _node_insertion_options(node, truck, drone1, drone2, truck_times, drone_times):
    options = []

    truck_delta, truck_ins_idx = _best_truck_insert(node, truck, truck_times)
    if truck_ins_idx is not None:
        options.append((truck_delta, "truck", truck_ins_idx))

    drone_candidates = _candidate_drone_inserts(
        node, truck, drone1, drone2, truck_times, drone_times
    )
    if drone_candidates:
        drone_phase_pen = _drone_phase_penalty()
        for drone_score, route_id, new_target in drone_candidates:
            options.append((drone_score + drone_phase_pen, "drone", route_id, new_target))

    options.sort(key=lambda x: x[0])
    return options


def operator(current_solution):
    """
    op6: LNS destroy/repair
    - Destroy: remove several high-cost truck customers (excluding drone endpoints).
    - Repair: reinsert each removed node using best truck insertion or feasible drone insertion.
    """
    assert_context_is_set()
    truck_times, drone_times, _, depot = get_operator_context()

                                                                       
                                                
    candidate = _clone_solution(current_solution)
    truck, drone1, drone2 = candidate
    if len(truck) <= 4:
        return current_solution

    remove_count = _destroy_size(len(truck))
    destroy_indices = _pick_destroy_indices(truck, drone1, drone2, truck_times, depot, remove_count)
    if not destroy_indices:
        return current_solution

    removed_nodes = []
    for rem_idx in sorted(destroy_indices, reverse=True):
        removed_nodes.append(truck[rem_idx])
        truck = truck[:rem_idx] + truck[rem_idx + 1 :]
        drone1 = _shift_route_after_truck_removal(drone1, rem_idx)
        drone2 = _shift_route_after_truck_removal(drone2, rem_idx)
        if drone1 is None or drone2 is None:
            return current_solution

    pending_nodes = removed_nodes[:]
    while pending_nodes:
        best_choice = None
        best_node = None

        for node in pending_nodes:
            options = _node_insertion_options(
                node, truck, drone1, drone2, truck_times, drone_times
            )
            if not options:
                continue

            best_opt = options[0]
            best_cost = best_opt[0]
            second_cost = options[1][0] if len(options) > 1 else (best_cost + 250.0)
            regret = second_cost - best_cost

            # Regret-2: insert first the node that would hurt most if postponed.
            # Tie-break by lower best insertion cost.
            cand = (regret, -best_cost, best_opt, node)
            if best_choice is None or cand[0] > best_choice[0] or (
                cand[0] == best_choice[0] and cand[1] > best_choice[1]
            ):
                best_choice = cand
                best_node = node

        if best_choice is None:
            return current_solution

        _, _, chosen_opt, _ = best_choice
        _, mode, *data = chosen_opt
        if mode == "truck":
            truck_ins_idx = data[0]
            truck = truck[:truck_ins_idx] + [best_node] + truck[truck_ins_idx:]
        else:
            best_route_id, best_new_target = data
            if best_route_id == 1:
                drone1 = best_new_target
            else:
                drone2 = best_new_target

        pending_nodes.remove(best_node)

    new_solution = [truck, drone1, drone2]
    if new_solution == current_solution:
        return current_solution
    return new_solution
