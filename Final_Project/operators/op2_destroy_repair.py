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


def set_search_progress(progress):
    global _SEARCH_PROGRESS
    _SEARCH_PROGRESS = max(0.0, min(1.0, float(progress)))


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
    # tiers based on truck customer count, not total n.
    # this way destroy size adapts as customers move to drones.
    # tried 5 once, way too disruptive on big instances, 3 is the sweet spot.
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


# top_k cheapest truck insertion positions for node, best first
def _top_truck_inserts(node, truck, truck_times, top_k=3):
    scored = []
    for ins_idx in range(1, len(truck)):
        a = truck[ins_idx - 1]
        b = truck[ins_idx]
        delta = truck_times[a][node] + truck_times[node][b] - truck_times[a][b]
        scored.append((delta, ins_idx))
    scored.sort()
    return scored[:top_k]


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

    # keep top-3 truck spots so we're not stuck with a single greedy choice
    for delta, ins_idx in _top_truck_inserts(node, truck, truck_times, top_k=3):
        options.append((delta, "truck", ins_idx))

    drone_candidates = _candidate_drone_inserts(
        node, truck, drone1, drone2, truck_times, drone_times
    )
    if drone_candidates:
        drone_phase_pen = _drone_phase_penalty()
        for drone_score, route_id, new_target in drone_candidates:
            options.append((drone_score + drone_phase_pen, "drone", route_id, new_target))

    options.sort(key=lambda x: x[0])
    return options


# vehicle-level regret-2: |best_truck - best_drone|.
# bigger gap = more decisive node, place it first.
def _vehicle_regret(node, truck, drone1, drone2, truck_times, drone_times):
    truck_opts = _top_truck_inserts(node, truck, truck_times, top_k=3)
    truck_cost = truck_opts[0][0] if truck_opts else None

    drone_cands = _candidate_drone_inserts(
        node, truck, drone1, drone2, truck_times, drone_times
    )
    drone_phase_pen = _drone_phase_penalty()
    drone_cost = (drone_cands[0][0] + drone_phase_pen) if drone_cands else None

    if truck_cost is None and drone_cost is None:
        return None
    if truck_cost is None:
        return float("inf"), None, drone_cands, False
    if drone_cost is None:
        return float("inf"), truck_opts, None, True
    regret = abs(truck_cost - drone_cost)
    prefer_truck = truck_cost <= drone_cost
    return regret, truck_opts, drone_cands, prefer_truck


# destroy: drop a few high-cost truck customers (skipping drone endpoints).
# repair: reinsert each one into truck or drone via vehicle-regret ordering.
def operator(current_solution):
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
        # rank by regret = |best_truck_cost - best_drone_cost|.
        # place the node with the largest gap first into its cheaper vehicle
        # so we don't waste good slots on nodes that don't care much.
        best_node = None
        best_regret = -1.0
        best_truck_opts = None
        best_drone_cands = None
        best_prefer_truck = True

        for node in pending_nodes:
            r = _vehicle_regret(node, truck, drone1, drone2, truck_times, drone_times)
            if r is None:
                continue
            regret, truck_opts, drone_cands, prefer_truck = r
            if regret > best_regret:
                best_regret = regret
                best_node = node
                best_truck_opts = truck_opts
                best_drone_cands = drone_cands
                best_prefer_truck = prefer_truck

        if best_node is None:
            return current_solution

        # insert into the preferred vehicle, rank-biased pick to keep
        # a bit of randomness in which exact position
        if best_prefer_truck and best_truck_opts:
            chosen = _rank_biased_pick(best_truck_opts, top_k=min(3, len(best_truck_opts)))
            _, ins_idx = chosen
            truck = truck[:ins_idx] + [best_node] + truck[ins_idx:]
        elif best_drone_cands:
            chosen = _rank_biased_pick(best_drone_cands, top_k=min(3, len(best_drone_cands)))
            _, route_id, new_target = chosen
            if route_id == 1:
                drone1 = new_target
            else:
                drone2 = new_target
        else:
            # nothing in the preferred vehicle, fall back to the other one
            if best_truck_opts:
                chosen = _rank_biased_pick(best_truck_opts, top_k=min(3, len(best_truck_opts)))
                _, ins_idx = chosen
                truck = truck[:ins_idx] + [best_node] + truck[ins_idx:]
            elif best_drone_cands:
                chosen = _rank_biased_pick(best_drone_cands, top_k=min(3, len(best_drone_cands)))
                _, route_id, new_target = chosen
                if route_id == 1:
                    drone1 = new_target
                else:
                    drone2 = new_target
            else:
                return current_solution

        pending_nodes.remove(best_node)

    new_solution = [truck, drone1, drone2]
    if new_solution == current_solution:
        return current_solution
    return new_solution
