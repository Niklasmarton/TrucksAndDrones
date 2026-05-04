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
    remap_drone_route_by_endpoint_nodes,
    repair_drone_route,
)

_MAX_REBUILD_TRIPS = 6   # rebuild at most this many worst-synced trips
_TOP_K_PAIRS = 5         # rank-biased pool size when picking a new pair
_EXPLORE_PROB = 0.10     # chance to shuffle scored trips before picking
_PARTITION_PERTURB = 3   # customers swapped between truck and drones per escape


def set_search_progress(progress):
    return None


def _clone_solution(solution):
    return [solution[0][:], solution[1][:], solution[2][:]]


# cut the internal route at 3 points into A B C D, reconnect as A C B D.
# smallest non-sequential reconnection that 2-opt/3-opt can't undo, so it really
# crosses a basin. needs at least 8 internal nodes.
def _double_bridge(truck):
    internal = truck[1:-1]
    if len(internal) < 8:
        return None

    # 3 distinct cut points so every segment has at least 1 node
    positions = list(range(1, len(internal)))
    if len(positions) < 3:
        return None
    cuts = sorted(random.sample(positions, 3))
    a, b, c = cuts

    seg_a = internal[:a]
    seg_b = internal[a:b]
    seg_c = internal[b:c]
    seg_d = internal[c:]

    # A -> C -> B -> D
    new_internal = seg_a + seg_c + seg_b + seg_d
    return [truck[0]] + new_internal + [truck[-1]]


# lower = better. penalises drone hover (truck slow) and big flight/leg mismatch.
def _trip_sync_score(node, launch_idx, land_idx, truck, T, D):
    if not (0 <= launch_idx < land_idx < len(truck)):
        return float("inf")
    l_node = truck[launch_idx]
    r_node = truck[land_idx]
    drone_flight = D[l_node][node] + D[node][r_node]
    truck_leg = sum(T[truck[i]][truck[i + 1]] for i in range(launch_idx, land_idx))
    hover_penalty = max(0.0, truck_leg - drone_flight)
    mismatch = abs(drone_flight - truck_leg)
    return 2.0 * hover_penalty + 0.5 * mismatch + 0.01 * drone_flight


# rank-biased pick over top-k pairs. rank 0 -> weight k, rank 1 -> k-1, etc.
# scored_pairs must be sorted best-first.
def _biased_top_k_pick(scored_pairs, top_k):
    if not scored_pairs:
        return None
    k = min(len(scored_pairs), top_k)
    pool = scored_pairs[:k]
    weights = [k - i for i in range(k)]
    chosen = random.choices(pool, weights=weights, k=1)[0]
    return chosen[1], chosen[2]


# after the truck topology changes, re-pick (launch, land) for the worst-synced trips.
# score every trip, pick the worst MAX_REBUILD_TRIPS, and try every feasible pair.
def _rebuild_worst_trips(drone_route, other_route, truck, T, D, flight_limit):
    if not drone_route:
        return drone_route

    scored_trips = [(
        _trip_sync_score(node, l, r, truck, T, D), pos
    ) for pos, (node, l, r) in enumerate(drone_route)]

    # worst first, optionally shuffle the top to add variety
    scored_trips.sort(reverse=True)
    if random.random() < _EXPLORE_PROB:
        k_top = min(len(scored_trips), _MAX_REBUILD_TRIPS * 2)
        pool = scored_trips[:k_top]
        random.shuffle(pool)
        candidates = pool
    else:
        candidates = scored_trips

    route = drone_route[:]
    improved = False

    for _, pos in candidates[:_MAX_REBUILD_TRIPS]:
        node, old_l, old_r = route[pos]

        # endpoints already used by sibling trips in this route
        used_l = {l for i, (_, l, _) in enumerate(route) if i != pos}
        used_r = {r for i, (_, _, r) in enumerate(route) if i != pos}

        # every feasible (launch, land) pair for this node
        nt = len(truck)
        pair_candidates = []
        for l_idx in range(nt - 1):
            if l_idx in used_l:
                continue
            for r_idx in range(l_idx + 1, nt):
                if r_idx in used_r:
                    continue
                ft = D[truck[l_idx]][node] + D[node][truck[r_idx]]
                if ft > flight_limit:
                    continue
                score = _trip_sync_score(node, l_idx, r_idx, truck, T, D)
                pair_candidates.append((score, l_idx, r_idx))

        if not pair_candidates:
            continue

        pair_candidates.sort()
        picked = _biased_top_k_pick(pair_candidates, _TOP_K_PAIRS)
        if picked is None:
            continue
        new_l, new_r = picked
        if new_l == old_l and new_r == old_r:
            continue

        # tentatively apply and check timeline feasibility
        candidate_route = route[:]
        candidate_route[pos] = (node, new_l, new_r)
        candidate_route.sort(key=lambda x: (x[1], x[2], x[0]))
        if drone_route_is_feasible(candidate_route):
            route = candidate_route
            improved = True

    return route if improved else drone_route


def _best_truck_ins(node, truck, T):
    best_idx, best_delta = None, None
    for i in range(1, len(truck)):
        a, b = truck[i - 1], truck[i]
        d = T[a][node] + T[node][b] - T[a][b]
        if best_delta is None or d < best_delta:
            best_delta, best_idx = d, i
    return best_idx


# the double-bridge only reshapes the truck route, the drone/truck partition is
# untouched. this perturbation moves a few drone customers onto the truck and
# pushes a few truck customers down to drones to also shuffle that partition.
def _perturb_drone_partition(truck, drone1, drone2, T, D, flight_limit):
    all_drone = ([(1, n, l, r) for n, l, r in drone1] +
                 [(2, n, l, r) for n, l, r in drone2])
    if not all_drone:
        return truck, drone1, drone2

    n_move = min(_PARTITION_PERTURB, len(all_drone))
    to_move = random.sample(all_drone, n_move)

    t, d1, d2 = truck[:], drone1[:], drone2[:]

    # phase 1: drone -> truck. greedy cheapest insert, shift drone indices.
    for rid, node, _, _ in to_move:
        if rid == 1:
            d1 = [(cn, cl, cr) for cn, cl, cr in d1 if cn != node]
        else:
            d2 = [(cn, cl, cr) for cn, cl, cr in d2 if cn != node]

        ins = _best_truck_ins(node, t, T)
        if ins is None:
            return truck, drone1, drone2

        t = t[:ins] + [node] + t[ins:]
        # shift drone indices at or after the insertion point
        d1 = [(cn, cl + (1 if cl >= ins else 0), cr + (1 if cr >= ins else 0))
              for cn, cl, cr in d1]
        d1.sort(key=lambda x: (x[1], x[2], x[0]))
        d2 = [(cn, cl + (1 if cl >= ins else 0), cr + (1 if cr >= ins else 0))
              for cn, cl, cr in d2]
        d2.sort(key=lambda x: (x[1], x[2], x[0]))

    # phase 2: truck -> drones, rank-biased pick by drone-friendliness
    used_l = {cl for _, cl, _ in d1} | {cl for _, cl, _ in d2}
    used_r = {cr for _, _, cr in d1} | {cr for _, _, cr in d2}
    eligible = [i for i in range(1, len(t) - 1)
                if i not in used_l and i not in used_r]

    # score = best feasible flight / flight_limit (lower = more drone-friendly)
    scored = []
    for idx in eligible:
        node = t[idx]
        best_ft = float("inf")
        for li in range(len(t) - 1):
            if li in used_l:
                continue
            for ri in range(li + 1, len(t)):
                if ri in used_r:
                    continue
                ft = D[t[li]][node] + D[node][t[ri]]
                if ft <= flight_limit and ft < best_ft:
                    best_ft = ft
        if best_ft <= flight_limit:
            scored.append((best_ft / flight_limit, idx, node))

    scored.sort()
    pool = scored[:min(len(scored), n_move * 3)]

    moved = 0
    while pool and moved < n_move:
        pick_k = min(len(pool), 4)
        weights = [pick_k - i for i in range(pick_k)]
        _, idx, node = random.choices(pool[:pick_k], weights=weights, k=1)[0]
        pool = [(s, i, n) for s, i, n in pool if n != node]

        used_l = {cl for _, cl, _ in d1} | {cl for _, cl, _ in d2}
        used_r = {cr for _, _, cr in d1} | {cr for _, _, cr in d2}

        success = False
        for target in (d1, d2):
            pair = build_drone_pair(node, t, target, len(target),
                                   max_neighbors=8, pair_explore_prob=0.2)
            if pair is None:
                continue
            pl, pr = pair
            # pair endpoints can't be the node's own current truck position
            if pl == idx or pr == idx:
                continue
            if pl in used_l or pr in used_r:
                continue

            # pull node off the truck, shift everything that came after
            t_new = t[:idx] + t[idx + 1:]
            d1_new = [(cn, cl - (1 if cl > idx else 0), cr - (1 if cr > idx else 0))
                      for cn, cl, cr in d1]
            d1_new.sort(key=lambda x: (x[1], x[2], x[0]))
            d2_new = [(cn, cl - (1 if cl > idx else 0), cr - (1 if cr > idx else 0))
                      for cn, cl, cr in d2]
            d2_new.sort(key=lambda x: (x[1], x[2], x[0]))

            nl = pl - (1 if pl > idx else 0)
            nr = pr - (1 if pr > idx else 0)
            if nl >= nr or nl <= 0:
                continue

            if target is d1:
                d1_new.append((node, nl, nr))
                d1_new.sort(key=lambda x: (x[1], x[2], x[0]))
            else:
                d2_new.append((node, nl, nr))
                d2_new.sort(key=lambda x: (x[1], x[2], x[0]))

            if not drone_route_is_feasible(d1_new) or not drone_route_is_feasible(d2_new):
                continue

            t, d1, d2 = t_new, d1_new, d2_new
            pool = [(s, i - (1 if i > idx else 0), n)
                    for s, i, n in pool if i != idx]
            moved += 1
            success = True
            break

    # last-ditch guard, return the original if anything broke
    if not drone_route_is_feasible(d1) or not drone_route_is_feasible(d2):
        return truck, drone1, drone2

    return t, d1, d2


# 1. double-bridge the truck route (basin escape)
# 2. remap drone routes to follow their endpoint nodes
# 3. repair any ordering/overlap from the remap
# 4. rebuild the worst-synced trips for the new topology
# 5. perturb the drone/truck partition that the double-bridge doesn't touch
def operator(current_solution):
    assert_context_is_set()
    T, D, flight_limit, _ = get_operator_context()

    truck, drone1, drone2 = current_solution

    new_truck = _double_bridge(truck)
    if new_truck is None:
        return current_solution

    new_drone1 = remap_drone_route_by_endpoint_nodes(truck, new_truck, drone1)
    new_drone2 = remap_drone_route_by_endpoint_nodes(truck, new_truck, drone2)
    if new_drone1 is None or new_drone2 is None:
        return current_solution

    if not drone_route_is_feasible(new_drone1):
        repaired = repair_drone_route(new_truck, new_drone1)
        new_drone1 = repaired if repaired is not None else []

    if not drone_route_is_feasible(new_drone2):
        repaired = repair_drone_route(new_truck, new_drone2)
        new_drone2 = repaired if repaired is not None else []

    new_drone1 = _rebuild_worst_trips(new_drone1, new_drone2, new_truck, T, D, flight_limit)
    new_drone2 = _rebuild_worst_trips(new_drone2, new_drone1, new_truck, T, D, flight_limit)

    new_truck, new_drone1, new_drone2 = _perturb_drone_partition(
        new_truck, new_drone1, new_drone2, T, D, flight_limit
    )

    return [new_truck, new_drone1, new_drone2]


# fallback: large related-destroy + regret-2 repair. only used when the
# double-bridge primary can't produce a feasible candidate. not part of the
# normal operator pool, only the escape mechanism calls fallback().
def _lrd_rank_biased_pick(candidates, top_k):
    if not candidates:
        return None
    k = min(len(candidates), top_k)
    if k <= 1:
        return candidates[0]
    weights = [k - i for i in range(k)]
    return random.choices(candidates[:k], weights=weights, k=1)[0]


def _lrd_destroy_size(n_customers):
    low = max(4, int(round(0.08 * n_customers)))
    high = max(low, int(round(0.18 * n_customers)))
    high = min(high, max(4, n_customers))
    return random.randint(low, high)


def _lrd_pick_related_indices(truck, truck_times, depot, count):
    customer_positions = [i for i in range(1, len(truck) - 1) if truck[i] != depot]
    if not customer_positions:
        return []

    seed_idx = random.choice(customer_positions)
    seed_node = truck[seed_idx]

    scored = []
    for idx in customer_positions:
        node = truck[idx]
        distance = truck_times[seed_node][node]
        order_gap = abs(idx - seed_idx)
        relatedness = distance + 0.20 * order_gap
        scored.append((relatedness, idx))
    scored.sort(key=lambda x: x[0])

    pool_size = min(len(scored), max(count * 3, 12))
    pool = scored[:pool_size]
    work = pool[:]
    chosen = []
    top_k = min(len(work), max(count * 2, 6))
    while work and len(chosen) < count:
        picked = _lrd_rank_biased_pick(work, top_k=top_k)
        if picked is None:
            break
        _, idx = picked
        chosen.append(idx)
        work.remove(picked)
        top_k = min(len(work), max(count * 2, 6))

    return sorted(set(chosen), reverse=True)


def _lrd_shift_or_drop_after_truck_removal(route, removed_idx, removed_node):
    kept = []
    orphan_nodes = []
    for node, launch_idx, land_idx in route:
        if node == removed_node or launch_idx == removed_idx or land_idx == removed_idx:
            orphan_nodes.append(node)
            continue
        if launch_idx > removed_idx:
            launch_idx -= 1
        if land_idx > removed_idx:
            land_idx -= 1
        kept.append((node, launch_idx, land_idx))
    kept.sort(key=lambda x: (x[1], x[2], x[0]))
    return kept, orphan_nodes


def _lrd_shift_route_after_truck_insert(route, insert_idx):
    shifted = []
    for node, launch_idx, land_idx in route:
        if launch_idx >= insert_idx:
            launch_idx += 1
        if land_idx >= insert_idx:
            land_idx += 1
        shifted.append((node, launch_idx, land_idx))
    shifted.sort(key=lambda x: (x[1], x[2], x[0]))
    return shifted


def _lrd_best_two_truck_inserts(node, truck, truck_times):
    best = None
    second = None
    for ins_idx in range(1, len(truck)):
        a = truck[ins_idx - 1]
        b = truck[ins_idx]
        delta = truck_times[a][node] + truck_times[node][b] - truck_times[a][b]
        cand = (delta, ins_idx)
        if best is None or cand < best:
            second = best
            best = cand
        elif second is None or cand < second:
            second = cand
    return best, second


def _lrd_regret_repair(pending_nodes, truck, drone1, drone2, truck_times):
    pending = pending_nodes[:]

    while pending:
        best_choice = None
        for node in pending:
            best_ins, second_ins = _lrd_best_two_truck_inserts(node, truck, truck_times)
            if best_ins is None:
                continue
            best_delta, best_idx = best_ins
            second_delta = second_ins[0] if second_ins is not None else (best_delta + 1e6)
            regret = second_delta - best_delta
            cand = (-regret, best_delta, best_idx, node)
            if best_choice is None or cand < best_choice:
                best_choice = cand

        if best_choice is None:
            return None

        _, _, ins_idx, node = best_choice
        truck = truck[:ins_idx] + [node] + truck[ins_idx:]
        drone1 = _lrd_shift_route_after_truck_insert(drone1, ins_idx)
        drone2 = _lrd_shift_route_after_truck_insert(drone2, ins_idx)
        pending.remove(node)

    return [truck, drone1, drone2]


def fallback(current_solution):
    assert_context_is_set()
    truck_times, _, _, depot = get_operator_context()

    truck, drone1, drone2 = _clone_solution(current_solution)
    n_customers = max(1, len(truck) - 2)
    if n_customers < 8:
        return current_solution

    remove_count = min(n_customers, _lrd_destroy_size(n_customers))
    destroy_indices = _lrd_pick_related_indices(truck, truck_times, depot, remove_count)
    if not destroy_indices:
        return current_solution

    pending = []
    for rem_idx in destroy_indices:
        removed_node = truck[rem_idx]
        pending.append(removed_node)
        truck = truck[:rem_idx] + truck[rem_idx + 1:]
        drone1, orphan1 = _lrd_shift_or_drop_after_truck_removal(drone1, rem_idx, removed_node)
        drone2, orphan2 = _lrd_shift_or_drop_after_truck_removal(drone2, rem_idx, removed_node)
        pending.extend(orphan1)
        pending.extend(orphan2)

    unique_pending = []
    seen = set()
    for node in pending:
        if node in seen:
            continue
        seen.add(node)
        unique_pending.append(node)

    candidate = _lrd_regret_repair(unique_pending, truck, drone1, drone2, truck_times)
    if candidate is None or candidate == current_solution:
        return current_solution
    return candidate
