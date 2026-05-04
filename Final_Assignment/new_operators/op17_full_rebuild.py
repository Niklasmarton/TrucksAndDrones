"""
Op17: Aggressive ruin-and-recreate. Mirrors fast_tsp_operator approach:
  1. Remove K random truck customers + K random drone customers.
  2. Strip ALL remaining drone customers (full drone partition reset).
  3. Merge removed-from-drone into truck pool, TSP-rebuild the truck route.
  4. Reassign every "loose" customer (removed-from-truck + stripped drones)
     to the drones via either:
       - small-n branch: optimal recursive (launch, land, drone) assignment
         scored by total arrival time. Also tries all truck permutations
         when truck-set is small enough.
       - large-n branch: greedy earliest-arrival insertion.

Unlike op11 (which preserves drone structure), op17 wipes the drone partition
each call. This is what lets ALNS migrate between truck-set basins.
The optimal-assign branch is required to reach 585 on R_10: greedy insertion
cannot construct the depot-launching/depot-landing drone tuples that the
optimal solution uses.
"""
from pathlib import Path
import random
import sys
from itertools import permutations

ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
CORE_DIR = ASSIGNMENT_DIR / "core"
NEW_OPS_DIR = ASSIGNMENT_DIR / "new_operators"
if str(CORE_DIR) not in sys.path:
    sys.path.append(str(CORE_DIR))
if str(NEW_OPS_DIR) not in sys.path:
    sys.path.append(str(NEW_OPS_DIR))

from operator_context import assert_context_is_set, get_operator_context, set_operator_context
from drone_route_utils import build_drone_pair, drone_route_is_feasible

import op11_TSP_drone_rebuild as op11
import op16_optimal_drone_assign as op16

_SMALL_N_THRESHOLD = 12
_OPT_BRANCH_PROB = 0.25
_OPT_TRUCK_CANDIDATES = 4
_OPT_MAX_LEAVES = 30000


def _clone(solution):
    return [solution[0][:], solution[1][:], solution[2][:]]


def _route_endpoint_unique(route):
    used_l, used_r = set(), set()
    for _, l, r in route:
        if l in used_l or r in used_r:
            return False
        used_l.add(l); used_r.add(r)
    return True


def _greedy_assign_one(customer, truck, drone1, drone2,
                       truck_times, drone_times, flight_range, depot):
    """Try to insert `customer` into best drone slot; if none feasible, into
    best truck position. Returns (truck, drone1, drone2)."""
    prefix = [0.0]
    for i in range(len(truck) - 1):
        prefix.append(prefix[-1] + truck_times[truck[i]][truck[i + 1]])

    best_did = None
    best_pair = None
    best_arrival = float("inf")

    for did, route in ((1, drone1), (2, drone2)):
        for ins in range(len(route) + 1):
            pair = build_drone_pair(
                customer, truck, route, ins,
                max_neighbors=12, pair_explore_prob=0.0,
            )
            if pair is None:
                continue
            launch_idx, land_idx = pair
            launch_node = truck[launch_idx]
            sortie = drone_times[launch_node][customer] + drone_times[customer][truck[land_idx]]
            if sortie > flight_range:
                continue
            arrival = prefix[launch_idx] + drone_times[launch_node][customer]
            if arrival < best_arrival:
                test_route = route[:]
                test_route.insert(ins, (customer, launch_idx, land_idx))
                test_route.sort(key=lambda x: (x[1], x[2], x[0]))
                if not drone_route_is_feasible(test_route):
                    continue
                if not _route_endpoint_unique(test_route):
                    continue
                best_arrival = arrival
                best_did = did
                best_pair = (test_route, route)

    if best_did is not None:
        new_route, _ = best_pair
        if best_did == 1:
            return truck[:], new_route, drone2[:]
        else:
            return truck[:], drone1[:], new_route

    best_delta = float("inf")
    best_idx = 1
    for idx in range(1, len(truck)):
        a, b = truck[idx - 1], truck[idx]
        delta = truck_times[a][customer] + truck_times[customer][b] - truck_times[a][b]
        if delta < best_delta:
            best_delta = delta
            best_idx = idx
    new_truck = truck[:best_idx] + [customer] + truck[best_idx:]
    shifted_d1 = [(c, l + (l >= best_idx), r + (r >= best_idx)) for c, l, r in drone1]
    shifted_d2 = [(c, l + (l >= best_idx), r + (r >= best_idx)) for c, l, r in drone2]
    return new_truck, shifted_d1, shifted_d2


def _enumerate_drone_options(customer, truck, drone_times, flight_range):
    """All feasible (drone_id, launch_idx, land_idx) tuples for `customer`
    given the truck route, ignoring flight-range hover-wait (full-objective
    eval will catch any infeasibility)."""
    n = len(truck)
    opts = []
    for launch in range(n - 1):
        for land in range(launch + 1, n):
            ln = truck[launch]
            ld = truck[land]
            sortie = drone_times[ln][customer] + drone_times[customer][ld]
            if sortie > flight_range:
                continue
            for did in (1, 2):
                opts.append((did, launch, land))
    return opts


def _optimal_assign(truck_route, loose_custs,
                    truck_times, drone_times, flight_range, depot,
                    leaf_budget=_OPT_MAX_LEAVES):
    """Recursive optimal assignment of `loose_custs` to drone slots given
    `truck_route`. Returns (best_score, best_d1, best_d2) or (inf, [], []).

    `leaf_budget` caps the number of leaf evaluations to keep wall-time bounded.
    """
    if not loose_custs:
        sol = [truck_route[:], [], []]
        s = op16._total_arrival_time(sol, truck_times, drone_times, depot, flight_range)
        return s, [], []

    options = {}
    for c in loose_custs:
        opts = _enumerate_drone_options(c, truck_route, drone_times, flight_range)
        if not opts:
            return float("inf"), [], []
        options[c] = opts

    custs_sorted = sorted(loose_custs, key=lambda c: len(options[c]))

    best = [float("inf"), [], []]
    leaves = [0]

    def recurse(idx, d1, d2):
        if leaves[0] >= leaf_budget:
            return
        if idx == len(custs_sorted):
            leaves[0] += 1
            s = op16._total_arrival_time(
                [truck_route, d1, d2], truck_times, drone_times, depot, flight_range
            )
            if s < best[0]:
                best[0] = s
                best[1] = d1[:]
                best[2] = d2[:]
            return
        c = custs_sorted[idx]
        for did, launch, land in options[c]:
            target = d1 if did == 1 else d2
            conflict = False
            for _, l, r in target:
                if l == launch or r == land:
                    conflict = True
                    break
            if conflict:
                continue
            if did == 1:
                d1.append((c, launch, land))
                recurse(idx + 1, d1, d2)
                d1.pop()
            else:
                d2.append((c, launch, land))
                recurse(idx + 1, d1, d2)
                d2.pop()
            if leaves[0] >= leaf_budget:
                return

    recurse(0, [], [])
    if best[0] == float("inf"):
        return float("inf"), [], []
    d1_sorted = sorted(best[1], key=lambda x: (x[1], x[2], x[0]))
    d2_sorted = sorted(best[2], key=lambda x: (x[1], x[2], x[0]))
    return best[0], d1_sorted, d2_sorted


def _small_n_rebuild(truck_pool, loose_custs,
                     truck_times, drone_times, flight_range, depot):
    """Try a small set of truck route candidates (TSP-optimal + a few random
    shuffles), do optimal drone assignment for each, return best.

    Bounded to _OPT_TRUCK_CANDIDATES routes for wall-time control."""
    if not truck_pool:
        return None

    skeleton = [depot] + truck_pool + [depot]
    tsp_route = op11._optimize_truck_route_with_tsp(skeleton, truck_times, depot)

    seen = {tuple(tsp_route)}
    truck_candidates = [tsp_route]
    while len(truck_candidates) < _OPT_TRUCK_CANDIDATES:
        shuffled = truck_pool[:]
        random.shuffle(shuffled)
        cand = [depot] + shuffled + [depot]
        key = tuple(cand)
        if key in seen:
            continue
        seen.add(key)
        truck_candidates.append(cand)

    best_score = float("inf")
    best_sol = None
    for truck_route in truck_candidates:
        s, d1, d2 = _optimal_assign(
            truck_route, loose_custs, truck_times, drone_times, flight_range, depot
        )
        if s < best_score:
            best_score = s
            best_sol = (truck_route[:], d1, d2)
    if best_sol is None:
        return None
    return best_sol


def operator(current_solution):
    assert_context_is_set()
    truck_times, drone_times, flight_range, depot = get_operator_context()

    truck, drone1, drone2 = _clone(current_solution)
    n_truck_custs = len(truck) - 2
    n_drone_custs = len(drone1) + len(drone2)
    n_total = n_truck_custs + n_drone_custs

    if n_total == 0:
        return current_solution

    if n_total <= 10:
        max_remove = 3
    elif n_total <= 20:
        max_remove = 4
    elif n_total <= 50:
        max_remove = 6
    else:
        max_remove = 6

    k_truck = random.randint(1, max(1, min(max_remove, n_truck_custs))) if n_truck_custs else 0
    k_drone = random.randint(1, max(1, min(max_remove, n_drone_custs))) if n_drone_custs else 0

    truck_remove_idx = []
    if k_truck > 0:
        truck_remove_idx = random.sample(range(1, len(truck) - 1), k_truck)

    truck_removed_custs = [truck[i] for i in truck_remove_idx]
    truck_kept_custs = [n for i, n in enumerate(truck[1:-1], start=1) if i not in set(truck_remove_idx)]

    drone_pool = [(1, i) for i in range(len(drone1))] + [(2, i) for i in range(len(drone2))]
    drone_chosen = random.sample(drone_pool, k_drone) if k_drone > 0 else []
    chosen_custs = set()
    for did, idx in drone_chosen:
        route = drone1 if did == 1 else drone2
        chosen_custs.add(route[idx][0])

    all_drone_custs = [c for c, _, _ in drone1 + drone2]
    stripped_drone_custs = [c for c in all_drone_custs if c not in chosen_custs]

    truck_pool = truck_kept_custs + list(chosen_custs)
    if not truck_pool:
        if not stripped_drone_custs and not truck_removed_custs:
            return current_solution
        seed = (stripped_drone_custs.pop() if stripped_drone_custs
                else truck_removed_custs.pop())
        truck_pool = [seed]

    loose_custs = truck_removed_custs + stripped_drone_custs

    use_optimal = (
        n_total <= _SMALL_N_THRESHOLD
        and k_truck >= 2
        and k_drone >= 2
        and random.random() < _OPT_BRANCH_PROB
    )

    if use_optimal:
        result = _small_n_rebuild(
            truck_pool, loose_custs,
            truck_times, drone_times, flight_range, depot,
        )
        if result is None:
            return current_solution
        new_truck, new_d1, new_d2 = result
        if not drone_route_is_feasible(new_d1) or not drone_route_is_feasible(new_d2):
            return current_solution
        if not _route_endpoint_unique(new_d1) or not _route_endpoint_unique(new_d2):
            return current_solution
        return [new_truck, new_d1, new_d2]

    skeleton_truck = [depot] + truck_pool + [depot]
    new_truck = op11._optimize_truck_route_with_tsp(skeleton_truck, truck_times, depot)

    to_assign = loose_custs[:]
    to_assign.sort(key=lambda c: drone_times[depot][c])

    cur_truck = new_truck
    cur_d1 = []
    cur_d2 = []
    for cust in to_assign:
        cur_truck, cur_d1, cur_d2 = _greedy_assign_one(
            cust, cur_truck, cur_d1, cur_d2,
            truck_times, drone_times, flight_range, depot,
        )

    if not drone_route_is_feasible(cur_d1) or not drone_route_is_feasible(cur_d2):
        return current_solution
    if not _route_endpoint_unique(cur_d1) or not _route_endpoint_unique(cur_d2):
        return current_solution

    return [cur_truck, cur_d1, cur_d2]


def set_search_progress(progress):
    return None
