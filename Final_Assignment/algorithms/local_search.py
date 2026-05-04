"""Exhaustive local-search routine for ALNS intensification.

Three deterministic best-improvement sweeps applied in cycle until a full
cycle finds no improvement:

    1. truck_2opt_sweep         - reverse every truck segment, refit drones
    2. truck_drone_reassign_sweep - exhaustive truck<->drone single-move
    3. drone_window_sweep       - exhaustive launch/land retiming per trip

Each sweep evaluates every candidate, applies the single best-improving move
(if any), then re-runs from the new state. Returns when a sweep finds nothing.

Designed to be invoked at every new global best and once at end-of-run.
Caps: per-call iteration limit + optional wall-clock budget.
"""

from __future__ import annotations

import time
from pathlib import Path
import sys

CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.append(str(CORE_DIR))

from drone_route_utils import (
    build_drone_pair,
    drone_route_is_feasible,
    pair_fits_in_drone,
    remap_drone_route_by_endpoint_nodes,
    repair_drone_route,
)


def _clone(sol):
    return [sol[0][:], sol[1][:], sol[2][:]]


# ---------------------------------------------------------------------------
# Sweep 1: truck 2-opt with drone refit
# ---------------------------------------------------------------------------

def _two_opt_apply(truck, i, j):
    """Return new truck with truck[i..j] reversed (inclusive)."""
    return truck[:i] + truck[i : j + 1][::-1] + truck[j + 1 :]


def _two_opt_estimate_delta(truck, i, j, T):
    """Quick estimate of distance change from reversing truck[i..j]."""
    if i <= 0 or j >= len(truck) - 1:
        return 0.0
    a = truck[i - 1]
    b = truck[i]
    c = truck[j]
    d = truck[j + 1]
    return T[a][c] + T[b][d] - T[a][b] - T[c][d]


def truck_2opt_sweep(sol, evaluate, ctx, current_cost, deadline=None):
    """Best-improvement 2-opt on the truck route. Refits drones via endpoint
    remapping; falls back to repair_drone_route on minor breaks. Returns
    (new_sol, new_cost, improved_bool)."""
    T = ctx["T"]
    truck = sol[0]
    n = len(truck)
    if n < 5:
        return sol, current_cost, False

    # Build candidate list (i, j) with 1 <= i < j <= n-2 (skip depot endpoints).
    candidates = []
    for i in range(1, n - 1):
        for j in range(i + 1, n - 1):
            candidates.append((i, j))

    # Pruning: on routes > 20 nodes, evaluate only top-k by estimated delta.
    if n > 20:
        candidates.sort(key=lambda ij: _two_opt_estimate_delta(truck, ij[0], ij[1], T))
        k = max(20, n * n // 2)
        candidates = candidates[:k]

    best_delta = 0.0
    best_move = None
    best_sol = None

    for (i, j) in candidates:
        if deadline is not None and time.perf_counter() > deadline:
            break
        # Quick estimate: skip if truck-only delta is non-improving by >5% of current_cost
        # (drone refit can only worsen, never improve, the truck delta — but it can
        # affect drone arrival times. So we use a conservative skip filter only for
        # very-large positive estimated deltas.)
        est = _two_opt_estimate_delta(truck, i, j, T)
        if est > 0 and est > 0.02 * current_cost:
            continue

        new_truck = _two_opt_apply(truck, i, j)
        new_d1 = remap_drone_route_by_endpoint_nodes(truck, new_truck, sol[1])
        new_d2 = remap_drone_route_by_endpoint_nodes(truck, new_truck, sol[2])
        if new_d1 is None or new_d2 is None:
            continue

        if not drone_route_is_feasible(new_d1):
            new_d1 = repair_drone_route(new_truck, new_d1, max_repairs=10, max_neighbors=8)
            if new_d1 is None:
                continue
        if not drone_route_is_feasible(new_d2):
            new_d2 = repair_drone_route(new_truck, new_d2, max_repairs=10, max_neighbors=8)
            if new_d2 is None:
                continue

        cand_sol = [new_truck, new_d1, new_d2]
        feasible, cand_cost = evaluate(cand_sol)
        if not feasible:
            continue
        delta = cand_cost - current_cost
        if delta < best_delta - 1e-9:
            best_delta = delta
            best_move = (i, j)
            best_sol = cand_sol

    if best_move is None:
        return sol, current_cost, False
    return best_sol, current_cost + best_delta, True


# ---------------------------------------------------------------------------
# Sweep 2: exhaustive truck <-> drone reassignment
# ---------------------------------------------------------------------------

def _try_truck_to_drone(sol, c_idx_in_truck, evaluate, ctx, current_cost):
    """Move truck[c_idx_in_truck] into a drone route at its best position.
    Returns (best_sol, best_cost) or (None, current_cost) if no improvement."""
    truck = sol[0]
    if c_idx_in_truck <= 0 or c_idx_in_truck >= len(truck) - 1:
        return None, current_cost
    node = truck[c_idx_in_truck]

    # Tentative truck without this node.
    new_truck = truck[:c_idx_in_truck] + truck[c_idx_in_truck + 1 :]

    best_cost = current_cost
    best_sol = None

    for drone_idx in (1, 2):
        drone_route = sol[drone_idx]
        # Remap drone route indices to the new truck (which is shorter by 1).
        # Indices > c_idx_in_truck shift down by 1.
        remapped = []
        ok = True
        for cust, li, di in drone_route:
            new_li = li if li < c_idx_in_truck else (li - 1 if li > c_idx_in_truck else None)
            new_di = di if di < c_idx_in_truck else (di - 1 if di > c_idx_in_truck else None)
            if new_li is None or new_di is None:
                ok = False
                break
            remapped.append((cust, new_li, new_di))
        if not ok:
            continue
        if not drone_route_is_feasible(remapped):
            continue

        # Try every insertion position in the remapped drone route.
        for ins_idx in range(len(remapped) + 1):
            pair = build_drone_pair(
                node, new_truck, remapped, ins_idx,
                preferred_pair=None, max_neighbors=8, pair_explore_prob=0.0,
            )
            if pair is None:
                continue
            launch_idx, land_idx = pair
            if not pair_fits_in_drone(remapped, ins_idx, launch_idx, land_idx):
                continue
            new_drone = remapped[:]
            new_drone.insert(ins_idx, (node, launch_idx, land_idx))
            new_drone.sort(key=lambda x: (x[1], x[2], x[0]))

            other_drone_idx = 2 if drone_idx == 1 else 1
            other_remapped = []
            ok2 = True
            for cust, li, di in sol[other_drone_idx]:
                nl = li if li < c_idx_in_truck else (li - 1 if li > c_idx_in_truck else None)
                nd = di if di < c_idx_in_truck else (di - 1 if di > c_idx_in_truck else None)
                if nl is None or nd is None:
                    ok2 = False
                    break
                other_remapped.append((cust, nl, nd))
            if not ok2:
                continue
            if not drone_route_is_feasible(other_remapped):
                continue

            cand = [None, None, None]
            cand[0] = new_truck
            cand[drone_idx] = new_drone
            cand[other_drone_idx] = other_remapped

            feasible, cand_cost = evaluate(cand)
            if not feasible:
                continue
            if cand_cost < best_cost - 1e-9:
                best_cost = cand_cost
                best_sol = cand

    return best_sol, best_cost


def _try_drone_to_truck(sol, drone_idx, trip_idx, evaluate, ctx, current_cost):
    """Move drone[drone_idx][trip_idx] customer onto truck at cheapest position.
    Returns (best_sol, best_cost) or (None, current_cost)."""
    T = ctx["T"]
    truck = sol[0]
    drone_route = sol[drone_idx]
    if trip_idx < 0 or trip_idx >= len(drone_route):
        return None, current_cost
    node = drone_route[trip_idx][0]

    # Remove trip from drone route.
    new_drone = drone_route[:trip_idx] + drone_route[trip_idx + 1 :]

    best_cost = current_cost
    best_sol = None

    # Try every insertion position in truck (after position 0, before last).
    for ins_idx in range(1, len(truck)):
        a = truck[ins_idx - 1]
        b = truck[ins_idx]
        # Skip clearly-bad inserts (more than 5% of objective worse on truck delta).
        truck_delta = T[a][node] + T[node][b] - T[a][b]
        if truck_delta > 0.05 * current_cost:
            continue
        new_truck = truck[:ins_idx] + [node] + truck[ins_idx:]
        # Shift drone-route indices >= ins_idx by +1.
        shifted_active = []
        for cust, li, di in new_drone:
            shifted_active.append((cust, li + 1 if li >= ins_idx else li,
                                   di + 1 if di >= ins_idx else di))
        other_drone_idx = 2 if drone_idx == 1 else 1
        shifted_other = []
        for cust, li, di in sol[other_drone_idx]:
            shifted_other.append((cust, li + 1 if li >= ins_idx else li,
                                  di + 1 if di >= ins_idx else di))

        if not drone_route_is_feasible(shifted_active):
            continue
        if not drone_route_is_feasible(shifted_other):
            continue

        cand = [None, None, None]
        cand[0] = new_truck
        cand[drone_idx] = shifted_active
        cand[other_drone_idx] = shifted_other

        feasible, cand_cost = evaluate(cand)
        if not feasible:
            continue
        if cand_cost < best_cost - 1e-9:
            best_cost = cand_cost
            best_sol = cand

    return best_sol, best_cost


def truck_drone_reassign_sweep(sol, evaluate, ctx, current_cost, deadline=None):
    """Exhaustive single-customer reassignment: every truck->drone and
    drone->truck move. Apply the single best-improving."""
    truck = sol[0]
    best_sol = None
    best_cost = current_cost

    # Truck -> drone: every interior truck customer.
    for c_idx in range(1, len(truck) - 1):
        if deadline is not None and time.perf_counter() > deadline:
            break
        cand_sol, cand_cost = _try_truck_to_drone(sol, c_idx, evaluate, ctx, best_cost)
        if cand_sol is not None and cand_cost < best_cost - 1e-9:
            best_cost = cand_cost
            best_sol = cand_sol

    # Drone -> truck: every drone trip.
    for d_idx in (1, 2):
        for t_idx in range(len(sol[d_idx])):
            if deadline is not None and time.perf_counter() > deadline:
                break
            cand_sol, cand_cost = _try_drone_to_truck(sol, d_idx, t_idx, evaluate, ctx, best_cost)
            if cand_sol is not None and cand_cost < best_cost - 1e-9:
                best_cost = cand_cost
                best_sol = cand_sol

    if best_sol is None:
        return sol, current_cost, False
    return best_sol, best_cost, True


# ---------------------------------------------------------------------------
# Sweep 3: exhaustive drone window retiming
# ---------------------------------------------------------------------------

def drone_window_sweep(sol, evaluate, ctx, current_cost, deadline=None):
    """For each drone trip, exhaustively try every (launch_idx, land_idx) pair
    valid given neighboring trips and flight constraints. Apply best-improving."""
    T = ctx["T"]
    D = ctx["D"]
    flight_limit = ctx["flight_limit"]
    truck = sol[0]
    n_truck = len(truck)
    if n_truck < 4:
        return sol, current_cost, False

    best_sol = None
    best_cost = current_cost

    for d_idx in (1, 2):
        drone_route = sol[d_idx]
        for trip_idx in range(len(drone_route)):
            if deadline is not None and time.perf_counter() > deadline:
                break
            cust, cur_li, cur_di = drone_route[trip_idx]
            # Valid range: launch >= prev_land (or 0), land <= next_launch (or n-1).
            prev_land = 0 if trip_idx == 0 else drone_route[trip_idx - 1][2]
            next_launch = (n_truck - 1) if trip_idx == len(drone_route) - 1 else drone_route[trip_idx + 1][1]

            for li in range(prev_land, next_launch):
                if li < 1 or li >= n_truck - 1:
                    # launch can be at depot start? No: launch must be a truck stop, not endpoint.
                    # depot-as-launch is allowed in some schemes but here we keep launch interior.
                    if li == 0:
                        pass  # allow li=0 (depot launch)
                    else:
                        continue
                for di in range(li + 1, next_launch + 1):
                    if di < 2 or di >= n_truck:
                        continue
                    if li == cur_li and di == cur_di:
                        continue
                    # Flight feasibility.
                    launch_node = truck[li]
                    land_node = truck[di]
                    trip_dist = D[launch_node][cust] + D[cust][land_node]
                    if trip_dist > flight_limit:
                        continue
                    new_drone = drone_route[:trip_idx] + [(cust, li, di)] + drone_route[trip_idx + 1 :]
                    if not drone_route_is_feasible(new_drone):
                        continue

                    cand = [sol[0][:], sol[1][:], sol[2][:]]
                    cand[d_idx] = new_drone
                    feasible, cand_cost = evaluate(cand)
                    if not feasible:
                        continue
                    if cand_cost < best_cost - 1e-9:
                        best_cost = cand_cost
                        best_sol = cand

    if best_sol is None:
        return sol, current_cost, False
    return best_sol, best_cost, True


# ---------------------------------------------------------------------------
# Outer cycle
# ---------------------------------------------------------------------------

def local_search(
    sol,
    current_cost,
    evaluate,
    ctx,
    max_cycles=20,
    time_budget_seconds=None,
):
    """Run sweeps 1->2->3 in cycle until a full cycle finds no improvement,
    or until cycle/time caps hit. Returns (new_sol, new_cost, n_cycles_used)."""
    deadline = (time.perf_counter() + time_budget_seconds) if time_budget_seconds else None
    cur_sol = _clone(sol)
    cur_cost = current_cost
    cycles_used = 0

    for _ in range(max_cycles):
        if deadline is not None and time.perf_counter() > deadline:
            break
        cycles_used += 1
        any_improved = False

        cur_sol, cur_cost, imp1 = truck_2opt_sweep(cur_sol, evaluate, ctx, cur_cost, deadline)
        if imp1:
            any_improved = True
        if deadline is not None and time.perf_counter() > deadline:
            break

        cur_sol, cur_cost, imp2 = truck_drone_reassign_sweep(cur_sol, evaluate, ctx, cur_cost, deadline)
        if imp2:
            any_improved = True
        if deadline is not None and time.perf_counter() > deadline:
            break

        cur_sol, cur_cost, imp3 = drone_window_sweep(cur_sol, evaluate, ctx, cur_cost, deadline)
        if imp3:
            any_improved = True

        if not any_improved:
            break

    return cur_sol, cur_cost, cycles_used
