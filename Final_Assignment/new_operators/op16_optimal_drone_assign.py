"""
Op16: Optimal drone assignment local search — small instances only (n ≤ 15).

The structural basin problem on small R-type instances (e.g. R_10 finds 631 but
optimal is 595) is caused by the algorithm settling into a drone assignment that
looks locally good but sits in a different basin from the global optimum.
Normal single-node operators cannot cross the valley because every intermediate
step looks worse.

This operator does a systematic steepest-descent search over drone assignments:
  1. Try all single-customer vehicle reassignments (truck→drone1, truck→drone2,
     drone1→truck, drone2→truck, drone1↔drone2).
  2. Score each candidate using the actual makespan simulation (not a proxy).
  3. Commit the globally-best improving move, then repeat until no improvement.
  4. Also tries a fresh greedy assignment from the TSP-optimal backbone to escape
     basins that are unreachable via single moves from the current solution.

For n ≤ 10 the full greedy rebuild is also attempted from multiple random truck
orderings.  For n ≤ 15 the local search alone is very fast (~50 iterations,
~45 candidates each).

Only activated when the total number of customers ≤ _MAX_CUSTOMERS.
"""
import random
from pathlib import Path
import sys

ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
CORE_DIR = ASSIGNMENT_DIR / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.append(str(CORE_DIR))

from drone_route_utils import build_drone_pair, drone_route_is_feasible
from operator_context import assert_context_is_set, get_operator_context

try:
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp as _pywrapcp
    _ortools_available = True
except Exception:
    _ortools_available = False

_MAX_CUSTOMERS = 15
_MAX_LS_ITERS = 60
_GREEDY_RANDOM_RESTARTS = 3   # extra random-order restarts for n ≤ 10

# Op16 is phase-aware: it skips the early search phase to let the ALNS freely
# explore different truck route structures.  If op16 fires in the very first
# iterations it converges the drone assignment to a local optimum (e.g. 807 on
# R_10) that traps the entire run.  Withholding op16 until 25% of iterations
# have passed lets op1/op11/op12 establish a good truck backbone first.
_SEARCH_PROGRESS = 0.0
_EARLY_SKIP_THRESHOLD = 0.25


def set_search_progress(progress):
    global _SEARCH_PROGRESS
    _SEARCH_PROGRESS = max(0.0, min(1.0, float(progress)))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _clone_solution(solution):
    return [solution[0][:], solution[1][:], solution[2][:]]


def _route_endpoint_unique(route):
    used_launch = set()
    used_land = set()
    for _, launch_idx, land_idx in route:
        if launch_idx in used_launch or land_idx in used_land:
            return False
        used_launch.add(launch_idx)
        used_land.add(land_idx)
    return True


def _extract_customers(solution, depot):
    """Return sorted list of all customer nodes (truck + both drones)."""
    truck, drone1, drone2 = solution
    customers = set()
    for node in truck:
        if node != depot:
            customers.add(node)
    for node, _, _ in drone1 + drone2:
        customers.add(node)
    return sorted(customers)


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def _total_arrival_time(solution, truck_times, drone_times, depot, flight_range):
    """
    Compute the actual objective: sum of all customer arrival times / 100.

    Matches CalCulateTotalArrivalTime exactly, including the flight-range
    feasibility check: if any drone trip's effective flight time (including
    hover wait) exceeds flight_range, returns float("inf").

    - Every non-depot truck stop contributes its truck arrival time.
    - Every drone customer contributes the time the drone reaches that customer.
    - Drone availability cascades: a drone cannot launch until it has returned
      from its previous trip.
    - The truck waits at each landing position until all drones that land there
      have returned (their return time is based on t_arrival at the launch node
      and drone availability).
    """
    truck, drone1, drone2 = solution
    n = len(truck)
    if n < 2:
        return float("inf")

    # Build drone return map: position i -> list of (drone_id, launch_idx, customer)
    drone_return_map = {}
    for drone_id, route in enumerate([drone1, drone2]):
        for customer, launch_idx, return_idx in route:
            if not (0 <= launch_idx < n and 0 <= return_idx < n and launch_idx < return_idx):
                return float("inf")
            drone_return_map.setdefault(return_idx, []).append((drone_id, launch_idx, customer))

    # Simulate truck movement (node-keyed, matching CalCulateTotalArrivalTime).
    t_arrival = {truck[0]: 0.0}
    t_departure = {truck[0]: 0.0}
    drone_availability = [0.0, 0.0]
    total = 0.0

    for i in range(1, n):
        prev_node = truck[i - 1]
        curr_node = truck[i]

        truck_arrival_time = t_departure[prev_node] + truck_times[prev_node][curr_node]
        t_arrival[curr_node] = truck_arrival_time

        drone_returns = []
        for drone_id, launch_idx, customer in drone_return_map.get(i, []):
            launch_node = truck[launch_idx]
            flight_out = drone_times[launch_node][customer]
            flight_back = drone_times[customer][curr_node]
            total_flight = flight_out + flight_back

            actual_launch = max(t_arrival[launch_node], drone_availability[drone_id])
            drone_cust_arrival = actual_launch + flight_out
            drone_return_time = actual_launch + total_flight
            drone_availability[drone_id] = drone_return_time
            drone_returns.append(drone_return_time)
            total += drone_cust_arrival       # drone customer arrival time

            # Check flight-range feasibility (matches CalCulateTotalArrivalTime)
            if curr_node != depot:
                drone_wait = max(truck_arrival_time - drone_return_time, 0.0)
            else:
                drone_wait = 0.0
            if total_flight + drone_wait > flight_range:
                return float("inf")

        if drone_returns:
            t_departure[curr_node] = max(truck_arrival_time, max(drone_returns))
        else:
            t_departure[curr_node] = truck_arrival_time

        if curr_node != depot:
            total += truck_arrival_time       # truck customer arrival time

    return total / 100.0


# ---------------------------------------------------------------------------
# TSP backbone
# ---------------------------------------------------------------------------

def _route_cost(route, truck_times):
    return sum(truck_times[route[i]][route[i + 1]] for i in range(len(route) - 1))


def _nearest_neighbor(nodes, truck_times, depot):
    unvisited = set(nodes)
    unvisited.discard(depot)
    route = [depot]
    current = depot
    while unvisited:
        nxt = min(unvisited, key=lambda n: truck_times[current][n])
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    route.append(depot)
    return route


def _two_opt(route, truck_times, max_passes=50):
    best = route[:]
    n = len(best)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        passes += 1
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                a, b = best[i - 1], best[i]
                c, d = best[j], best[j + 1]
                delta = (truck_times[a][c] + truck_times[b][d]
                         - truck_times[a][b] - truck_times[c][d])
                if delta < -1e-9:
                    best = best[:i] + list(reversed(best[i:j + 1])) + best[j + 1:]
                    improved = True
                    break
            if improved:
                break
    return best


def _solve_with_ortools_tsp(nodes, truck_times, depot):
    """Solve TSP for nodes; return route starting/ending at depot, or None."""
    if not _ortools_available or len(nodes) <= 2:
        return None

    idx_of = {node: i for i, node in enumerate(nodes)}
    depot_idx = idx_of.get(depot)
    if depot_idx is None:
        return None

    n = len(nodes)
    scale = 1000
    dist = [[int(truck_times[a][b] * scale) for b in nodes] for a in nodes]

    try:
        manager = _pywrapcp.RoutingIndexManager(n, 1, depot_idx)
        routing = _pywrapcp.RoutingModel(manager)

        def _dist_cb(from_idx, to_idx):
            return dist[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]

        cb_idx = routing.RegisterTransitCallback(_dist_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(cb_idx)

        params = _pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC
        )
        params.time_limit.seconds = 1

        sol = routing.SolveWithParameters(params)
        if sol is None:
            return None

        index = routing.Start(0)
        tour_indices = []
        while not routing.IsEnd(index):
            tour_indices.append(manager.IndexToNode(index))
            index = sol.Value(routing.NextVar(index))

        if len(tour_indices) != n:
            return None

        p = tour_indices.index(depot_idx)
        tour_fwd = tour_indices[p:] + tour_indices[:p]
        tour_rev = [tour_fwd[0]] + list(reversed(tour_fwd[1:]))

        route_fwd = [nodes[i] for i in tour_fwd] + [depot]
        route_rev = [nodes[i] for i in tour_rev] + [depot]
        if _route_cost(route_fwd, truck_times) <= _route_cost(route_rev, truck_times):
            return route_fwd
        return route_rev
    except Exception:
        return None


def _get_backbone(customers, truck_times, depot):
    """Return TSP-optimal truck route visiting all customers."""
    nodes = [depot] + customers
    route = _solve_with_ortools_tsp(nodes, truck_times, depot)
    if route is None:
        route = _nearest_neighbor(nodes, truck_times, depot)
        route = _two_opt(route, truck_times)
    return route


# ---------------------------------------------------------------------------
# Index-shifting helpers
# ---------------------------------------------------------------------------

def _shift_after_removal(route, removed_idx):
    """
    Shift all launch/land indices after removing truck[removed_idx].
    Returns None if removed_idx is a drone endpoint (invalid removal).
    """
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


def _shift_after_insertion(route, insert_idx):
    """Shift all launch/land indices after inserting at truck[insert_idx]."""
    shifted = []
    for node, launch_idx, land_idx in route:
        if launch_idx >= insert_idx:
            launch_idx += 1
        if land_idx >= insert_idx:
            land_idx += 1
        shifted.append((node, launch_idx, land_idx))
    shifted.sort(key=lambda x: (x[1], x[2], x[0]))
    return shifted


def _best_truck_insert_idx(node, truck, truck_times):
    best_delta = float("inf")
    best_idx = 1
    for ins_idx in range(1, len(truck)):
        a, b = truck[ins_idx - 1], truck[ins_idx]
        delta = truck_times[a][node] + truck_times[node][b] - truck_times[a][b]
        if delta < best_delta:
            best_delta = delta
            best_idx = ins_idx
    return best_idx


# ---------------------------------------------------------------------------
# Single-move generators
# ---------------------------------------------------------------------------

def _try_truck_to_drone(solution, cust_idx, target_drone_id, truck_times, drone_times, depot, flight_range):
    """
    Move truck[cust_idx] to target_drone_id.  Returns best-scoring feasible
    candidate solution, or None if no feasible insertion exists.
    """
    truck, drone1, drone2 = solution
    if cust_idx <= 0 or cust_idx >= len(truck) - 1:
        return None

    node = truck[cust_idx]
    new_truck = truck[:cust_idx] + truck[cust_idx + 1:]

    new_d1 = _shift_after_removal(drone1, cust_idx)
    new_d2 = _shift_after_removal(drone2, cust_idx)
    if new_d1 is None or new_d2 is None:
        return None

    target = new_d1 if target_drone_id == 1 else new_d2
    other = new_d2 if target_drone_id == 1 else new_d1

    if not drone_route_is_feasible(other) or not _route_endpoint_unique(other):
        return None

    best_sol = None
    best_score = float("inf")

    for ins_idx in range(len(target) + 1):
        pair = build_drone_pair(
            node, new_truck, target, ins_idx,
            max_neighbors=12, pair_explore_prob=0.0,
        )
        if pair is None:
            continue

        new_target = target[:]
        new_target.insert(ins_idx, (node, pair[0], pair[1]))
        new_target.sort(key=lambda x: (x[1], x[2], x[0]))

        if not drone_route_is_feasible(new_target):
            continue
        if not _route_endpoint_unique(new_target):
            continue

        cand = ([new_truck, new_target, other] if target_drone_id == 1
                else [new_truck, other, new_target])
        score = _total_arrival_time(cand, truck_times, drone_times, depot, flight_range)
        if score < best_score:
            best_score = score
            best_sol = cand

    return best_sol


def _try_drone_to_truck(solution, drone_id, trip_idx, truck_times, drone_times):
    """Move drone route trip at trip_idx back to the truck."""
    truck, drone1, drone2 = solution
    route = drone1 if drone_id == 1 else drone2

    if trip_idx < 0 or trip_idx >= len(route):
        return None

    node, _, _ = route[trip_idx]
    new_route = route[:trip_idx] + route[trip_idx + 1:]

    ins_idx = _best_truck_insert_idx(node, truck, truck_times)
    new_truck = truck[:ins_idx] + [node] + truck[ins_idx:]

    if drone_id == 1:
        shifted_d1 = _shift_after_insertion(new_route, ins_idx)
        shifted_d2 = _shift_after_insertion(drone2, ins_idx)
    else:
        shifted_d1 = _shift_after_insertion(drone1, ins_idx)
        shifted_d2 = _shift_after_insertion(new_route, ins_idx)

    if not drone_route_is_feasible(shifted_d1) or not _route_endpoint_unique(shifted_d1):
        return None
    if not drone_route_is_feasible(shifted_d2) or not _route_endpoint_unique(shifted_d2):
        return None

    return [new_truck, shifted_d1, shifted_d2]


def _try_drone_to_drone(solution, from_drone_id, trip_idx, truck_times, drone_times, depot, flight_range):
    """
    Move drone trip at trip_idx from from_drone_id to the other drone.
    The truck does not change; only the drone assignment changes.
    """
    truck, drone1, drone2 = solution
    from_route = drone1 if from_drone_id == 1 else drone2
    to_route = drone2 if from_drone_id == 1 else drone1

    if trip_idx < 0 or trip_idx >= len(from_route):
        return None

    node, _, _ = from_route[trip_idx]
    new_from = from_route[:trip_idx] + from_route[trip_idx + 1:]

    if not drone_route_is_feasible(new_from) or not _route_endpoint_unique(new_from):
        return None

    best_sol = None
    best_score = float("inf")

    for ins_idx in range(len(to_route) + 1):
        pair = build_drone_pair(
            node, truck, to_route, ins_idx,
            max_neighbors=12, pair_explore_prob=0.0,
        )
        if pair is None:
            continue

        new_to = to_route[:]
        new_to.insert(ins_idx, (node, pair[0], pair[1]))
        new_to.sort(key=lambda x: (x[1], x[2], x[0]))

        if not drone_route_is_feasible(new_to):
            continue
        if not _route_endpoint_unique(new_to):
            continue

        cand = ([truck[:], new_from, new_to] if from_drone_id == 1
                else [truck[:], new_to, new_from])
        score = _total_arrival_time(cand, truck_times, drone_times, depot, flight_range)
        if score < best_score:
            best_score = score
            best_sol = cand

    return best_sol


# ---------------------------------------------------------------------------
# Local search
# ---------------------------------------------------------------------------

def _local_search(solution, truck_times, drone_times, depot, flight_range, max_iters=_MAX_LS_ITERS):
    """
    Steepest-descent local search over drone assignments.
    Each iteration finds the single move (across all move types) that most
    reduces the total arrival time, then commits it.  Stops when no improving
    move exists.
    """
    best = _clone_solution(solution)
    best_score = _total_arrival_time(best, truck_times, drone_times, depot, flight_range)

    for _ in range(max_iters):
        best_cand = None
        best_cand_score = best_score

        truck, drone1, drone2 = best

        # truck → drone moves
        for cust_idx in range(1, len(truck) - 1):
            for drone_id in (1, 2):
                cand = _try_truck_to_drone(best, cust_idx, drone_id, truck_times, drone_times, depot, flight_range)
                if cand is None:
                    continue
                score = _total_arrival_time(cand, truck_times, drone_times, depot, flight_range)
                if score < best_cand_score - 1e-6:
                    best_cand_score = score
                    best_cand = cand

        # drone → truck moves
        for drone_id, route in ((1, drone1), (2, drone2)):
            for trip_idx in range(len(route)):
                cand = _try_drone_to_truck(best, drone_id, trip_idx, truck_times, drone_times)
                if cand is None:
                    continue
                score = _total_arrival_time(cand, truck_times, drone_times, depot, flight_range)
                if score < best_cand_score - 1e-6:
                    best_cand_score = score
                    best_cand = cand

        # drone ↔ drone moves (reassign customer between the two drones)
        for from_id, route in ((1, drone1), (2, drone2)):
            for trip_idx in range(len(route)):
                cand = _try_drone_to_drone(best, from_id, trip_idx, truck_times, drone_times, depot, flight_range)
                if cand is None:
                    continue
                score = _total_arrival_time(cand, truck_times, drone_times, depot, flight_range)
                if score < best_cand_score - 1e-6:
                    best_cand_score = score
                    best_cand = cand

        if best_cand is None:
            break

        best = best_cand
        best_score = best_cand_score

    return best


# ---------------------------------------------------------------------------
# Greedy rebuild from backbone
# ---------------------------------------------------------------------------

def _greedy_assign_from_backbone(backbone, truck_times, drone_times, depot, flight_range):
    """
    Starting with all customers on the truck in backbone order, greedily move
    customers to drones whenever it reduces the total arrival time.
    """
    truck = backbone[:]
    drone1, drone2 = [], []

    current_sol = [truck, drone1, drone2]
    current_score = _total_arrival_time(current_sol, truck_times, drone_times, depot, flight_range)

    changed = True
    while changed:
        changed = False
        truck = current_sol[0]
        for cust_idx in range(1, len(truck) - 1):
            for drone_id in (1, 2):
                cand = _try_truck_to_drone(current_sol, cust_idx, drone_id, truck_times, drone_times, depot, flight_range)
                if cand is None:
                    continue
                score = _total_arrival_time(cand, truck_times, drone_times, depot, flight_range)
                if score < current_score - 1e-6:
                    current_sol = cand
                    current_score = score
                    changed = True
                    break
            if changed:
                break

    return current_sol


# ---------------------------------------------------------------------------
# Main operator
# ---------------------------------------------------------------------------

def operator(current_solution):
    assert_context_is_set()
    truck_times, drone_times, flight_range, depot = get_operator_context()

    # Phase guard — skip the early search so the ALNS can build a good truck
    # backbone before op16 locks in a drone assignment local optimum.
    if _SEARCH_PROGRESS < _EARLY_SKIP_THRESHOLD:
        return current_solution

    # Size guard — only run on small instances
    all_customers = _extract_customers(current_solution, depot)
    n = len(all_customers)
    if n > _MAX_CUSTOMERS:
        return current_solution
    if n == 0:
        return current_solution

    current_score = _total_arrival_time(current_solution, truck_times, drone_times, depot, flight_range)
    best = current_solution
    best_score = current_score

    def _track(sol):
        nonlocal best, best_score
        if sol is None:
            return
        s = _total_arrival_time(sol, truck_times, drone_times, depot, flight_range)
        if s < best_score - 1e-6:
            best = sol
            best_score = s

    # Strategy 1: local search from current solution
    _track(_local_search(current_solution, truck_times, drone_times, depot, flight_range))

    # Strategy 2: TSP-optimal backbone → greedy drone assignment → local search
    backbone = _get_backbone(all_customers, truck_times, depot)
    if backbone:
        greedy = _greedy_assign_from_backbone(backbone, truck_times, drone_times, depot, flight_range)
        _track(greedy)
        _track(_local_search(greedy, truck_times, drone_times, depot, flight_range))

    # Strategy 3: for very small instances, also try random backbone orderings
    if n <= 10:
        for _ in range(_GREEDY_RANDOM_RESTARTS):
            shuffled = all_customers[:]
            random.shuffle(shuffled)
            rand_backbone = [depot] + shuffled + [depot]
            rand_backbone = _two_opt(rand_backbone, truck_times)
            greedy = _greedy_assign_from_backbone(rand_backbone, truck_times, drone_times, depot, flight_range)
            _track(greedy)
            _track(_local_search(greedy, truck_times, drone_times, depot, flight_range))

    return best
