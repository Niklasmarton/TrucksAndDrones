from itertools import combinations
from pathlib import Path
import random
import sys

ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
CORE_DIR = ASSIGNMENT_DIR / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.append(str(CORE_DIR))

from operator_context import assert_context_is_set, get_operator_context, set_operator_context
from drone_route_utils import (
    build_drone_pair,
    drone_route_is_feasible,
    repair_drone_route,
    remap_drone_route_by_endpoint_nodes,
    insert_node_with_truck_fallback,
    enforce_wait_feasible_with_truck_fallback,
)

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None

try:
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp as _pywrapcp
    _ortools_available = True
except Exception:
    _ortools_available = False


_REMAP_REPAIR_ATTEMPTS = 12   # randomised remap+repair attempts per call
_FULL_REBUILD_ATTEMPTS = 6    # randomised full-rebuild attempts per call

# Diagnostic counters — reset between runs if needed.
_diag_calls = 0
_diag_tsp_improved = 0      # OR-Tools/2-opt found a strictly different truck route
_diag_returned_same = 0     # new_truck == old_truck → no-op
_diag_ortools_optimal = 0   # OR-Tools confirmed optimal (status=1)
_diag_ortools_timeout = 0   # OR-Tools hit time limit (status=3, may not be optimal)
_diag_ortools_fallback = 0  # OR-Tools not used; fell back to NN+2-opt
_diag_cost_improvements = []  # truck cost delta each time a different route was found


def reset_diagnostics():
    global _diag_calls, _diag_tsp_improved, _diag_returned_same
    global _diag_ortools_optimal, _diag_ortools_timeout, _diag_ortools_fallback
    global _diag_cost_improvements
    _diag_calls = 0
    _diag_tsp_improved = 0
    _diag_returned_same = 0
    _diag_ortools_optimal = 0
    _diag_ortools_timeout = 0
    _diag_ortools_fallback = 0
    _diag_cost_improvements = []


def get_diagnostics():
    improvements = _diag_cost_improvements
    return {
        "calls": _diag_calls,
        "tsp_improved": _diag_tsp_improved,
        "returned_same": _diag_returned_same,
        "improvement_rate": (_diag_tsp_improved / _diag_calls) if _diag_calls > 0 else 0.0,
        "ortools_optimal": _diag_ortools_optimal,
        "ortools_timeout": _diag_ortools_timeout,
        "ortools_fallback": _diag_ortools_fallback,
        "avg_truck_cost_improvement": (sum(improvements) / len(improvements)) if improvements else 0.0,
        "max_truck_cost_improvement": max(improvements) if improvements else 0.0,
    }


def set_search_progress(progress):
    return None


def _prefix_truck_times(truck_route, truck_times):
    prefix = [0.0]
    for i in range(len(truck_route) - 1):
        prefix.append(prefix[-1] + truck_times[truck_route[i]][truck_route[i + 1]])
    return prefix


def _sync_penalty(truck, drone1, drone2, truck_times, drone_times):
    """Synchronisation quality proxy — lower is better (identical to op10 scoring)."""
    prefix = _prefix_truck_times(truck, truck_times)
    pen = 0.0
    for node, launch_idx, land_idx in drone1 + drone2:
        launch_node = truck[launch_idx]
        land_node = truck[land_idx]
        sortie = drone_times[launch_node][node] + drone_times[node][land_node]
        truck_seg = prefix[land_idx] - prefix[launch_idx]
        pen += 3.0 * max(0.0, sortie - truck_seg) + 0.6 * abs(sortie - truck_seg) + 0.02 * sortie
    pen += 2.0 * abs(len(drone1) - len(drone2))
    return pen


def _total_arrival_time(truck, drone1, drone2, truck_times, drone_times, flight_range, depot):
    """Full objective: sum of all customer arrival times / 100. Mirrors
    CalCulateTotalArrivalTime semantics. Returns inf if infeasible."""
    n = len(truck)
    if n < 2:
        return float("inf")
    drone_return_map = {}
    for drone_id, route in enumerate([drone1, drone2]):
        for customer, launch_idx, return_idx in route:
            if not (0 <= launch_idx < n and 0 <= return_idx < n and launch_idx < return_idx):
                return float("inf")
            drone_return_map.setdefault(return_idx, []).append((drone_id, launch_idx, customer))
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
            total += drone_cust_arrival
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
            total += truck_arrival_time
    return total / 100.0


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


def _route_cost(route, truck_times):
    return sum(truck_times[route[i]][route[i + 1]] for i in range(len(route) - 1))


def _nearest_neighbor_route(nodes, truck_times, depot):
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


def _nearest_neighbor_from(start, nodes, truck_times):
    """NN tour starting at `start`, visiting `nodes`, returning to `start`."""
    unvisited = set(nodes)
    unvisited.discard(start)
    route = [start]
    current = start
    while unvisited:
        nxt = min(unvisited, key=lambda n: truck_times[current][n])
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    route.append(start)
    return route


def _multi_start_nn_2opt(nodes, truck_times, depot, n_extra_starts=3):
    """Multi-start NN + 2-opt. Tries depot + up to n_extra_starts random
    non-depot starts, 2-opts each, and keeps the best closed route at depot."""
    if depot not in nodes or len(nodes) <= 2:
        return None

    starts = [depot]
    others = [n for n in nodes if n != depot]
    if others:
        random.shuffle(others)
        starts.extend(others[:n_extra_starts])

    best_route = None
    best_cost = float("inf")
    for start in starts:
        tour = _nearest_neighbor_from(start, nodes, truck_times)
        # Rotate so depot is at the front (tour is a closed cycle, last == first).
        if start != depot:
            try:
                p = tour.index(depot)
            except ValueError:
                continue
            # tour is [start, ..., start]; cut last element, rotate, re-close at depot.
            cycle = tour[:-1]
            rotated = cycle[p:] + cycle[:p]
            tour = rotated + [depot]
        improved = _two_opt(tour, truck_times, max_passes=40)
        cost = _route_cost(improved, truck_times)
        if cost < best_cost:
            best_cost = cost
            best_route = improved
    return best_route


def _two_opt(route, truck_times, max_passes=30):
    best = route[:]
    n = len(best)
    passes = 0
    improved = True
    while improved and passes < max_passes:
        passes += 1
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                a = best[i - 1]
                b = best[i]
                c = best[j]
                d = best[j + 1]
                delta = truck_times[a][c] + truck_times[b][d] - truck_times[a][b] - truck_times[c][d]
                if delta < -1e-9:
                    best = best[:i] + list(reversed(best[i : j + 1])) + best[j + 1 :]
                    improved = True
                    break
            if improved:
                break
    return best


def _shortest_subtour(selected_edges, n):
    if not selected_edges:
        return []
    neighbors = {i: [] for i in range(n)}
    for i, j in selected_edges:
        neighbors[i].append(j)
        neighbors[j].append(i)

    unvisited = set(range(n))
    best_cycle = list(range(n + 1))
    while unvisited:
        start = unvisited.pop()
        cycle = [start]
        stack = [start]
        while stack:
            curr = stack.pop()
            for nxt in neighbors[curr]:
                if nxt in unvisited:
                    unvisited.remove(nxt)
                    stack.append(nxt)
                    cycle.append(nxt)
        if len(cycle) < len(best_cycle):
            best_cycle = cycle
    return best_cycle


def _solve_with_gurobi(nodes, truck_times, depot):
    if gp is None or GRB is None or len(nodes) <= 3:
        return None

    idx_of = {node: i for i, node in enumerate(nodes)}
    depot_idx = idx_of.get(depot)
    if depot_idx is None:
        return None

    n = len(nodes)
    dist = [[float(truck_times[a][b]) for b in nodes] for a in nodes]

    try:
        model = gp.Model("truck_tsp")
        model.Params.OutputFlag = 0
        model.Params.LazyConstraints = 1
        model.Params.TimeLimit = 2.0

        x = {}
        for i in range(n):
            for j in range(i + 1, n):
                x[i, j] = model.addVar(vtype=GRB.BINARY, obj=dist[i][j], name=f"x_{i}_{j}")

        for i in range(n):
            model.addConstr(
                gp.quicksum(x[min(i, j), max(i, j)] for j in range(n) if j != i) == 2,
                name=f"deg_{i}",
            )

        model._x = x
        model._n = n

        def _subtour_cb(m, where):
            if where != GRB.Callback.MIPSOL:
                return
            vals = m.cbGetSolution(m._x)
            selected = [(i, j) for (i, j), v in vals.items() if v > 0.5]
            tour = _shortest_subtour(selected, m._n)
            if len(tour) < m._n:
                m.cbLazy(
                    gp.quicksum(
                        m._x[min(i, j), max(i, j)] for i, j in combinations(tour, 2)
                    )
                    <= len(tour) - 1
                )

        model.optimize(_subtour_cb)
        # Accept optimal (2) or time-limit-with-feasible-solution (9).
        if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT) or model.SolCount == 0:
            return None

        selected = [(i, j) for (i, j), var in x.items() if var.X > 0.5]
        tour = _shortest_subtour(selected, n)
        if len(tour) != n:
            return None

        if depot_idx not in tour:
            return None
        p = tour.index(depot_idx)
        tour_fwd = tour[p:] + tour[:p]
        tour_rev = [tour_fwd[0]] + list(reversed(tour_fwd[1:]))

        route_fwd = [nodes[idx] for idx in tour_fwd] + [depot]
        route_rev = [nodes[idx] for idx in tour_rev] + [depot]
        if _route_cost(route_fwd, truck_times) <= _route_cost(route_rev, truck_times):
            return route_fwd
        return route_rev
    except Exception:
        return None


def _solve_with_ortools(nodes, truck_times, depot):
    if not _ortools_available or len(nodes) <= 2:
        return None, None

    idx_of = {node: i for i, node in enumerate(nodes)}
    depot_idx = idx_of.get(depot)
    if depot_idx is None:
        return None, None

    n = len(nodes)
    # Scale floats to integers (OR-Tools requires integer arc costs).
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
            return None, None

        # status: 1=optimal, 2=feasible(not optimal), 3=timeout, 4=infeasible
        status = routing.status()

        # Extract the ordered tour starting from depot.
        index = routing.Start(0)
        tour_indices = []
        while not routing.IsEnd(index):
            tour_indices.append(manager.IndexToNode(index))
            index = sol.Value(routing.NextVar(index))

        if len(tour_indices) != n:
            return None, None

        p = tour_indices.index(depot_idx)
        tour_fwd = tour_indices[p:] + tour_indices[:p]
        tour_rev = [tour_fwd[0]] + list(reversed(tour_fwd[1:]))

        route_fwd = [nodes[i] for i in tour_fwd] + [depot]
        route_rev = [nodes[i] for i in tour_rev] + [depot]
        if _route_cost(route_fwd, truck_times) <= _route_cost(route_rev, truck_times):
            return route_fwd, status
        return route_rev, status
    except Exception:
        return None, None


def _optimize_truck_route_with_tsp(truck, truck_times, depot):
    global _diag_ortools_optimal, _diag_ortools_timeout, _diag_ortools_fallback
    global _diag_cost_improvements

    if len(truck) <= 4:
        return truck[:]

    nodes = [depot] + [n for n in truck[1:-1] if n != depot]
    if len(nodes) <= 2:
        return truck[:]

    old_cost = _route_cost(truck, truck_times)

    # Multi-start NN + 2-opt: empirically beats OR-Tools on R-type instances
    # (random spatial layouts) because its stochastic variation between calls
    # gives op10 multiple route geometries to pick from over the run, rather
    # than always converging to the single truck-distance-optimal route.
    msr = _multi_start_nn_2opt(nodes, truck_times, depot, n_extra_starts=3)
    if msr is not None:
        new_cost = _route_cost(msr, truck_times)
        if new_cost < old_cost:
            _diag_cost_improvements.append(old_cost - new_cost)
        _diag_ortools_fallback += 1  # repurposed as "non-OR-Tools path used"
        return msr

    # Last-resort fallback: single-start NN + 2-opt.
    heuristic = _nearest_neighbor_route(nodes, truck_times, depot)
    heuristic = _two_opt(heuristic, truck_times, max_passes=40)
    new_cost = _route_cost(heuristic, truck_times)
    if new_cost < old_cost:
        _diag_cost_improvements.append(old_cost - new_cost)
    return heuristic


def _preferred_endpoints_by_customer(old_truck, drone1, drone2):
    pref = {}
    for route in (drone1, drone2):
        for node, launch_idx, land_idx in route:
            if not (0 <= launch_idx < len(old_truck) and 0 <= land_idx < len(old_truck)):
                continue
            pref[node] = (old_truck[launch_idx], old_truck[land_idx])
    return pref


def _map_preferred_pair(pref_endpoint_nodes, new_truck):
    if pref_endpoint_nodes is None:
        return None
    launch_node, land_node = pref_endpoint_nodes
    depot = new_truck[0]
    # Depot appears twice (index 0 and len-1); .index() always returns 0,
    # so handle each depot endpoint explicitly to preserve the launch=0 /
    # land=len-1 cases across TSP rebuilds.
    if launch_node == depot:
        launch_idx = 0
    elif launch_node in new_truck:
        launch_idx = new_truck.index(launch_node)
    else:
        return None
    if land_node == depot:
        land_idx = len(new_truck) - 1
    elif land_node in new_truck:
        land_idx = new_truck.index(land_node)
    else:
        return None
    if launch_idx >= land_idx:
        return None
    return (launch_idx, land_idx)


def _pair_endpoints_available(route, launch_idx, land_idx):
    for _, l, r in route:
        if l == launch_idx or r == land_idx:
            return False
    return True


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
    return best_idx


def _shift_route_after_truck_insert(route, insert_idx):
    shifted = []
    for node, launch_idx, land_idx in route:
        if launch_idx >= insert_idx:
            launch_idx += 1
        if land_idx >= insert_idx:
            land_idx += 1
        shifted.append((node, launch_idx, land_idx))
    shifted.sort(key=lambda x: (x[1], x[2], x[0]))
    return shifted


def _try_insert_into_drone_route(node, truck, route, preferred_by_customer):
    preferred_pair = _map_preferred_pair(preferred_by_customer.get(node), truck)
    for ins_idx in range(len(route) + 1):
        pair = build_drone_pair(
            node,
            truck,
            route,
            ins_idx,
            preferred_pair=preferred_pair,
            max_neighbors=8,
            pair_explore_prob=0.0,
        )
        if pair is None:
            continue
        launch_idx, land_idx = pair
        if not _pair_endpoints_available(route, launch_idx, land_idx):
            continue
        route.insert(ins_idx, (node, launch_idx, land_idx))
        route.sort(key=lambda x: (x[1], x[2], x[0]))
        return True
    return False


def _insert_customer_with_fallback(
    node,
    truck,
    primary_route,
    secondary_route,
    preferred_by_customer,
    truck_times,
):
    if _try_insert_into_drone_route(node, truck, primary_route, preferred_by_customer):
        return truck, primary_route, secondary_route, True

    if _try_insert_into_drone_route(node, truck, secondary_route, preferred_by_customer):
        return truck, primary_route, secondary_route, True

    if node in truck:
        return truck, primary_route, secondary_route, True

    ins_idx = _best_truck_insert(node, truck, truck_times)
    if ins_idx is None:
        return truck, primary_route, secondary_route, False

    truck = truck[:ins_idx] + [node] + truck[ins_idx:]
    primary_route = _shift_route_after_truck_insert(primary_route, ins_idx)
    secondary_route = _shift_route_after_truck_insert(secondary_route, ins_idx)
    return truck, primary_route, secondary_route, True


def _rebuild_drones_with_utils(truck, old_truck, drone1_old, drone2_old, truck_times, pair_explore_prob=0.0):
    customers1 = [node for node, _, _ in drone1_old]
    customers2 = [node for node, _, _ in drone2_old]
    preferred_by_customer = _preferred_endpoints_by_customer(old_truck, drone1_old, drone2_old)
    truck = truck[:]
    drone1 = []
    drone2 = []

    for node in customers1:
        preferred_pair = _map_preferred_pair(preferred_by_customer.get(node), truck)
        inserted = insert_node_with_truck_fallback(
            node,
            truck,
            drone1,
            drone2,
            preferred_pair=preferred_pair,
            max_neighbors=16,
            pair_explore_prob=pair_explore_prob,
        )
        if inserted is None:
            return None
        truck, drone1, drone2, _ = inserted

    for node in customers2:
        preferred_pair = _map_preferred_pair(preferred_by_customer.get(node), truck)
        inserted = insert_node_with_truck_fallback(
            node,
            truck,
            drone2,
            drone1,
            preferred_pair=preferred_pair,
            max_neighbors=16,
            pair_explore_prob=pair_explore_prob,
        )
        if inserted is None:
            return None
        truck, drone2, drone1, _ = inserted

    drone1 = repair_drone_route(truck, drone1, max_repairs=max(48, 5 * len(drone1)), max_neighbors=16)
    if drone1 is None:
        return None
    drone2 = repair_drone_route(truck, drone2, max_repairs=max(48, 5 * len(drone2)), max_neighbors=16)
    if drone2 is None:
        return None
    truck, drone1, drone2, ok = enforce_wait_feasible_with_truck_fallback(
        truck,
        drone1,
        drone2,
        max_iterations=max(400, 16 * (len(drone1) + len(drone2) + 1)),
    )
    if not ok:
        return None
    if not _route_endpoint_unique(drone1) or not _route_endpoint_unique(drone2):
        return None
    return [truck, drone1, drone2]


def _clean_remap(new_truck, old_truck, drone1_old, drone2_old):
    """
    Remap drone assignments to the new truck order without any repair.
    Returns the candidate only when the remap is clean — no trip flips and
    both routes are already feasible.  A clean remap means the drone structure
    is fully preserved, so a TSP-shorter truck route is a strict improvement.
    """
    drone1 = remap_drone_route_by_endpoint_nodes(old_truck, new_truck, drone1_old)
    drone2 = remap_drone_route_by_endpoint_nodes(old_truck, new_truck, drone2_old)
    if drone1 is None or drone2 is None:
        return None
    if not drone_route_is_feasible(drone1) or not drone_route_is_feasible(drone2):
        return None
    if not _route_endpoint_unique(drone1) or not _route_endpoint_unique(drone2):
        return None
    return [new_truck[:], drone1, drone2]


def _remap_and_repair(new_truck, old_truck, drone1_old, drone2_old, pair_explore_prob=0.15):
    """
    Remap existing drone assignments to the new truck order, then repair any
    trips that became infeasible.  pair_explore_prob adds randomness so
    multiple calls can produce different candidates.
    """
    drone1 = remap_drone_route_by_endpoint_nodes(old_truck, new_truck, drone1_old)
    drone2 = remap_drone_route_by_endpoint_nodes(old_truck, new_truck, drone2_old)
    if drone1 is None or drone2 is None:
        return None

    drone1 = repair_drone_route(new_truck, drone1, max_repairs=max(40, 4 * len(drone1)), max_neighbors=14)
    if drone1 is None:
        return None
    drone2 = repair_drone_route(new_truck, drone2, max_repairs=max(40, 4 * len(drone2)), max_neighbors=14)
    if drone2 is None:
        return None

    truck, drone1, drone2, ok = enforce_wait_feasible_with_truck_fallback(
        new_truck, drone1, drone2,
        max_iterations=max(300, 12 * (len(drone1) + len(drone2) + 1)),
    )
    if not ok:
        return None
    if not _route_endpoint_unique(drone1) or not _route_endpoint_unique(drone2):
        return None
    return [truck, drone1, drone2]


def operator(current_solution):
    global _diag_calls, _diag_tsp_improved, _diag_returned_same
    assert_context_is_set()
    truck_times, drone_times, flight_range, depot = get_operator_context()

    old_truck, drone1_old, drone2_old = _clone_solution(current_solution)
    _diag_calls += 1

    current_obj = _total_arrival_time(
        old_truck, drone1_old, drone2_old, truck_times, drone_times, flight_range, depot
    )

    new_truck = _optimize_truck_route_with_tsp(old_truck, truck_times, depot)
    if not new_truck or new_truck == old_truck:
        _diag_returned_same += 1
        return current_solution
    _diag_tsp_improved += 1

    # B: short-circuit on clean_remap success. When the TSP reorder didn't flip
    # any drone endpoint pair, the drone structure is preserved exactly. If the
    # resulting solution beats current, return it immediately — the 18 noisy
    # rebuild attempts can't do better than a structurally-preserved improvement.
    candidates = []
    clean = _clean_remap(new_truck, old_truck, drone1_old, drone2_old)
    if clean is not None:
        clean_obj = _total_arrival_time(
            clean[0], clean[1], clean[2], truck_times, drone_times, flight_range, depot
        )
        if clean_obj < current_obj:
            return clean
        if clean_obj != float("inf"):
            candidates.append((clean_obj, clean))

    def _try(c):
        if c is not None and c != current_solution:
            obj = _total_arrival_time(c[0], c[1], c[2], truck_times, drone_times, flight_range, depot)
            if obj != float("inf"):
                candidates.append((obj, c))

    # Strategy 2: remap+repair — run many times with randomness; tries to keep
    # inverted-trip customers on drones by searching for new valid pairs.
    for _ in range(_REMAP_REPAIR_ATTEMPTS):
        _try(_remap_and_repair(new_truck, old_truck, drone1_old, drone2_old))

    # Strategy 3: full rebuild — run several times with increasing explore prob.
    for i in range(_FULL_REBUILD_ATTEMPTS):
        explore = 0.0 if i == 0 else 0.1 + 0.05 * i
        _try(_rebuild_drones_with_utils(new_truck, old_truck, drone1_old, drone2_old, truck_times, pair_explore_prob=explore))

    if not candidates:
        return current_solution

    candidates.sort(key=lambda x: x[0])
    # A: only return improving candidates. If the best we found is worse than
    # current, return current_solution so ALNS doesn't see a forced uphill move.
    if candidates[0][0] >= current_obj:
        return current_solution
    return candidates[0][1]
