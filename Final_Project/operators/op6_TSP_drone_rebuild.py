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


_REMAP_REPAIR_ATTEMPTS = 12
_FULL_REBUILD_ATTEMPTS = 6


def set_search_progress(progress):
    return None


def _total_arrival_time(truck, drone1, drone2, truck_times, drone_times, flight_range, depot):
    # full objective, same as CalCulateTotalArrivalTime. inf if infeasible.
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


# NN tour from `start`, visiting all nodes, returning to `start`
def _nearest_neighbor_from(start, nodes, truck_times):
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


# multi-start NN + 2-opt. tries depot plus a few random non-depot starts,
# runs 2-opt on each, keeps the cheapest closed route through the depot.
def _multi_start_nn_2opt(nodes, truck_times, depot, n_extra_starts=3):
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
        # rotate so depot is at the front (tour is a closed cycle)
        if start != depot:
            try:
                p = tour.index(depot)
            except ValueError:
                continue
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


def _optimize_truck_route_with_tsp(truck, truck_times, depot):
    if len(truck) <= 4:
        return truck[:]

    nodes = [depot] + [n for n in truck[1:-1] if n != depot]
    if len(nodes) <= 2:
        return truck[:]

    # multi-start NN + 2-opt gives op10 different route geometries
    # to pick from across calls
    msr = _multi_start_nn_2opt(nodes, truck_times, depot, n_extra_starts=3)
    if msr is not None:
        return msr

    heuristic = _nearest_neighbor_route(nodes, truck_times, depot)
    return _two_opt(heuristic, truck_times, max_passes=40)


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
    # depot is at both index 0 and len-1, but .index() always returns 0,
    # so handle each end explicitly to keep the depot-launch / depot-land cases
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


# remap drone assignments to the new truck order, no repair.
# only returns the candidate if everything still lines up cleanly.
def _clean_remap(new_truck, old_truck, drone1_old, drone2_old):
    drone1 = remap_drone_route_by_endpoint_nodes(old_truck, new_truck, drone1_old)
    drone2 = remap_drone_route_by_endpoint_nodes(old_truck, new_truck, drone2_old)
    if drone1 is None or drone2 is None:
        return None
    if not drone_route_is_feasible(drone1) or not drone_route_is_feasible(drone2):
        return None
    if not _route_endpoint_unique(drone1) or not _route_endpoint_unique(drone2):
        return None
    return [new_truck[:], drone1, drone2]


# remap drones to the new truck order and repair anything that broke.
# pair_explore_prob adds randomness so repeat calls return different candidates.
def _remap_and_repair(new_truck, old_truck, drone1_old, drone2_old, pair_explore_prob=0.15):
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
    assert_context_is_set()
    truck_times, drone_times, flight_range, depot = get_operator_context()

    old_truck, drone1_old, drone2_old = _clone_solution(current_solution)

    current_obj = _total_arrival_time(
        old_truck, drone1_old, drone2_old, truck_times, drone_times, flight_range, depot
    )

    new_truck = _optimize_truck_route_with_tsp(old_truck, truck_times, depot)
    if not new_truck or new_truck == old_truck:
        return current_solution

    # if the clean remap works the drone structure is identical,
    # so a shorter truck route is a strict improvement, skip the rebuilds.
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

    # remap + repair: try to keep drone customers on drones with new pairs
    for _ in range(_REMAP_REPAIR_ATTEMPTS):
        _try(_remap_and_repair(new_truck, old_truck, drone1_old, drone2_old))

    # full rebuild with growing explore probability
    for i in range(_FULL_REBUILD_ATTEMPTS):
        explore = 0.0 if i == 0 else 0.1 + 0.05 * i
        _try(_rebuild_drones_with_utils(new_truck, old_truck, drone1_old, drone2_old, truck_times, pair_explore_prob=explore))

    if not candidates:
        return current_solution

    candidates.sort(key=lambda x: x[0])
    # only return improving candidates so ALNS doesn't see a forced uphill move
    if candidates[0][0] >= current_obj:
        return current_solution
    return candidates[0][1]
