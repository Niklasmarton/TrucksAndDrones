import copy
import random

# Operator context must be set by the algorithm that uses the operator
# (e.g., local_search.py / simulated_annealing.py).
T = None
D = None
flight_limit = None
depot = 0


def set_operator_context(truck_times, drone_times, flight_range, depot_index=0):
    global T, D, flight_limit, depot
    T = truck_times
    D = drone_times
    flight_limit = flight_range
    depot = depot_index


def _assert_context_is_set():
    if T is None or D is None or flight_limit is None:
        raise ValueError(
            "Operator context is not set. Call set_operator_context(T, D, flight_limit, depot) "
            "before using operator()."
        )


# Solution format: [truck, drone1, drone2]
# Each drone route is list[tuple(node, launch_idx, land_idx)]


def _internal_truck_indices(truck_route):
    return list(range(1, len(truck_route) - 1))


def _pair_fits_in_drone(drone_route, insertion_index, launch_idx, land_idx):
    if launch_idx >= land_idx:
        return False

    prev_land = 0 if insertion_index == 0 else drone_route[insertion_index - 1][2]
    next_launch = float("inf") if insertion_index == len(drone_route) else drone_route[insertion_index][1]

    return launch_idx >= prev_land and land_idx <= next_launch


def build_drone_pair(node, truck_route, drone_route, insertion_index, preferred_pair=None, max_neighbors=5):
    """
    Returns (launch_idx, land_idx) for node at given insertion position in a drone route.
    Enforces increasing index order and timeline feasibility at insertion point.
    """
    internal_idx = _internal_truck_indices(truck_route)
    if len(internal_idx) < 2:
        return None

    # Try keeping existing pair first (for drone->drone moves).
    if preferred_pair is not None:
        launch_idx, land_idx = preferred_pair
        if 0 <= launch_idx < len(truck_route) and 0 <= land_idx < len(truck_route):
            launch_city = truck_route[launch_idx]
            land_city = truck_route[land_idx]
            trip = D[launch_city][node] + D[node][land_city]
            if trip <= flight_limit and _pair_fits_in_drone(drone_route, insertion_index, launch_idx, land_idx):
                return launch_idx, land_idx

    # Nearby truck indices around served node first.
    ranked_idx = sorted(internal_idx, key=lambda idx: T[node][truck_route[idx]])
    nearest_idx = ranked_idx[:max_neighbors]

    def best_from(index_pool):
        best_pair = None
        best_cost = float("inf")
        for launch_idx in index_pool:
            for land_idx in index_pool:
                if launch_idx >= land_idx:
                    continue
                if not _pair_fits_in_drone(drone_route, insertion_index, launch_idx, land_idx):
                    continue

                launch_city = truck_route[launch_idx]
                land_city = truck_route[land_idx]
                trip = D[launch_city][node] + D[node][land_city]
                if trip > flight_limit:
                    continue

                if trip < best_cost:
                    best_cost = trip
                    best_pair = (launch_idx, land_idx)
        return best_pair

    pair = best_from(nearest_idx)
    if pair is not None:
        return pair

    # Fallback: expand search to all internal truck indices.
    return best_from(ranked_idx)


def operator(current_solution):
    _assert_context_is_set()

    original_solution = copy.deepcopy(current_solution)
    new_solution = copy.deepcopy(current_solution)

    truck = new_solution[0]
    drone1 = new_solution[1]
    drone2 = new_solution[2]

    # Choose a valid source route.
    valid_sources = []
    if len(truck) > 2:
        valid_sources.append(0)
    if drone1:
        valid_sources.append(1)
    if drone2:
        valid_sources.append(2)
    if not valid_sources:
        return new_solution

    removal_choice = random.choice(valid_sources)
    insert_choice = random.randint(0, 2)

    removed_value = None
    removed_tuple = None
    removal_index = None

    # Remove from source.
    if removal_choice == 0:
        removal_index = random.randint(1, len(truck) - 2)
        removed_value = truck.pop(removal_index)
    else:
        source_route = new_solution[removal_choice]
        removal_index = random.randint(0, len(source_route) - 1)
        removed_tuple = source_route.pop(removal_index)
        removed_value = removed_tuple[0]

    # Insert into truck (only node id should be inserted).
    if insert_choice == 0:
        if len(truck) < 2:
            return original_solution

        min_idx = 1
        max_idx = len(truck) - 1

        if removal_choice == 0:
            candidate_indices = [i for i in range(min_idx, max_idx + 1) if i != removal_index]
            if not candidate_indices:
                return original_solution
            insertion_index = random.choice(candidate_indices)
        else:
            insertion_index = random.randint(min_idx, max_idx)

        truck.insert(insertion_index, removed_value)
        return new_solution

    # Insert into drone route (1 or 2).
    target_route = new_solution[insert_choice]
    insertion_index = random.randint(0, len(target_route))

    # Truck -> drone: build a fresh pair.
    if removal_choice == 0:
        pair = build_drone_pair(removed_value, truck, target_route, insertion_index)
        if pair is None:
            return original_solution
        target_route.insert(insertion_index, (removed_value, pair[0], pair[1]))
        return new_solution

    # Drone -> drone (or same drone): try keep existing pair if feasible,
    # otherwise find a different pair for same served node.
    preferred_pair = (removed_tuple[1], removed_tuple[2])
    pair = build_drone_pair(removed_value, truck, target_route, insertion_index, preferred_pair=preferred_pair)
    if pair is None:
        return original_solution

    target_route.insert(insertion_index, (removed_value, pair[0], pair[1]))
    return new_solution
