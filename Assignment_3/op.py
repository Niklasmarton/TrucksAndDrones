import random


# Operator context must be set by the algorithm that uses the operator
# (e.g., local_search.py / simulated_annealing.py).
T = None
D = None
flight_limit = None
depot = 0
_op_step = 0
_EXPLORE_PROB = 0.25
_PAIR_EXPLORE_PROB = 0.30


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


def _clone_solution(solution):
    # Tuples inside drone routes are immutable, so shallow-copying route lists is sufficient.
    return [solution[0][:], solution[1][:], solution[2][:]]


# Solution format: [truck, drone1, drone2]
# Each drone route is list[tuple(node, launch_idx, land_idx)]


def _internal_truck_indices(truck_route):
    return list(range(1, len(truck_route) - 1))


def _desired_drone_total(total_customers):
    # Baseline target used across dataset sizes.
    return max(4, total_customers // 6)


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
    neighbor_cap = max(2, min(max_neighbors, len(ranked_idx)))
    nearest_idx = ranked_idx[:neighbor_cap]

    def feasible_pairs(index_pool):
        pairs = []
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
                pairs.append((trip, launch_idx, land_idx))
        return pairs

    def pick_from_pairs(pairs):
        if not pairs:
            return None
        pairs.sort(key=lambda x: x[0])
        if random.random() < _PAIR_EXPLORE_PROB:
            k = min(4, len(pairs))
            _, launch_idx, land_idx = random.choice(pairs[:k])
        else:
            _, launch_idx, land_idx = pairs[0]
        return (launch_idx, land_idx)

    pair = pick_from_pairs(feasible_pairs(nearest_idx))
    if pair is not None:
        return pair

    # Fallback: sample a limited subset of remaining indices instead of full scan.
    remaining_idx = ranked_idx[neighbor_cap:]
    if not remaining_idx:
        return None
    sample_size = min(16, len(remaining_idx))
    sampled_idx = random.sample(remaining_idx, sample_size)
    return pick_from_pairs(feasible_pairs(sampled_idx))


def _select_valid_source(solution, step):
    truck, drone1, drone2 = solution
    valid_sources = []
    if len(truck) > 2:
        valid_sources.append(0)
    if drone1:
        valid_sources.append(1)
    if drone2:
        valid_sources.append(2)
    if not valid_sources:
        return None

    # Mild bias toward truck removals when drones are under-utilized.
    total_customers = (len(truck) - 2) + len(drone1) + len(drone2)
    drone_total = len(drone1) + len(drone2)
    desired_drone_total = _desired_drone_total(total_customers)
    if 0 in valid_sources and drone_total < desired_drone_total and random.random() < 0.55:
        return 0

    if random.random() < _EXPLORE_PROB:
        return random.choice(valid_sources)
    # Deterministic selection with mild cycling:
    # prefer truck moves every third step when possible.
    if step % 3 == 0 and 0 in valid_sources:
        return 0
    # Otherwise choose the non-truck source with most assignments.
    if 1 in valid_sources or 2 in valid_sources:
        d1_len = len(drone1) if 1 in valid_sources else -1
        d2_len = len(drone2) if 2 in valid_sources else -1
        return 1 if d1_len >= d2_len else 2
    return 0


def _select_insert_target(removal_choice, solution, step):
    truck, drone1, drone2 = solution
    total_customers = (len(truck) - 2) + len(drone1) + len(drone2)
    drone_total = len(drone1) + len(drone2)
    desired_drone_total = _desired_drone_total(total_customers)

    if random.random() < _EXPLORE_PROB:
        return random.randint(0, 2)
    if removal_choice == 0:
        # Truck removal: balance truck-route refinement with truck->drone assignment.
        if drone_total < desired_drone_total:
            if random.random() < 0.70:
                return 1 if len(drone1) <= len(drone2) else 2
            return 0
        if random.random() < 0.35:
            return 1 if len(drone1) <= len(drone2) else 2
        return 0

    # Drone removal: if drones are under-utilized, keep assignment on drones more often.
    if drone_total < desired_drone_total:
        if random.random() < 0.60:
            return 2 if removal_choice == 1 else 1
        return 0

    # Otherwise, favor returning to truck moderately.
    if random.random() < 0.70:
        return 0
    return 2 if removal_choice == 1 else 1


def _attempt_operator_move(current_solution, step):
    new_solution = _clone_solution(current_solution)
    truck = new_solution[0]

    removal_choice = _select_valid_source(new_solution, step)
    if removal_choice is None:
        return None
    insert_choice = _select_insert_target(removal_choice, new_solution, step)

    removed_tuple = None
    if removal_choice == 0:
        if random.random() < _EXPLORE_PROB:
            removal_index = random.randint(1, len(truck) - 2)
        else:
            removal_index = 1 + (step % (len(truck) - 2))
        removed_value = truck.pop(removal_index)
    else:
        source_route = new_solution[removal_choice]
        if random.random() < _EXPLORE_PROB:
            removal_index = random.randint(0, len(source_route) - 1)
        else:
            removal_index = step % len(source_route)
        removed_tuple = source_route.pop(removal_index)
        removed_value = removed_tuple[0]

    # Insert into truck.
    if insert_choice == 0:
        min_idx = 1
        max_idx = len(truck) - 1
        if min_idx > max_idx:
            return None

        if removal_choice == 0:
            candidate_indices = [i for i in range(min_idx, max_idx + 1) if i != removal_index]
        else:
            candidate_indices = list(range(min_idx, max_idx + 1))

        if not candidate_indices:
            return None
        if random.random() < _EXPLORE_PROB:
            insertion_index = random.choice(candidate_indices)
        else:
            insertion_index = candidate_indices[step % len(candidate_indices)]
        truck.insert(insertion_index, removed_value)
        return new_solution

    # Insert into drone.
    target_route = new_solution[insert_choice]
    if removal_choice == insert_choice and insert_choice != 0:
        # Avoid exact remove+reinsert at the same position.
        candidate_indices = [i for i in range(0, len(target_route) + 1) if i != removal_index]
        if not candidate_indices:
            return None
        if random.random() < _EXPLORE_PROB:
            insertion_index = random.choice(candidate_indices)
        else:
            insertion_index = candidate_indices[step % len(candidate_indices)]
    else:
        if random.random() < _EXPLORE_PROB:
            insertion_index = random.randint(0, len(target_route))
        else:
            insertion_index = step % (len(target_route) + 1)

    if removal_choice == 0:
        pair = build_drone_pair(
            removed_value,
            truck,
            target_route,
            insertion_index,
            max_neighbors=6,
        )
    else:
        preferred_pair = (removed_tuple[1], removed_tuple[2])
        pair = build_drone_pair(
            removed_value,
            truck,
            target_route,
            insertion_index,
            preferred_pair=preferred_pair,
            max_neighbors=6,
        )

    if pair is None:
        return None

    target_route.insert(insertion_index, (removed_value, pair[0], pair[1]))
    return new_solution


def operator(current_solution):
    global _op_step
    _assert_context_is_set()
    max_attempts = 4
    for _ in range(max_attempts):
        candidate = _attempt_operator_move(current_solution, _op_step)
        _op_step += 1
        if candidate is None:
            continue
        if candidate != current_solution:
            return candidate
    return current_solution
