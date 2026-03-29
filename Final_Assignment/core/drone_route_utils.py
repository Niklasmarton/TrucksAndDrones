import random

from operator_context import get_operator_context


def internal_truck_indices(truck_route):
    return list(range(1, len(truck_route) - 1))


def pair_fits_in_drone(drone_route, insertion_index, launch_idx, land_idx):
    if launch_idx >= land_idx:
        return False

    prev_land = 0 if insertion_index == 0 else drone_route[insertion_index - 1][2]
    next_launch = float("inf") if insertion_index == len(drone_route) else drone_route[insertion_index][1]
    return launch_idx >= prev_land and land_idx <= next_launch


def build_drone_pair(
    node,
    truck_route,
    drone_route,
    insertion_index,
    preferred_pair=None,
    max_neighbors=5,
    pair_explore_prob=0.30,
):
    """
    Returns (launch_idx, land_idx) for node at given insertion position in a drone route.
    Enforces increasing index order and timeline feasibility at insertion point.
    """
    T, D, flight_limit, _ = get_operator_context()
    internal_idx = internal_truck_indices(truck_route)
    if len(internal_idx) < 2:
        return None

    if preferred_pair is not None:
        launch_idx, land_idx = preferred_pair
        if 0 <= launch_idx < len(truck_route) and 0 <= land_idx < len(truck_route):
            launch_city = truck_route[launch_idx]
            land_city = truck_route[land_idx]
            trip = D[launch_city][node] + D[node][land_city]
            if trip <= flight_limit and pair_fits_in_drone(drone_route, insertion_index, launch_idx, land_idx):
                return launch_idx, land_idx

    ranked_idx = sorted(internal_idx, key=lambda idx: T[node][truck_route[idx]])
    neighbor_cap = max(2, min(max_neighbors, len(ranked_idx)))
    nearest_idx = ranked_idx[:neighbor_cap]

    # Prefix truck times for O(1) truck segment duration lookups.
    truck_prefix = [0.0]
    for i in range(len(truck_route) - 1):
        truck_prefix.append(truck_prefix[-1] + T[truck_route[i]][truck_route[i + 1]])

    def feasible_pairs(index_pool):
        pairs = []
        for launch_idx in index_pool:
            for land_idx in index_pool:
                if launch_idx >= land_idx:
                    continue
                if not pair_fits_in_drone(drone_route, insertion_index, launch_idx, land_idx):
                    continue
                launch_city = truck_route[launch_idx]
                land_city = truck_route[land_idx]
                trip = D[launch_city][node] + D[node][land_city]
                if trip > flight_limit:
                    continue
                truck_time = truck_prefix[land_idx] - truck_prefix[launch_idx]
                wait_excess = max(0.0, trip - truck_time)
                mismatch = abs(trip - truck_time)
                # Prefer pairs that synchronize drone sortie with truck travel.
                score = 3.0 * wait_excess + 0.6 * mismatch + 0.02 * trip
                pairs.append((score, trip, launch_idx, land_idx))
        return pairs

    def pick_from_pairs(pairs):
        if not pairs:
            return None
        pairs.sort(key=lambda x: (x[0], x[1]))
        if random.random() < pair_explore_prob:
            k = min(4, len(pairs))
            _, _, launch_idx, land_idx = random.choice(pairs[:k])
        else:
            _, _, launch_idx, land_idx = pairs[0]
        return launch_idx, land_idx

    pair = pick_from_pairs(feasible_pairs(nearest_idx))
    if pair is not None:
        return pair

    remaining_idx = ranked_idx[neighbor_cap:]
    if not remaining_idx:
        return None
    sample_size = min(16, len(remaining_idx))
    sampled_idx = random.sample(remaining_idx, sample_size)
    return pick_from_pairs(feasible_pairs(sampled_idx))


def map_index_after_pop_insert(idx, pop_idx, insert_idx):
    if idx > pop_idx:
        idx -= 1
    if idx >= insert_idx:
        idx += 1
    return idx


def update_drone_route_indices(drone_route, pop_idx, insert_idx, removed_old_idx):
    """
    Remap trip indices so each trip still points to the same endpoint nodes
    after truck pop+insert.
    """
    updated = []
    for cust, launch_idx, land_idx in drone_route:
        new_launch = insert_idx if launch_idx == removed_old_idx else map_index_after_pop_insert(
            launch_idx, pop_idx, insert_idx
        )
        new_land = insert_idx if land_idx == removed_old_idx else map_index_after_pop_insert(
            land_idx, pop_idx, insert_idx
        )
        updated.append((cust, new_launch, new_land))

    updated.sort(key=lambda x: (x[1], x[2], x[0]))
    return updated


def remap_drone_route_by_endpoint_nodes(old_truck, new_truck, drone_route):
    """
    Keep each drone trip tied to the same endpoint nodes after a truck route move
    that reorders many positions (e.g., 2-opt).
    """
    new_index_of = {}
    for idx, node in enumerate(new_truck):
        if node not in new_index_of:
            new_index_of[node] = idx

    updated = []
    for cust, launch_idx, land_idx in drone_route:
        if not (0 <= launch_idx < len(old_truck) and 0 <= land_idx < len(old_truck)):
            return None

        launch_node = old_truck[launch_idx]
        land_node = old_truck[land_idx]
        new_launch = new_index_of.get(launch_node)
        new_land = new_index_of.get(land_node)
        if new_launch is None or new_land is None:
            return None
        updated.append((cust, new_launch, new_land))

    updated.sort(key=lambda x: (x[1], x[2], x[0]))
    return updated


def drone_route_is_feasible(drone_route):
    """
    Feasibility at drone-route timeline level:
    - launch < land
    - no overlap inside a single drone route
    """
    trips = sorted(drone_route, key=lambda x: (x[1], x[2], x[0]))
    prev_land = None
    for _, launch_idx, land_idx in trips:
        if launch_idx >= land_idx:
            return False
        if prev_land is not None and launch_idx < prev_land:
            return False
        prev_land = land_idx
    return True


def repair_drone_route(truck_route, drone_route, max_repairs=5, max_neighbors=6):
    """
    Repairs only trips that violate launch/land order or overlap.
    """
    route = sorted(drone_route, key=lambda x: (x[1], x[2], x[0]))
    repairs = 0
    i = 0

    while i < len(route):
        cust, launch_idx, land_idx = route[i]
        prev_land = 0 if i == 0 else route[i - 1][2]
        is_ok = (launch_idx < land_idx) and (launch_idx >= prev_land)

        if is_ok:
            i += 1
            continue

        repairs += 1
        if repairs > max_repairs:
            return None

        old_trip = route.pop(i)
        insertion_index = i
        preferred_pair = (old_trip[1], old_trip[2])

        pair = build_drone_pair(
            cust,
            truck_route,
            route,
            insertion_index,
            preferred_pair=preferred_pair,
            max_neighbors=max_neighbors,
        )
        if pair is None:
            return None

        new_launch, new_land = pair
        if not pair_fits_in_drone(route, insertion_index, new_launch, new_land):
            return None

        route.insert(insertion_index, (cust, new_launch, new_land))
        route.sort(key=lambda x: (x[1], x[2], x[0]))
        i = max(0, i - 1)

    return route
