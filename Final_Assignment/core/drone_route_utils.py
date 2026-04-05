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
                # Drone arrives at land node at: truck_arrival_at_launch + drone_trip_time
                # (if drone is faster than truck segment, truck arrival at land determines it,
                # but we still prefer an earlier absolute launch to minimise makespan).
                arrival_at_land = truck_prefix[launch_idx] + trip
                total_route_time = truck_prefix[-1] if truck_prefix else 1.0
                arrival_penalty = arrival_at_land / max(1.0, total_route_time)
                # Prefer pairs that synchronize well AND launch early for earlier arrival.
                score = 3.0 * wait_excess + 0.6 * mismatch + 0.02 * trip + 0.3 * arrival_penalty
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


# ---------------------------------------------------------------------------
# Helpers shared by insert_node_with_truck_fallback and
# enforce_wait_feasible_with_truck_fallback
# ---------------------------------------------------------------------------

def _shift_route_indices_after_insert(route, insert_idx):
    shifted = []
    for node, launch_idx, land_idx in route:
        if launch_idx >= insert_idx:
            launch_idx += 1
        if land_idx >= insert_idx:
            land_idx += 1
        shifted.append((node, launch_idx, land_idx))
    shifted.sort(key=lambda x: (x[1], x[2], x[0]))
    return shifted


def _best_truck_insert_idx(node, truck, T):
    best_delta = None
    best_idx = None
    for ins_idx in range(1, len(truck)):
        a, b = truck[ins_idx - 1], truck[ins_idx]
        delta = T[a][node] + T[node][b] - T[a][b]
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_idx = ins_idx
    return best_idx


def _endpoints_available(route, launch_idx, land_idx):
    for _, l, r in route:
        if l == launch_idx or r == land_idx:
            return False
    return True


def _find_drone_route_conflict(route):
    """Return the customer node of the first conflicting trip, or None."""
    used_launch = set()
    used_land = set()
    prev_land = 0
    for cust, launch_idx, land_idx in sorted(route, key=lambda x: (x[1], x[2], x[0])):
        if (
            launch_idx >= land_idx
            or launch_idx < prev_land
            or launch_idx in used_launch
            or land_idx in used_land
        ):
            return cust
        used_launch.add(launch_idx)
        used_land.add(land_idx)
        prev_land = land_idx
    return None


# ---------------------------------------------------------------------------
# Public helpers used by operators
# ---------------------------------------------------------------------------

def insert_node_with_truck_fallback(
    node,
    truck,
    primary_drone,
    secondary_drone,
    preferred_pair=None,
    max_neighbors=5,
    pair_explore_prob=0.0,
):
    """
    Insert *node* preferentially into primary_drone, then secondary_drone,
    then the truck as a last resort.

    Returns (truck, primary_drone, secondary_drone, inserted_as_drone)
    where inserted_as_drone is True when the node went into a drone route.
    Returns None on hard failure (e.g. truck is malformed).
    """
    T, _, _, _ = get_operator_context()

    for drone_route in (primary_drone, secondary_drone):
        for ins_idx in range(len(drone_route) + 1):
            pair = build_drone_pair(
                node, truck, drone_route, ins_idx,
                preferred_pair=preferred_pair,
                max_neighbors=max_neighbors,
                pair_explore_prob=pair_explore_prob,
            )
            if pair is None:
                continue
            launch_idx, land_idx = pair
            if not _endpoints_available(drone_route, launch_idx, land_idx):
                continue
            new_route = drone_route[:]
            new_route.insert(ins_idx, (node, launch_idx, land_idx))
            new_route.sort(key=lambda x: (x[1], x[2], x[0]))
            if drone_route is primary_drone:
                return truck[:], new_route, secondary_drone[:], True
            else:
                return truck[:], primary_drone[:], new_route, True

    # Fall back: insert into truck
    if node in truck:
        return truck[:], primary_drone[:], secondary_drone[:], False

    ins_idx = _best_truck_insert_idx(node, truck, T)
    if ins_idx is None:
        return None

    new_truck = truck[:ins_idx] + [node] + truck[ins_idx:]
    new_primary = _shift_route_indices_after_insert(primary_drone, ins_idx)
    new_secondary = _shift_route_indices_after_insert(secondary_drone, ins_idx)
    return new_truck, new_primary, new_secondary, False


def enforce_wait_feasible_with_truck_fallback(truck, drone1, drone2, max_iterations=100):
    """
    Repair any overlapping, out-of-order, or duplicate-endpoint trips in
    drone1 / drone2 by moving conflicting customers to the truck.

    Returns (truck, drone1, drone2, ok).  ok=False if unresolvable within
    max_iterations.
    """
    T, _, _, _ = get_operator_context()
    truck = truck[:]
    drone1 = drone1[:]
    drone2 = drone2[:]

    for _ in range(max_iterations):
        conflict1 = _find_drone_route_conflict(drone1)
        conflict2 = _find_drone_route_conflict(drone2)

        if conflict1 is None and conflict2 is None:
            return truck, drone1, drone2, True

        node = conflict1 if conflict1 is not None else conflict2
        if conflict1 is not None:
            drone1 = [t for t in drone1 if t[0] != node]
        else:
            drone2 = [t for t in drone2 if t[0] != node]

        ins_idx = _best_truck_insert_idx(node, truck, T)
        if ins_idx is None:
            return truck, drone1, drone2, False

        truck = truck[:ins_idx] + [node] + truck[ins_idx:]
        drone1 = _shift_route_indices_after_insert(drone1, ins_idx)
        drone2 = _shift_route_indices_after_insert(drone2, ins_idx)

    return truck, drone1, drone2, False
