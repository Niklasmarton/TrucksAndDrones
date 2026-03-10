import random
from pathlib import Path
import sys

CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.append(str(CORE_DIR))

from drone_route_utils import build_drone_pair, drone_route_is_feasible
from operator_context import assert_context_is_set, get_operator_context, set_operator_context


# Exploration settings
_EXPLORE_PROB = 0.12
_PAIR_EXPLORE_PROB = 0.12
_TRUCK_TO_DRONE_BIAS = 0.0

# New tuning knobs
_MAX_ATTEMPTS = 5
_TOP_K_TRUCK_REMOVALS = 4
_TOP_K_TRUCK_INSERTIONS = 5
_MAX_DRONE_INSERTION_TRIALS = 4


def reset_operator_state():
    pass


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def set_truck_to_drone_bias(bias):
    """
    Configure how strongly op1 should favor truck->drone reinsert moves.
    0.0 keeps baseline behavior, 1.0 applies the strongest bias.
    """
    global _TRUCK_TO_DRONE_BIAS
    _TRUCK_TO_DRONE_BIAS = _clamp01(bias)


def _clone_solution(solution):
    return [solution[0][:], solution[1][:], solution[2][:]]



def _truck_removal_gain(truck, idx, truck_times):
    """
    Gain from removing truck[idx] from the truck route.
    """
    if idx <= 0 or idx >= len(truck) - 1:
        return 0.0

    prev_node = truck[idx - 1]
    node = truck[idx]
    next_node = truck[idx + 1]

    return (
        truck_times[prev_node][node]
        + truck_times[node][next_node]
        - truck_times[prev_node][next_node]
    )


def _truck_insertion_delta(truck, insertion_index, node, truck_times):
    """
    Delta from inserting node at truck[insertion_index].
    insertion_index is the position where node would be inserted.
    """
    if insertion_index <= 0 or insertion_index >= len(truck):
        return float("inf")

    left = truck[insertion_index - 1]
    right = truck[insertion_index]

    return (
        truck_times[left][node]
        + truck_times[node][right]
        - truck_times[left][right]
    )


def _sample_top_indices(scored_items, k):
    """
    Returns up to k best indices from a list of (score, value), sorted descending.
    """
    scored_items.sort(key=lambda x: x[0], reverse=True)
    return [value for _, value in scored_items[:k]]


def _candidate_truck_removal_indices(truck):
    """
    Prefer truck nodes whose removal saves a lot of truck time.
    """
    truck_times, _, _, _ = get_operator_context()
    indices = list(range(1, len(truck) - 1))
    if not indices:
        return []

    scored = [(_truck_removal_gain(truck, idx, truck_times), idx) for idx in indices]
    best = _sample_top_indices(scored, _TOP_K_TRUCK_REMOVALS)

    # Keep some diversity by mixing with all indices
    remaining = [idx for idx in indices if idx not in best]
    random.shuffle(remaining)
    return best + remaining


def _choose_truck_removal_index(truck):
    candidates = _candidate_truck_removal_indices(truck)
    if not candidates:
        return None

    if random.random() < _EXPLORE_PROB:
        return random.choice(candidates[: min(len(candidates), 8)])

    # Rank-weighted random over top-k: rank 1 gets weight 1, rank 2 gets 1/2, etc.
    # Strongly biases toward the best node while still visiting lower-ranked nodes.
    k = min(_TOP_K_TRUCK_REMOVALS, len(candidates))
    weights = [1.0 / (i + 1) for i in range(k)]
    total = sum(weights)
    r = random.random() * total
    for i, w in enumerate(weights):
        r -= w
        if r <= 0:
            return candidates[i]
    return candidates[0]


def _choose_drone_removal_index(route):
    if not route:
        return None

    return random.randint(0, len(route) - 1)


def _select_valid_source(solution):
    """
    Soft source choice:
    - slight preference for truck when truck->drone bias is high
    - otherwise keep all sources possible
    - no fixed target drone count
    """
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

    if len(valid_sources) == 1:
        return valid_sources[0]

    truck_pick_prob = 0.34 + 0.36 * _TRUCK_TO_DRONE_BIAS

    if 0 in valid_sources and random.random() < truck_pick_prob:
        return 0

    if random.random() < _EXPLORE_PROB:
        return random.choice(valid_sources)

    # Mild preference to take from the longer drone route, otherwise truck.
    if 1 in valid_sources or 2 in valid_sources:
        d1_len = len(drone1) if 1 in valid_sources else -1
        d2_len = len(drone2) if 2 in valid_sources else -1

        if random.random() < 0.33 and 0 in valid_sources:
            return 0

        return 1 if d1_len >= d2_len else 2

    return 0


def _select_insert_target(removal_choice, solution):
    """
    Soft target choice:
    - when removing from truck, use a bias toward drones
    - otherwise prefer truck more often, but still allow drone-to-drone moves
    """
    truck, drone1, drone2 = solution

    if random.random() < _EXPLORE_PROB:
        choices = [0, 1, 2]
        if removal_choice != 0 and not (drone1 or drone2):
            return 0
        return random.choice(choices)

    if removal_choice == 0:
        # Truck -> drone bias only, no desired drone count target.
        truck_to_drone_prob = 0.30 + 0.45 * _TRUCK_TO_DRONE_BIAS
        if random.random() < truck_to_drone_prob:
            return 1 if len(drone1) <= len(drone2) else 2
        return 0

    # Drone source: usually try truck, sometimes swap drone route
    if random.random() < 0.70:
        return 0
    return 2 if removal_choice == 1 else 1


def _candidate_truck_insertion_indices(truck, removed_value, exclude_index=None):
    """
    Build promising truck insertion positions, ranked by insertion delta.
    """
    truck_times, _, _, _ = get_operator_context()
    min_idx = 1
    max_idx = len(truck) - 1

    if min_idx > max_idx:
        return []

    candidate_indices = list(range(min_idx, max_idx + 1))
    if exclude_index is not None:
        candidate_indices = [i for i in candidate_indices if i != exclude_index]

    if not candidate_indices:
        return []

    scored = []
    for idx in candidate_indices:
        delta = _truck_insertion_delta(truck, idx, removed_value, truck_times)
        scored.append((-delta, idx))  # smaller delta is better

    best = _sample_top_indices(scored, _TOP_K_TRUCK_INSERTIONS)

    remaining = [idx for idx in candidate_indices if idx not in best]
    random.shuffle(remaining)
    return best + remaining


def _choose_truck_insertion_index(truck, removed_value, exclude_index=None):
    candidates = _candidate_truck_insertion_indices(
        truck,
        removed_value,
        exclude_index=exclude_index,
    )
    if not candidates:
        return None

    if random.random() < _EXPLORE_PROB:
        return random.choice(candidates[: min(len(candidates), 8)])

    return candidates[0]


def _drone_pair_score(node, truck, pair):
    """
    Higher is better.
    Prefer drone pairs where truck has enough time between launch and landing.
    If matrices unavailable, return 0.
    """
    truck_times, drone_times, _, _ = get_operator_context()

    if pair is None:
        return 0.0

    launch_idx, land_idx = pair

    if not (0 <= launch_idx < len(truck) and 0 <= land_idx < len(truck)):
        return float("-inf")
    if launch_idx >= land_idx:
        return float("-inf")

    launch_node = truck[launch_idx]
    land_node = truck[land_idx]

    truck_segment_time = 0.0
    for i in range(launch_idx, land_idx):
        truck_segment_time += truck_times[truck[i]][truck[i + 1]]

    drone_flight_time = drone_times[launch_node][node] + drone_times[node][land_node]

    # Positive slack is good, negative means likely waiting.
    slack = truck_segment_time - drone_flight_time
    return slack


def _candidate_drone_insertion_indices(target_route, preferred_index=None):
    indices = list(range(0, len(target_route) + 1))
    if preferred_index is None or preferred_index not in indices:
        random.shuffle(indices)
        return indices

    remaining = [i for i in indices if i != preferred_index]
    random.shuffle(remaining)
    return [preferred_index] + remaining


def _choose_best_drone_insertion(
    removed_value,
    truck,
    target_route,
    preferred_pair=None,
    preferred_index=None,
):
    """
    Try a few insertion positions into the drone route.
    Keep the feasible one with the best pair score.
    """
    candidate_indices = _candidate_drone_insertion_indices(
        target_route,
        preferred_index=preferred_index,
    )

    best = None
    trials = 0

    for insertion_index in candidate_indices:
        pair = build_drone_pair(
            removed_value,
            truck,
            target_route,
            insertion_index,
            preferred_pair=preferred_pair,
            max_neighbors=20,
            pair_explore_prob=_PAIR_EXPLORE_PROB,
        )

        if pair is None:
            continue

        score = _drone_pair_score(removed_value, truck, pair)

        candidate = (score, insertion_index, pair)
        if best is None or candidate[0] > best[0]:
            best = candidate

        trials += 1
        if trials >= _MAX_DRONE_INSERTION_TRIALS:
            break

    return best  # (score, insertion_index, pair) or None


def _attempt_operator_move(current_solution):
    new_solution = _clone_solution(current_solution)
    truck = new_solution[0]

    removal_choice = _select_valid_source(new_solution)
    if removal_choice is None:
        return None

    insert_choice = _select_insert_target(removal_choice, new_solution)

    removed_tuple = None
    removed_value = None
    original_truck_removal_index = None

    # ----- Remove -----
    if removal_choice == 0:
        if len(truck) <= 2:
            return None

        removal_index = _choose_truck_removal_index(truck)
        if removal_index is None:
            return None

        original_truck_removal_index = removal_index
        removed_value = truck.pop(removal_index)

    else:
        source_route = new_solution[removal_choice]
        if not source_route:
            return None

        removal_index = _choose_drone_removal_index(source_route)
        if removal_index is None:
            return None

        removed_tuple = source_route.pop(removal_index)
        removed_value = removed_tuple[0]

    # ----- Insert into truck -----
    if insert_choice == 0:
        exclude_index = original_truck_removal_index if removal_choice == 0 else None
        insertion_index = _choose_truck_insertion_index(
            truck,
            removed_value,
            exclude_index=exclude_index,
        )
        if insertion_index is None:
            return None

        truck.insert(insertion_index, removed_value)
        return new_solution

    # ----- Insert into drone -----
    target_route = new_solution[insert_choice]

    preferred_pair = None
    preferred_index = None
    if removed_tuple is not None:
        preferred_pair = (removed_tuple[1], removed_tuple[2])
        # Try roughly same relative position first if reinserting within drone world
        if insert_choice == removal_choice:
            preferred_index = removal_index

    best_drone_insert = _choose_best_drone_insertion(
        removed_value,
        truck,
        target_route,
        preferred_pair=preferred_pair,
        preferred_index=preferred_index,
    )

    if best_drone_insert is None:
        return None

    _, insertion_index, pair = best_drone_insert
    target_route.insert(insertion_index, (removed_value, pair[0], pair[1]))
    if not drone_route_is_feasible(target_route):
        return None
    return new_solution


def _truck_total_cost(truck, truck_times):
    cost = 0.0
    for i in range(len(truck) - 1):
        cost += truck_times[truck[i]][truck[i + 1]]
    return cost


def operator(current_solution):
    assert_context_is_set()
    truck_times, _, _, _ = get_operator_context()

    old_truck_cost = _truck_total_cost(current_solution[0], truck_times)

    candidates = []
    for _ in range(_MAX_ATTEMPTS):
        candidate = _attempt_operator_move(current_solution)
        if candidate is None or candidate == current_solution:
            continue
        score = _truck_total_cost(candidate[0], truck_times) - old_truck_cost
        candidates.append((score, candidate))

    if not candidates:
        return current_solution

    candidates.sort(key=lambda x: x[0])
    if random.random() < 0.25:
        return random.choice(candidates[: min(3, len(candidates))])[1]
    return candidates[0][1]