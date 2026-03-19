"""
This operator is a reinsertion operator

It is based on the one used in the previous assignment - it takes a node from either truck or drone and reinserts it back into either truck or drone
The idea was having an operator that makes smaller, but hopefully, good moves, simply moving a node between truck and drones

For the truck, the thought was to fin the most costly nodes and choose randomly between top k nodes but with a rank bias (meaning that be worse nodes will be chosen with a higher likelihood)
As for the drone, I found that choosing random routes for removal produced better solutions
For insertion from drone to truck, I calculate the truck delta (decreased delivery time by insertion) and insert greedy
For drone insertion, the operator first builds a set of candidate insertion positions in the target drone route, then evaluates feasible launch/landing pairs for each candidate position. The feasible options are then scored based on the travel time of the truck compared to the travel time of the drone. Then, the insertion chooses the best option greedily. 

The operator is also biased towards building the truck route early in the run to build a good basis. From the middle of the run it chooses truck<-> drone about 50/50 and later on it focuses on refining drone routes. 

I also have explore prob, which gives the operator a diversifying effect where it does not choose the insertion greedily, but randomly. Through testing, this haad good effects on the solutions. 
"""
import random
from pathlib import Path
import sys

CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.append(str(CORE_DIR))

from drone_route_utils import build_drone_pair, drone_route_is_feasible
from operator_context import assert_context_is_set, get_operator_context, set_operator_context


                      
_EXPLORE_PROB = 0.12
_PAIR_EXPLORE_PROB = 0.10
_TRUCK_TO_DRONE_BIAS = 0.0
_SEARCH_PROGRESS = 0.5
_SYNC_PEN_WEIGHT = 0.15

              
_MAX_ATTEMPTS = 5
_TOP_K_TRUCK_REMOVALS = 4
_TOP_K_TRUCK_INSERTIONS = 5
_MAX_DRONE_INSERTION_TRIALS = 6


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


def set_search_progress(progress):
    global _SEARCH_PROGRESS
    _SEARCH_PROGRESS = _clamp01(progress)


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

                                                    
    remaining = [idx for idx in indices if idx not in best]
    random.shuffle(remaining)
    return best + remaining


def _choose_truck_removal_index(truck):
    candidates = _candidate_truck_removal_indices(truck)
    if not candidates:
        return None

    if random.random() < _EXPLORE_PROB:
        return random.choice(candidates[: min(len(candidates), 8)])

                                                                                  
                                                                                   
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

    if _SEARCH_PROGRESS < 0.35:
                                     
        truck_pick_prob = 0.22 + 0.18 * _TRUCK_TO_DRONE_BIAS
    elif _SEARCH_PROGRESS < 0.75:
                                     
        truck_pick_prob = 0.34 + 0.36 * _TRUCK_TO_DRONE_BIAS
    else:
                                               
        truck_pick_prob = 0.28 + 0.22 * _TRUCK_TO_DRONE_BIAS

    if 0 in valid_sources and random.random() < truck_pick_prob:
        return 0

    if random.random() < _EXPLORE_PROB:
        return random.choice(valid_sources)

                                                                           
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
                                            
                                                 
        if _SEARCH_PROGRESS < 0.35:
            truck_to_drone_prob = 0.18 + 0.22 * _TRUCK_TO_DRONE_BIAS
        elif _SEARCH_PROGRESS < 0.75:
            truck_to_drone_prob = 0.34 + 0.40 * _TRUCK_TO_DRONE_BIAS
        else:
            truck_to_drone_prob = 0.24 + 0.26 * _TRUCK_TO_DRONE_BIAS
        if random.random() < truck_to_drone_prob:
            return 1 if len(drone1) <= len(drone2) else 2
        return 0

                                                                 
    if _SEARCH_PROGRESS < 0.35:
        truck_insert_prob = 0.85
    elif _SEARCH_PROGRESS < 0.75:
        truck_insert_prob = 0.70
    else:
        truck_insert_prob = 0.78
    if random.random() < truck_insert_prob:
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
        scored.append((-delta, idx))                           

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

    return best                                          


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

                                   
    target_route = new_solution[insert_choice]

    preferred_pair = None
    preferred_index = None
    if removed_tuple is not None:
        preferred_pair = (removed_tuple[1], removed_tuple[2])
                                                                                    
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


def _prefix_truck_times(truck_route, truck_times):
    prefix = [0.0]
    for i in range(len(truck_route) - 1):
        prefix.append(prefix[-1] + truck_times[truck_route[i]][truck_route[i + 1]])
    return prefix


def _trip_penalty(trip, truck_route, prefix, drone_times):
    node, launch_idx, land_idx = trip
    if (
        launch_idx < 0
        or land_idx < 0
        or launch_idx >= len(truck_route)
        or land_idx >= len(truck_route)
        or launch_idx >= land_idx
    ):
                                                                   
        return 1e9
    launch_node = truck_route[launch_idx]
    land_node = truck_route[land_idx]
    sortie_time = drone_times[launch_node][node] + drone_times[node][land_node]
    truck_time = prefix[land_idx] - prefix[launch_idx]
    wait_excess = max(0.0, sortie_time - truck_time)
    mismatch = abs(sortie_time - truck_time)
    return 3.0 * wait_excess + 0.6 * mismatch + 0.02 * sortie_time


def _solution_sync_penalty(solution, truck_times, drone_times):
    truck, drone1, drone2 = solution
    prefix = _prefix_truck_times(truck, truck_times)
    pen = 0.0
    for trip in drone1:
        pen += _trip_penalty(trip, truck, prefix, drone_times)
    for trip in drone2:
        pen += _trip_penalty(trip, truck, prefix, drone_times)
    pen += 2.0 * abs(len(drone1) - len(drone2))
    return pen


def operator(current_solution):
    assert_context_is_set()
    truck_times, drone_times, _, _ = get_operator_context()

    old_truck_cost = _truck_total_cost(current_solution[0], truck_times)
    old_sync_penalty = _solution_sync_penalty(current_solution, truck_times, drone_times)

    candidates = []
    for _ in range(_MAX_ATTEMPTS):
        candidate = _attempt_operator_move(current_solution)
        if candidate is None or candidate == current_solution:
            continue
        truck_delta = _truck_total_cost(candidate[0], truck_times) - old_truck_cost
        sync_delta = _solution_sync_penalty(candidate, truck_times, drone_times) - old_sync_penalty
        score = truck_delta + _SYNC_PEN_WEIGHT * sync_delta
        candidates.append((score, candidate))

    if not candidates:
        return current_solution

    candidates.sort(key=lambda x: x[0])
    if _SEARCH_PROGRESS < 0.35:
        explore_pick_prob = 0.14
    elif _SEARCH_PROGRESS < 0.75:
        explore_pick_prob = 0.20
    else:
        explore_pick_prob = 0.08
    if random.random() < explore_pick_prob:
        return random.choice(candidates[: min(3, len(candidates))])[1]
    return candidates[0][1]
