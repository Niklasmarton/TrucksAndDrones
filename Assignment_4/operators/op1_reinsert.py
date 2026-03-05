import random
from pathlib import Path
import sys

CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.append(str(CORE_DIR))

from drone_route_utils import build_drone_pair, pair_fits_in_drone
from operator_context import assert_context_is_set, set_operator_context

# Compatibility alias for older imports.
_pair_fits_in_drone = pair_fits_in_drone

_op_step = 0
_EXPLORE_PROB = 0.12
_PAIR_EXPLORE_PROB = 0.12


def reset_operator_state():
    global _op_step
    _op_step = 0


def _clone_solution(solution):
    # Tuples inside drone routes are immutable, so shallow-copying route lists is sufficient.
    return [solution[0][:], solution[1][:], solution[2][:]]


def _desired_drone_total(total_customers):
    return max(4, total_customers // 6)


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

    total_customers = (len(truck) - 2) + len(drone1) + len(drone2)
    drone_total = len(drone1) + len(drone2)
    desired_drone_total = _desired_drone_total(total_customers)

    if 0 in valid_sources and drone_total < desired_drone_total and random.random() < 0.45:
        return 0

    if random.random() < _EXPLORE_PROB:
        return random.choice(valid_sources)

    if step % 3 == 0 and 0 in valid_sources:
        return 0

    if 1 in valid_sources or 2 in valid_sources:
        d1_len = len(drone1) if 1 in valid_sources else -1
        d2_len = len(drone2) if 2 in valid_sources else -1
        return 1 if d1_len >= d2_len else 2

    return 0


def _select_insert_target(removal_choice, solution):
    truck, drone1, drone2 = solution
    total_customers = (len(truck) - 2) + len(drone1) + len(drone2)
    drone_total = len(drone1) + len(drone2)
    desired_drone_total = _desired_drone_total(total_customers)

    if random.random() < _EXPLORE_PROB:
        return random.randint(0, 2)

    if removal_choice == 0:
        if drone_total < desired_drone_total:
            if random.random() < 0.65:
                return 1 if len(drone1) <= len(drone2) else 2
            return 0
        if random.random() < 0.35:
            return 1 if len(drone1) <= len(drone2) else 2
        return 0

    if drone_total < desired_drone_total:
        if random.random() < 0.55:
            return 2 if removal_choice == 1 else 1
        return 0

    if random.random() < 0.70:
        return 0
    return 2 if removal_choice == 1 else 1


def _attempt_operator_move(current_solution, step):
    new_solution = _clone_solution(current_solution)
    truck = new_solution[0]

    removal_choice = _select_valid_source(new_solution, step)
    if removal_choice is None:
        return None

    insert_choice = _select_insert_target(removal_choice, new_solution)
    removed_tuple = None

    if removal_choice == 0:
        if len(truck) <= 2:
            return None
        if random.random() < _EXPLORE_PROB:
            removal_index = random.randint(1, len(truck) - 2)
        else:
            removal_index = 1 + (step % (len(truck) - 2))
        removed_value = truck.pop(removal_index)
    else:
        source_route = new_solution[removal_choice]
        if not source_route:
            return None
        if random.random() < _EXPLORE_PROB:
            removal_index = random.randint(0, len(source_route) - 1)
        else:
            removal_index = step % len(source_route)
        removed_tuple = source_route.pop(removal_index)
        removed_value = removed_tuple[0]

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

    target_route = new_solution[insert_choice]
    if removal_choice == insert_choice and insert_choice != 0:
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
            pair_explore_prob=_PAIR_EXPLORE_PROB,
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
            pair_explore_prob=_PAIR_EXPLORE_PROB,
        )

    if pair is None:
        return None

    target_route.insert(insertion_index, (removed_value, pair[0], pair[1]))
    return new_solution


def operator(current_solution):
    global _op_step
    assert_context_is_set()
    max_attempts = 4

    for _ in range(max_attempts):
        candidate = _attempt_operator_move(current_solution, _op_step)
        _op_step += 1
        if candidate is None:
            continue
        if candidate != current_solution:
            return candidate

    return current_solution
