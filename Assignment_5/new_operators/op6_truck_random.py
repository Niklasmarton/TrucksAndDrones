import random
from pathlib import Path
import sys

ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
CORE_DIR = ASSIGNMENT_DIR / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.append(str(CORE_DIR))

from operator_context import assert_context_is_set, get_operator_context, set_operator_context


def set_search_progress(progress):
    return None


def _clone_solution(solution):
    return [solution[0][:], solution[1][:], solution[2][:]]


def _destroy_size(n_customers):
    upper = max(2, round(0.05 * n_customers))
    upper = max(1, int(upper))
    return random.randint(1, upper)


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


def _shift_or_drop_after_truck_removal(route, removed_idx):
    kept = []
    orphan_nodes = []
    for node, launch_idx, land_idx in route:
        if launch_idx == removed_idx or land_idx == removed_idx:
            orphan_nodes.append(node)
            continue
        if launch_idx > removed_idx:
            launch_idx -= 1
        if land_idx > removed_idx:
            land_idx -= 1
        kept.append((node, launch_idx, land_idx))
    kept.sort(key=lambda x: (x[1], x[2], x[0]))
    return kept, orphan_nodes


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


def operator(current_solution):
    assert_context_is_set()
    truck_times, _, _, depot = get_operator_context()

    truck, drone1, drone2 = _clone_solution(current_solution)
    truck_indices = [i for i in range(1, len(truck) - 1) if truck[i] != depot]
    if not truck_indices:
        return current_solution

    destroy_n = min(len(truck_indices), _destroy_size(max(1, len(truck) - 2)))
    picked = sorted(random.sample(truck_indices, destroy_n), reverse=True)

    removed_nodes = []
    for idx in picked:
        removed_node = truck[idx]
        removed_nodes.append(removed_node)
        truck = truck[:idx] + truck[idx + 1 :]
        drone1, orphan1 = _shift_or_drop_after_truck_removal(drone1, idx)
        drone2, orphan2 = _shift_or_drop_after_truck_removal(drone2, idx)
        removed_nodes.extend(orphan1)
        removed_nodes.extend(orphan2)

    # Deduplicate while preserving order.
    unique_nodes = []
    seen = set()
    for node in removed_nodes:
        if node in seen:
            continue
        seen.add(node)
        unique_nodes.append(node)

    # Greedy repair into truck to keep the operator plug-and-play for current ALNS.
    for node in unique_nodes:
        ins_idx = _best_truck_insert(node, truck, truck_times)
        if ins_idx is None:
            return current_solution
        truck = truck[:ins_idx] + [node] + truck[ins_idx:]
        drone1 = _shift_route_after_truck_insert(drone1, ins_idx)
        drone2 = _shift_route_after_truck_insert(drone2, ins_idx)

    candidate = [truck, drone1, drone2]
    if candidate == current_solution:
        return current_solution
    return candidate

