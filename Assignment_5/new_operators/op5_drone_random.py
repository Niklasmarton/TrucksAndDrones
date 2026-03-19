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
    truck_times, _, _, _ = get_operator_context()

    truck, drone1, drone2 = _clone_solution(current_solution)
    drone_served = [(1, i, trip[0]) for i, trip in enumerate(drone1)] + [
        (2, i, trip[0]) for i, trip in enumerate(drone2)
    ]
    if not drone_served:
        return current_solution

    destroy_n = min(len(drone_served), _destroy_size(max(1, len(truck) - 2)))
    picked = random.sample(drone_served, destroy_n)

    remove_idx_d1 = sorted([i for rid, i, _ in picked if rid == 1], reverse=True)
    remove_idx_d2 = sorted([i for rid, i, _ in picked if rid == 2], reverse=True)
    removed_nodes = [node for _, _, node in picked]

    for i in remove_idx_d1:
        if 0 <= i < len(drone1):
            drone1.pop(i)
    for i in remove_idx_d2:
        if 0 <= i < len(drone2):
            drone2.pop(i)

    # Greedy repair into truck so the operator remains usable in current ALNS flow.
    for node in removed_nodes:
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

