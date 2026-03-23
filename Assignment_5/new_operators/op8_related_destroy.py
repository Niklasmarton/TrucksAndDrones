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
    upper = max(3, round(0.05 * n_customers))
    upper = max(2, int(upper))
    return random.randint(2, upper)


def _rank_biased_pick(candidates, top_k):
    if not candidates:
        return None
    k = min(len(candidates), top_k)
    if k <= 1:
        return candidates[0]
    weights = [k - i for i in range(k)]
    return random.choices(candidates[:k], weights=weights, k=1)[0]


def _pick_related_indices(truck, truck_times, depot, count):
    customer_positions = [i for i in range(1, len(truck) - 1) if truck[i] != depot]
    if not customer_positions:
        return []

    seed_idx = random.choice(customer_positions)
    seed_node = truck[seed_idx]

    scored = []
    for idx in customer_positions:
        node = truck[idx]
        distance = truck_times[seed_node][node]
        order_gap = abs(idx - seed_idx)
        relatedness = distance + 0.15 * order_gap
        scored.append((relatedness, idx))

    scored.sort(key=lambda x: x[0])
    pool_size = min(len(scored), max(count * 3, 8))
    pool = scored[:pool_size]

    picked = []
    work = pool[:]
    top_k = min(len(work), max(count * 2, 5))
    while work and len(picked) < count:
        chosen = _rank_biased_pick(work, top_k=top_k)
        if chosen is None:
            break
        _, idx = chosen
        picked.append(idx)
        work.remove(chosen)
        top_k = min(len(work), max(count * 2, 5))

    return sorted(set(picked), reverse=True)


def _shift_or_drop_after_truck_removal(route, removed_idx, removed_node):
    kept = []
    orphan_nodes = []
    for node, launch_idx, land_idx in route:
        if node == removed_node or launch_idx == removed_idx or land_idx == removed_idx:
            orphan_nodes.append(node)
            continue
        if launch_idx > removed_idx:
            launch_idx -= 1
        if land_idx > removed_idx:
            land_idx -= 1
        kept.append((node, launch_idx, land_idx))
    kept.sort(key=lambda x: (x[1], x[2], x[0]))
    return kept, orphan_nodes


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
    return best_delta, best_idx


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
    n_customers = max(1, len(truck) - 2)
    remove_count = _destroy_size(n_customers)

    destroy_indices = _pick_related_indices(truck, truck_times, depot, remove_count)
    if not destroy_indices:
        return current_solution

    pending = []
    for rem_idx in destroy_indices:
        removed_node = truck[rem_idx]
        pending.append(removed_node)
        truck = truck[:rem_idx] + truck[rem_idx + 1 :]
        drone1, orphan1 = _shift_or_drop_after_truck_removal(drone1, rem_idx, removed_node)
        drone2, orphan2 = _shift_or_drop_after_truck_removal(drone2, rem_idx, removed_node)
        pending.extend(orphan1)
        pending.extend(orphan2)

    unique_pending = []
    seen = set()
    for node in pending:
        if node in seen:
            continue
        seen.add(node)
        unique_pending.append(node)

    while unique_pending:
        best = None
        best_node = None
        for node in unique_pending:
            delta, ins_idx = _best_truck_insert(node, truck, truck_times)
            if ins_idx is None:
                continue
            cand = (delta, ins_idx, node)
            if best is None or cand < best:
                best = cand
                best_node = node

        if best is None:
            return current_solution

        _, ins_idx, node = best
        truck = truck[:ins_idx] + [node] + truck[ins_idx:]
        drone1 = _shift_route_after_truck_insert(drone1, ins_idx)
        drone2 = _shift_route_after_truck_insert(drone2, ins_idx)
        unique_pending.remove(best_node)

    candidate = [truck, drone1, drone2]
    if candidate == current_solution:
        return current_solution
    return candidate

