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


def _rank_biased_pick(candidates, top_k):
    if not candidates:
        return None
    k = min(len(candidates), top_k)
    if k <= 1:
        return candidates[0]
    weights = [k - i for i in range(k)]
    return random.choices(candidates[:k], weights=weights, k=1)[0]


def _destroy_size(n_customers):
    low = max(4, int(round(0.08 * n_customers)))
    high = max(low, int(round(0.18 * n_customers)))
    high = min(high, max(4, n_customers))
    return random.randint(low, high)


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
        relatedness = distance + 0.20 * order_gap
        scored.append((relatedness, idx))
    scored.sort(key=lambda x: x[0])

    pool_size = min(len(scored), max(count * 3, 12))
    pool = scored[:pool_size]
    work = pool[:]
    chosen = []
    top_k = min(len(work), max(count * 2, 6))
    while work and len(chosen) < count:
        picked = _rank_biased_pick(work, top_k=top_k)
        if picked is None:
            break
        _, idx = picked
        chosen.append(idx)
        work.remove(picked)
        top_k = min(len(work), max(count * 2, 6))

    return sorted(set(chosen), reverse=True)


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


def _best_two_truck_inserts(node, truck, truck_times):
    best = None
    second = None
    for ins_idx in range(1, len(truck)):
        a = truck[ins_idx - 1]
        b = truck[ins_idx]
        delta = truck_times[a][node] + truck_times[node][b] - truck_times[a][b]
        cand = (delta, ins_idx)
        if best is None or cand < best:
            second = best
            best = cand
        elif second is None or cand < second:
            second = cand
    return best, second


def _regret_repair(pending_nodes, truck, drone1, drone2, truck_times):
    pending = pending_nodes[:]

    while pending:
        best_choice = None
        for node in pending:
            best_ins, second_ins = _best_two_truck_inserts(node, truck, truck_times)
            if best_ins is None:
                continue
            best_delta, best_idx = best_ins
            second_delta = second_ins[0] if second_ins is not None else (best_delta + 1e6)
            regret = second_delta - best_delta
            cand = (-regret, best_delta, best_idx, node)
            if best_choice is None or cand < best_choice:
                best_choice = cand

        if best_choice is None:
            return None

        _, _, ins_idx, node = best_choice
        truck = truck[:ins_idx] + [node] + truck[ins_idx:]
        drone1 = _shift_route_after_truck_insert(drone1, ins_idx)
        drone2 = _shift_route_after_truck_insert(drone2, ins_idx)
        pending.remove(node)

    return [truck, drone1, drone2]


def operator(current_solution):
    assert_context_is_set()
    truck_times, _, _, depot = get_operator_context()

    truck, drone1, drone2 = _clone_solution(current_solution)
    n_customers = max(1, len(truck) - 2)
    if n_customers < 8:
        return current_solution

    remove_count = min(n_customers, _destroy_size(n_customers))
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

    candidate = _regret_repair(unique_pending, truck, drone1, drone2, truck_times)
    if candidate is None or candidate == current_solution:
        return current_solution
    return candidate

