# ALNS solver for the truck and drone routing problem.
# entry point: solve(instance_path, time_limit_seconds) -> dict with
# best_objective, pipe, runtime, solution.
# all paths inside the project are relative to this file. no analysis,
# plotting or logging here, this is the lean runtime version.

from collections import OrderedDict
from pathlib import Path
import sys
import random
import time

PROJECT_DIR = Path(__file__).resolve().parents[1]
CORE_DIR = PROJECT_DIR / "core"
OPS_DIR = PROJECT_DIR / "operators"
IO_DIR = PROJECT_DIR / "instance_io"
ALG_DIR = PROJECT_DIR / "algorithm"
for _p in (CORE_DIR, OPS_DIR, IO_DIR, ALG_DIR):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

from read_file import read_instance
from CalCulateTotalArrivalTime import CalCulateTotalArrivalTime
from FeasibiltyCheck import SolutionFeasibility

import op1_reinsert as op1
import op2_destroy_repair as op2
import op3_or_opt as op3
import op4_related_destroy as op4
import op5_truck_2opt as op5
import op6_TSP_drone_rebuild as op6
import op7_truck_drone_swap as op7
import op8_drone_sync_tuner as op8
import op9_drone_relocate as op9
import op10_full_rebuild as op10

import escape

import local_search as ls


# solution helpers
def clone_solution(solution):
    return [solution[0][:], solution[1][:], solution[2][:]]


def load_instance(instance_path):
    return read_instance(str(instance_path))


def unpack_instance(instance_data):
    return {
        "T": instance_data["truck_times"],
        "D": instance_data["drone_times"],
        "flight_limit": instance_data["flight_limit"],
        "n_customers": instance_data["n_customers"],
        "depot": instance_data.get("depot_index", 0),
    }


def configure_operator_context(instance_data):
    T = instance_data["truck_times"]
    D = instance_data["drone_times"]
    fr = instance_data["flight_limit"]
    depot = instance_data.get("depot_index", 0)
    for op in (op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, escape):
        if hasattr(op, "set_operator_context"):
            op.set_operator_context(T, D, fr, depot)


def configure_operator_search_progress(progress):
    for op in (op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, escape):
        if hasattr(op, "set_search_progress"):
            op.set_search_progress(progress)


def build_evaluator(instance_data):
    ctx = unpack_instance(instance_data)
    calc = CalCulateTotalArrivalTime()
    calc.truck_times = ctx["T"]
    calc.drone_times = ctx["D"]
    calc.flight_range = ctx["flight_limit"]
    calc.depot_index = ctx["depot"]

    checker = SolutionFeasibility(
        n_nodes=ctx["n_customers"] + 1,
        n_drones=2,
        depot_index=ctx["depot"],
        drone_times=ctx["D"],
        flight_range=ctx["flight_limit"],
    )
    return ctx, calc, checker


def to_parts_solution(solution):
    truck, drone1, drone2 = solution
    drone_serving_1 = [node for node, _, _ in drone1]
    drone_serving_2 = [node for node, _, _ in drone2]
    launch_1 = [li + 1 for _, li, _ in drone1]
    land_1 = [ri + 1 for _, _, ri in drone1]
    launch_2 = [li + 1 for _, li, _ in drone2]
    land_2 = [ri + 1 for _, _, ri in drone2]
    return {
        "part1": truck,
        "part2": drone_serving_1 + [-1] + drone_serving_2,
        "part3": launch_1 + [-1] + launch_2,
        "part4": land_1 + [-1] + land_2,
    }


def evaluate_solution(solution, calc, checker):
    parts = to_parts_solution(solution)
    if not checker.is_solution_feasible(parts):
        return False, float("inf")
    total_time, _, _, calc_feasible = calc.calculate_total_waiting_time(parts)
    if not calc_feasible:
        return False, float("inf")
    return True, total_time


def solution_key(solution):
    truck, drone1, drone2 = solution
    return (tuple(truck), tuple(drone1), tuple(drone2))


def fast_precheck_solution(solution, ctx):
    truck, drone1, drone2 = solution
    D = ctx["D"]
    flight_limit = ctx["flight_limit"]
    n_customers = ctx["n_customers"]
    depot = ctx["depot"]

    if len(truck) < 2 or truck[0] != depot or truck[-1] != depot:
        return False
    truck_customers = [n for n in truck if n != depot]
    if len(truck_customers) != len(set(truck_customers)):
        return False

    drone_nodes = []
    for route in (drone1, drone2):
        prev_land = 0
        used_launch = set()
        used_land = set()
        for node, launch_idx, land_idx in route:
            if node == depot:
                return False
            if not (0 <= launch_idx < land_idx < len(truck)):
                return False
            if launch_idx in used_launch or land_idx in used_land:
                return False
            if launch_idx < prev_land:
                return False
            launch_node = truck[launch_idx]
            land_node = truck[land_idx]
            if D[launch_node][node] + D[node][land_node] > flight_limit:
                return False
            used_launch.add(launch_idx)
            used_land.add(land_idx)
            prev_land = land_idx
            drone_nodes.append(node)

    if len(drone_nodes) != len(set(drone_nodes)):
        return False
    all_served = truck_customers + drone_nodes
    if len(all_served) != n_customers or len(set(all_served)) != n_customers:
        return False
    return True


def format_solution_pipe(solution):
    parts = to_parts_solution(solution)
    return (
        f"{','.join(str(x) for x in parts['part1'])} | "
        f"{','.join(str(x) for x in parts['part2'])} | "
        f"{','.join(str(x) for x in parts['part3'])} | "
        f"{','.join(str(x) for x in parts['part4'])}"
    )


# initial solution: NN walk with cyclic vehicle assignment.
#  1. NN walk assigns each customer to truck/drone1/drone2 in a shuffled cycle
#  2. all customers start on the truck in NN visit order
#  3. the ones marked for a drone get pulled off the truck one at a time,
#     using the best static-hover-feasible (launch, land) pair. if no valid
#     pair exists the customer stays on the truck.
#  4. availability pass drops drone trips that break the multi-trip timeline,
#     dropped customers fall back to the truck.
#  5. dynamic-calculator safety pass catches any trip that still slips through
#     because of cross-drone cascade effects.
def build_initial_solution(instance_data):
    n_customers = instance_data["n_customers"]
    depot = instance_data.get("depot_index", 0)
    T = instance_data["truck_times"]
    D = instance_data["drone_times"]
    flight_limit = instance_data["flight_limit"]

    # --- Step 1: NN walk with cyclic vehicle assignment ---
    unvisited = list(range(1, n_customers + 1))
    visit_order = []
    assignment = {}  # node -> 0 (truck), 1 (drone1), 2 (drone2)
    current = depot
    cycle = [0, 1, 2]
    random.shuffle(cycle)
    step = 0
    while unvisited:
        nearest = min(unvisited, key=lambda n: T[current][n])
        unvisited.remove(nearest)
        current = nearest
        visit_order.append(nearest)
        assignment[nearest] = cycle[step % 3]
        step += 1

    # --- Step 2: All customers on the truck in NN visit order ---
    truck = [depot] + visit_order + [depot]
    drone1, drone2 = [], []

    def _truck_prefix():
        prefix = [0.0]
        for i in range(len(truck) - 1):
            prefix.append(prefix[-1] + T[truck[i]][truck[i + 1]])
        return prefix

    def _shift_after_remove(route, removed_idx):
        return [(n, l - (l > removed_idx), r - (r > removed_idx))
                for n, l, r in route]

    def _shift_after_insert(route, insert_idx):
        return [(n, l + (l >= insert_idx), r + (r >= insert_idx))
                for n, l, r in route]

    def _find_pair_in(node, temp_truck, drone_route):
        nt = len(temp_truck)
        prefix = [0.0]
        for i in range(nt - 1):
            prefix.append(prefix[-1] + T[temp_truck[i]][temp_truck[i + 1]])
        used_l = {l for _, l, _ in drone_route}
        used_r = {r for _, _, r in drone_route}

        sorted_trips = sorted(drone_route, key=lambda x: (x[1], x[2], x[0]))
        gaps = []
        for i in range(len(sorted_trips) + 1):
            prev_land = 0 if i == 0 else sorted_trips[i - 1][2]
            next_launch = float("inf") if i == len(sorted_trips) else sorted_trips[i][1]
            gaps.append((prev_land, next_launch))

        def _fits_in_gap(l, r):
            for prev_land, next_launch in gaps:
                if l >= prev_land and r <= next_launch:
                    return True
            return False

        # Conservative limit: leaves headroom for cross-drone truck delays.
        safe_limit = flight_limit * 0.88

        best_score, best_l, best_r = float("inf"), None, None
        for l in range(1, nt - 1):
            if l in used_l:
                continue
            for r in range(l + 1, nt - 1):
                if r in used_r:
                    continue
                if not _fits_in_gap(l, r):
                    continue
                df = D[temp_truck[l]][node] + D[node][temp_truck[r]]
                if df > safe_limit:
                    continue
                ts = prefix[r] - prefix[l]
                if max(df, ts) > safe_limit:
                    continue
                hover = max(0.0, ts - df)
                score = 2.0 * hover + 0.5 * abs(df - ts) + 0.01 * df
                if score < best_score:
                    best_score = score
                    best_l, best_r = l, r
        return (best_l, best_r) if best_l is not None else None

    def _availability_pass(drone_route):
        prefix = _truck_prefix()
        trips = sorted(drone_route, key=lambda x: (x[1], x[2], x[0]))
        kept, fallback, drone_est_return = [], [], 0.0
        for node, l, r in trips:
            df = D[truck[l]][node] + D[node][truck[r]]
            actual_launch = max(prefix[l], drone_est_return)
            drone_return_t = actual_launch + df
            drone_hover = max(0.0, prefix[r] - drone_return_t)
            if df + drone_hover > flight_limit:
                fallback.append(node)
            else:
                kept.append((node, l, r))
                drone_est_return = drone_return_t
        return kept, fallback

    # --- Step 3: Move drone-assigned customers from truck to their drone ---
    for target, drone_route in [(1, drone1), (2, drone2)]:
        candidates = [n for n in visit_order if assignment[n] == target]
        for node in candidates:
            node_idx = truck.index(node)
            all_used_l = {l for _, l, _ in drone1} | {l for _, l, _ in drone2}
            all_used_r = {r for _, _, r in drone1} | {r for _, _, r in drone2}
            if node_idx in all_used_l or node_idx in all_used_r:
                continue
            temp_truck = truck[:node_idx] + truck[node_idx + 1:]
            shifted_d1 = _shift_after_remove(drone1, node_idx)
            shifted_d2 = _shift_after_remove(drone2, node_idx)
            shifted_target = shifted_d1 if target == 1 else shifted_d2
            pair = _find_pair_in(node, temp_truck, shifted_target)
            if pair is None:
                continue
            truck[:] = temp_truck
            drone1[:] = shifted_d1
            drone2[:] = shifted_d2
            drone_route.append((node, pair[0], pair[1]))
            drone_route.sort(key=lambda x: (x[1], x[2], x[0]))

    # --- Step 4: Multi-trip availability pass ---
    drone1[:], fb1 = _availability_pass(drone1)
    drone2[:], fb2 = _availability_pass(drone2)
    for node in fb1 + fb2:
        _, ins_idx = _best_truck_insert(node, truck, T)
        if ins_idx is None:
            ins_idx = len(truck) - 1
        drone1[:] = _shift_after_insert(drone1, ins_idx)
        drone2[:] = _shift_after_insert(drone2, ins_idx)
        truck[:] = truck[:ins_idx] + [node] + truck[ins_idx:]

    # --- Step 5: Final safety pass with the dynamic calculator ---
    _, calc_local, checker_local = build_evaluator(instance_data)
    for _attempt in range(len(drone1) + len(drone2) + 2):
        feasible, _ = evaluate_solution([truck, drone1, drone2], calc_local, checker_local)
        if feasible:
            break
        if not drone1 and not drone2:
            break
        target_route = drone1 if len(drone1) >= len(drone2) else drone2
        worst = target_route[-1]
        target_route.pop()
        node_to_restore = worst[0]
        _, ins_idx = _best_truck_insert(node_to_restore, truck, T)
        if ins_idx is None:
            ins_idx = len(truck) - 1
        drone1[:] = _shift_after_insert(drone1, ins_idx)
        drone2[:] = _shift_after_insert(drone2, ins_idx)
        truck[:] = truck[:ins_idx] + [node_to_restore] + truck[ins_idx:]

    return [truck, drone1, drone2]


# operator dispatch and weight normalisation
def _normalize_weight_dict(weights, min_weight=0.03):
    fixed = {k: max(min_weight, float(v)) for k, v in weights.items()}
    s = sum(fixed.values())
    return {k: v / s for k, v in fixed.items()}


def _normalize_weight_dict_with_caps(weights, min_weight=0.03,
                                     lower_caps=None, upper_caps=None):
    lower_caps = lower_caps or {}
    upper_caps = upper_caps or {}
    keys = list(weights.keys())
    if not keys:
        return {}
    base = {k: max(min_weight, float(weights[k])) for k in keys}
    lower = {k: max(min_weight, float(lower_caps.get(k, min_weight))) for k in keys}
    upper = {k: float(upper_caps.get(k, 1.0)) for k in keys}
    for k in keys:
        if upper[k] < lower[k]:
            upper[k] = lower[k]
    x = {k: min(max(base[k], lower[k]), upper[k]) for k in keys}
    for _ in range(24):
        s = sum(x.values())
        if abs(s - 1.0) <= 1e-12:
            break
        if s > 1.0:
            free = [k for k in keys if x[k] > lower[k] + 1e-12]
            if not free: break
            excess = s - 1.0
            room = sum(x[k] - lower[k] for k in free)
            if room <= 1e-12: break
            for k in free:
                x[k] = max(lower[k], x[k] - excess * (x[k] - lower[k]) / room)
        else:
            free = [k for k in keys if x[k] < upper[k] - 1e-12]
            if not free: break
            deficit = 1.0 - s
            room = sum(upper[k] - x[k] for k in free)
            if room <= 1e-12: break
            for k in free:
                x[k] = min(upper[k], x[k] + deficit * (upper[k] - x[k]) / room)
    total = sum(x.values())
    if total <= 0:
        return _normalize_weight_dict(weights, min_weight=min_weight)
    return {k: x[k] / total for k in keys}


def roulette_pick(weights, op_names):
    r = random.random()
    acc = 0.0
    for op_name in op_names:
        acc += weights[op_name]
        if r <= acc:
            return op_name
    return op_names[-1]


def apply_main_operator(solution, op_name):
    if op_name == "op1":  return op1.operator(solution)
    if op_name == "op2":  return op2.operator(solution)
    if op_name == "op3":  return op3.operator(solution)
    if op_name == "op4":  return op4.operator(solution)
    if op_name == "op5":  return op5.operator(solution)
    if op_name == "op6":  return op6.operator(solution)
    if op_name == "op7":  return op7.operator(solution)
    if op_name == "op8":  return op8.operator(solution)
    if op_name == "op9":  return op9.operator(solution)
    if op_name == "op10": return op10.operator(solution)
    raise ValueError(f"Unknown operator {op_name}")


# escape: double-bridge primary, large related-destroy fallback, aggressive D&R last resort
def _truck_removal_gain(truck, idx, truck_times):
    if idx <= 0 or idx >= len(truck) - 1:
        return 0.0
    prev_node, node, next_node = truck[idx - 1], truck[idx], truck[idx + 1]
    return (truck_times[prev_node][node] + truck_times[node][next_node]
            - truck_times[prev_node][next_node])


def _best_truck_insert(node, truck, truck_times):
    best_cost = float("inf")
    best_idx = None
    for i in range(1, len(truck)):
        cost = (truck_times[truck[i - 1]][node] + truck_times[node][truck[i]]
                - truck_times[truck[i - 1]][truck[i]])
        if cost < best_cost:
            best_cost = cost
            best_idx = i
    return best_cost, best_idx


def _remove_conflicting_and_shift_after_removal(route, removed_idx, removed_node):
    survivors = []
    orphans = []
    for n, l, r in route:
        if l == removed_idx or r == removed_idx or n == removed_node:
            orphans.append(n)
            continue
        survivors.append((n, l - (l > removed_idx), r - (r > removed_idx)))
    return survivors, orphans


def _remove_conflicts_and_shift_after_insert(route, insert_idx):
    survivors = []
    orphans = []
    for n, l, r in route:
        new_l = l + (l >= insert_idx)
        new_r = r + (r >= insert_idx)
        if new_l >= new_r:
            orphans.append(n)
            continue
        survivors.append((n, new_l, new_r))
    return survivors, orphans


def _aggressive_escape_destroy_repair(solution, ctx):
    truck_times = ctx["T"]
    truck, drone1, drone2 = clone_solution(solution)
    if len(truck) <= 7:
        return solution
    scored = [(_truck_removal_gain(truck, idx, truck_times), idx)
              for idx in range(1, len(truck) - 1)]
    if not scored:
        return solution
    scored.sort(key=lambda x: x[0], reverse=True)
    top10 = scored[: min(10, len(scored))]
    picked = ([idx for _, idx in top10] if len(top10) <= 5
              else [idx for _, idx in random.sample(top10, 5)])
    picked = sorted(set(picked), reverse=True)

    pending = []
    for rem_idx in picked:
        rem_node = truck[rem_idx]
        pending.append(rem_node)
        truck = truck[:rem_idx] + truck[rem_idx + 1:]
        drone1, o1 = _remove_conflicting_and_shift_after_removal(drone1, rem_idx, rem_node)
        drone2, o2 = _remove_conflicting_and_shift_after_removal(drone2, rem_idx, rem_node)
        pending.extend(o1); pending.extend(o2)

    seen = set()
    pending_unique = []
    for n in pending:
        if n in seen: continue
        seen.add(n); pending_unique.append(n)

    for node in pending_unique:
        _, ins_idx = _best_truck_insert(node, truck, truck_times)
        if ins_idx is None:
            continue
        drone1, o1 = _remove_conflicts_and_shift_after_insert(drone1, ins_idx)
        drone2, o2 = _remove_conflicts_and_shift_after_insert(drone2, ins_idx)
        truck = truck[:ins_idx] + [node] + truck[ins_idx:]
        for orphan in o1 + o2:
            if orphan not in seen:
                seen.add(orphan); pending_unique.append(orphan)

    return [truck, drone1, drone2]


def _escape_with_related_large(incumbent, incumbent_cost, best_solution,
                               best_cost, ctx, cached_evaluate):
    current = clone_solution(best_solution)
    current_cost = best_cost
    improved_best = False
    feasible_steps = 0

    for _ in range(3):
        cand = escape.operator(current)
        if cand is current or cand == current:
            continue
        if not fast_precheck_solution(cand, ctx):
            continue
        feasible, cand_cost = cached_evaluate(cand)
        if not feasible:
            continue
        current = cand; current_cost = cand_cost
        feasible_steps += 1
        if current_cost < best_cost:
            improved_best = True
        break

    if feasible_steps == 0:
        n_customers = ctx.get("n_customers", 50)
        n_steps = max(3, min(6, n_customers // 15))
        for _ in range(n_steps):
            cand = escape.fallback(current)
            if cand is current or cand == current:
                continue
            if not fast_precheck_solution(cand, ctx):
                continue
            feasible, cand_cost = cached_evaluate(cand)
            if not feasible:
                continue
            current = cand; current_cost = cand_cost
            feasible_steps += 1
            if current_cost < best_cost:
                improved_best = True

    if feasible_steps == 0:
        fallback = _aggressive_escape_destroy_repair(clone_solution(best_solution), ctx)
        if fallback != best_solution and fast_precheck_solution(fallback, ctx):
            feasible, fb_cost = cached_evaluate(fallback)
            if feasible:
                current = fallback; current_cost = fb_cost
                feasible_steps += 1
                if current_cost < best_cost:
                    improved_best = True

    return current, current_cost, improved_best, feasible_steps


# ALNS main loop
def alns_improved(initial_solution, instance_data, ctx, calc, checker,
                  *, time_limit_seconds, warmup_iterations=500,
                  iterations=10_000_000, cache_limit=200_000,
                  reaction_factor=0.15, segment_length=None,
                  escape_stall_limit=None,
                  rrt_deviation_factor=0.13, rrt_decay_exponent=1.0,
                  reward_improve_threshold=50.0,
                  sigma_global_best=8.0, sigma_incumbent_improve=4.0,
                  sigma_small_improve=1.0, sigma_uphill_accepted=0.6,
                  enable_local_search=True,
                  ls_max_cycles=None,
                  ls_end_time_budget_seconds=None,
                  progress_label=""):
    # one ALNS run with end-of-run local search.
    # returns (best_solution, best_cost).
    # size-tiered operator set + RRT acceptance + double-bridge escape.
    n_cust = ctx.get("n_customers", 100)

    # operator set + RRT decay are picked based on instance size
    if n_cust <= 15:
        op_names = ["op1", "op2", "op3", "op4", "op5", "op6",
                    "op7", "op8", "op10"]
    elif n_cust <= 30:
        op_names = ["op1", "op2", "op3", "op5", "op7", "op8", "op10"]
    elif n_cust <= 60:
        op_names = ["op1", "op2", "op3", "op7", "op8", "op9"]
        rrt_deviation_factor = 0.05
    else:
        op_names = ["op1", "op2", "op3", "op7", "op8", "op9"]
        rrt_deviation_factor = 0.05
        rrt_decay_exponent = 1.5

    # end-of-run LS budget defaults
    if ls_max_cycles is None:
        ls_max_cycles = 30 if n_cust > 30 else 20
    if ls_end_time_budget_seconds is None:
        ls_end_time_budget_seconds = 40.0 if n_cust > 30 else 10.0

    # stall and segment defaults
    if segment_length is None:
        segment_length = max(15, n_cust)
    if escape_stall_limit is None:
        escape_stall_limit = max(800, 8 * n_cust) if n_cust > 30 else max(150, 4 * n_cust)

    eval_cache = OrderedDict()

    def cached_evaluate(sol):
        key = solution_key(sol)
        cached = eval_cache.get(key)
        if cached is not None:
            eval_cache.move_to_end(key)
            return cached
        result = evaluate_solution(sol, calc, checker)
        eval_cache[key] = result
        if cache_limit and len(eval_cache) > cache_limit:
            eval_cache.popitem(last=False)
        return result

    # Minimal stats: only what alns_improved consumes internally
    stats = {k: {"used": 0, "feasible": 0} for k in op_names}

    incumbent = clone_solution(initial_solution)
    feasible, incumbent_cost = cached_evaluate(incumbent)
    if not feasible:
        raise ValueError("Initial solution is not feasible.")

    best_solution = clone_solution(incumbent)
    best_cost = incumbent_cost

    total_steps = max(1, warmup_iterations + iterations)

    def _rrt_deviation(best_obj, g, G):
        G = max(1, int(G))
        g = max(0, min(int(g), G))
        frac = ((G - g) / G) ** float(rrt_decay_exponent)
        return float(rrt_deviation_factor) * frac * max(1.0, float(best_obj))

    # Warmup: random operator pick, no weights yet
    for w in range(warmup_iterations):
        configure_operator_search_progress(w / total_steps)
        op_name = random.choice(op_names)
        stats[op_name]["used"] += 1

        candidate = apply_main_operator(incumbent, op_name)
        if candidate is incumbent or candidate == incumbent:
            continue
        if not fast_precheck_solution(candidate, ctx):
            continue
        feasible, cand_cost = cached_evaluate(candidate)
        if not feasible:
            continue
        stats[op_name]["feasible"] += 1
        delta_e = cand_cost - incumbent_cost
        if delta_e < 0:
            incumbent = candidate
            incumbent_cost = cand_cost
            if incumbent_cost < best_cost:
                best_solution = clone_solution(incumbent)
                best_cost = incumbent_cost
        else:
            rrt_dev_w = _rrt_deviation(best_cost, w + 1, warmup_iterations)
            if cand_cost <= (best_cost + rrt_dev_w):
                incumbent = candidate
                incumbent_cost = cand_cost

    weights = {k: 1.0 / len(op_names) for k in op_names}
    segment_scores = {k: 0.0 for k in op_names}
    segment_uses = {k: 0 for k in op_names}

    no_best_improve_steps = 0
    _WARM_RESET_WINDOW = 500
    escape_warm_reset_until = -1

    _op6_suppressed = False

    _t_loop_start = time.perf_counter()
    _next_progress_min = 1
    _progress_prefix = f"[{progress_label}] " if progress_label else ""

    for it in range(iterations):
        _elapsed = time.perf_counter() - _t_loop_start
        if _elapsed >= time_limit_seconds:
            break
        if _elapsed >= 60.0 * _next_progress_min:
            print(f"{_progress_prefix}t={_next_progress_min}min  best={best_cost:.2f}", flush=True)
            _next_progress_min += 1
        _virt_total = max(1, int(time_limit_seconds))
        _virt_step = min(_virt_total, max(1, int(_elapsed)))

        if it <= escape_warm_reset_until:
            rrt_dev = float(rrt_deviation_factor) * max(1.0, float(best_cost))
        else:
            rrt_dev = _rrt_deviation(best_cost, _virt_step, _virt_total)

        configure_operator_search_progress(min(1.0, _elapsed / time_limit_seconds))
        op_name = roulette_pick(weights, op_names)
        stats[op_name]["used"] += 1
        segment_uses[op_name] += 1

        candidate = apply_main_operator(incumbent, op_name)
        if candidate is incumbent or candidate == incumbent:
            continue
        if not fast_precheck_solution(candidate, ctx):
            continue
        feasible, cand_cost = cached_evaluate(candidate)
        if not feasible:
            continue

        stats[op_name]["feasible"] += 1
        delta_e = cand_cost - incumbent_cost
        accepted = False
        improved_best = False

        if delta_e < 0:
            incumbent = candidate
            incumbent_cost = cand_cost
            accepted = True
            if incumbent_cost < best_cost:
                best_solution = clone_solution(incumbent)
                best_cost = incumbent_cost
                improved_best = True
        else:
            if cand_cost <= (best_cost + rrt_dev):
                incumbent = candidate
                incumbent_cost = cand_cost
                accepted = True

        # Reward bookkeeping (drives operator weight update each segment)
        improve_mag = max(0.0, -delta_e)
        improve_threshold = max(1.0, float(reward_improve_threshold))
        meaningful_improve = improve_mag >= improve_threshold
        magnitude_bonus = min(3.0, improve_mag / (2.0 * improve_threshold))
        segment_reward = 0.0

        if improved_best:
            segment_reward = float(sigma_global_best) + magnitude_bonus
            no_best_improve_steps = 0
        elif delta_e < 0:
            if meaningful_improve:
                segment_reward = float(sigma_incumbent_improve) + 0.5 * magnitude_bonus
            else:
                small_gain = improve_mag / improve_threshold
                segment_reward = float(sigma_small_improve) * max(0.0, min(1.0, small_gain))
            no_best_improve_steps += 1
        elif accepted:
            rrt_window = max(1.0, float(rrt_dev))
            if 0.0 < delta_e <= rrt_window:
                uphill_quality = max(0.0, 1.0 - (delta_e / rrt_window))
                segment_reward = float(sigma_uphill_accepted) * uphill_quality
            no_best_improve_steps += 1
        else:
            no_best_improve_steps += 1

        if segment_reward > 0.0:
            segment_scores[op_name] += segment_reward

        # End of segment: update operator weights with caps
        if (it + 1) % segment_length == 0:
            updated = {}
            progress = min(1.0, _elapsed / time_limit_seconds)
            for k in op_names:
                theta = segment_uses[k]
                if theta > 0:
                    effective_r = reaction_factor * (theta / (theta + 8.0))
                    updated[k] = (weights[k] * (1.0 - effective_r)
                                  + effective_r * (segment_scores[k] / theta))
                else:
                    if progress >= 0.7:
                        updated[k] = weights[k] * (1.0 - 0.35 * reaction_factor)
                    else:
                        updated[k] = weights[k] * (1.0 - reaction_factor)

            _upper = {}
            _lower = {}
            _suppress_check_due = _elapsed >= 0.20 * time_limit_seconds
            if "op6" in op_names and not _op6_suppressed and _suppress_check_due:
                if stats["op6"]["used"] >= 50 and (
                    stats["op6"]["feasible"] / stats["op6"]["used"] < 0.08
                ):
                    _op6_suppressed = True
            if _op6_suppressed:
                _upper["op6"] = 0.05
            weights = _normalize_weight_dict_with_caps(
                updated, min_weight=0.03,
                lower_caps=_lower, upper_caps=_upper,
            )
            segment_scores = {k: 0.0 for k in op_names}
            segment_uses = {k: 0 for k in op_names}

        # Escape on stall
        if no_best_improve_steps >= escape_stall_limit:
            incumbent, incumbent_cost, esc_improved, _ = _escape_with_related_large(
                incumbent, incumbent_cost, best_solution, best_cost,
                ctx, cached_evaluate,
            )
            if esc_improved and incumbent_cost < best_cost:
                best_solution = clone_solution(incumbent)
                best_cost = incumbent_cost
            no_best_improve_steps = 0
            escape_warm_reset_until = it + _WARM_RESET_WINDOW

    # End-of-run local search (best-improvement sweeps)
    if enable_local_search:
        ls_sol, ls_cost, _ = ls.local_search(
            best_solution, best_cost, cached_evaluate, ctx,
            max_cycles=ls_max_cycles,
            time_budget_seconds=ls_end_time_budget_seconds,
        )
        if ls_cost < best_cost - 1e-9:
            best_solution = clone_solution(ls_sol)
            best_cost = ls_cost

    # Hard feasibility re-check on the returned solution.
    best_parts = to_parts_solution(best_solution)
    assert checker.is_solution_feasible(best_parts), "Returned solution failed feasibility check."
    assert calc.calculate_total_waiting_time(best_parts)[3], "Returned solution failed cost feasibility."

    return best_solution, best_cost


# solve one instance file. time_limit_seconds is the ALNS budget; end-of-run
# LS adds up to ~40s on top for n>30 instances. returns a dict with keys
# best_objective, pipe, runtime, solution, n_customers.
def solve(instance_path, time_limit_seconds=600.0, seed=None):
    if seed is not None:
        random.seed(seed)

    instance_data = load_instance(instance_path)
    ctx, calc, checker = build_evaluator(instance_data)
    configure_operator_context(instance_data)

    initial = build_initial_solution(instance_data)

    instance_label = Path(instance_path).name
    t0 = time.perf_counter()
    best_solution, best_cost = alns_improved(
        initial, instance_data, ctx, calc, checker,
        time_limit_seconds=time_limit_seconds,
        progress_label=instance_label,
    )
    runtime = time.perf_counter() - t0

    return {
        "best_objective": float(best_cost),
        "pipe": format_solution_pipe(best_solution),
        "runtime": runtime,
        "solution": best_solution,
        "n_customers": instance_data["n_customers"],
    }
