from read_file import read_instance
from op import operator, set_operator_context
from pathlib import Path
import sys
import importlib.util
import copy
import time
import random

GENERAL_ALGORITHM_PATH = Path(__file__).resolve().parents[1] / "General_algorithm"
if str(GENERAL_ALGORITHM_PATH) not in sys.path:
    sys.path.append(str(GENERAL_ALGORITHM_PATH))

try:
    from CalCulateTotalArrivalTime import CalCulateTotalArrivalTime
    from FeasibiltyCheck import SolutionFeasibility
except ModuleNotFoundError:
    calc_path = GENERAL_ALGORITHM_PATH / "CalCulateTotalArrivalTime.py"
    feas_path = GENERAL_ALGORITHM_PATH / "FeasibiltyCheck.py"

    calc_spec = importlib.util.spec_from_file_location("CalCulateTotalArrivalTime", calc_path)
    calc_mod = importlib.util.module_from_spec(calc_spec)
    calc_spec.loader.exec_module(calc_mod)
    CalCulateTotalArrivalTime = calc_mod.CalCulateTotalArrivalTime

    feas_spec = importlib.util.spec_from_file_location("FeasibiltyCheck", feas_path)
    feas_mod = importlib.util.module_from_spec(feas_spec)
    feas_spec.loader.exec_module(feas_mod)
    SolutionFeasibility = feas_mod.SolutionFeasibility

absolute_path = "/Users/niklasmarton/Library/CloudStorage/OneDrive-Personlig/ITØK/Metaheuristics/TrucksAndDrones/Test_files/"
file_name = "R_100.txt"

instance = read_instance(f"{absolute_path}{file_name}")

# Unpack fields.
T = instance["truck_times"]
D = instance["drone_times"]
flight_limit = instance["flight_limit"]
n_customers = instance["n_customers"]
depot = instance.get("depot_index", 0)

set_operator_context(T, D, flight_limit, depot)


def to_parts_solution(solution):
    """
    Convert tuple-based solution format:
    [truck, drone1[(node, launch_idx, land_idx)], drone2[(node, launch_idx, land_idx)]]
    to the dictionary format expected by feasibility and objective functions.
    """
    truck, drone1, drone2 = solution

    drone_serving_1 = [node for node, _, _ in drone1]
    drone_serving_2 = [node for node, _, _ in drone2]

    # 1-based indices in part3/part4.
    launch_indices_1 = [launch_idx + 1 for _, launch_idx, _ in drone1]
    landing_indices_1 = [land_idx + 1 for _, _, land_idx in drone1]
    launch_indices_2 = [launch_idx + 1 for _, launch_idx, _ in drone2]
    landing_indices_2 = [land_idx + 1 for _, _, land_idx in drone2]

    drone_serving = drone_serving_1 + [-1] + drone_serving_2
    drone_total_launches = launch_indices_1 + [-1] + launch_indices_2
    drone_total_landings = landing_indices_1 + [-1] + landing_indices_2

    return {
        "part1": truck,
        "part2": drone_serving,
        "part3": drone_total_launches,
        "part4": drone_total_landings,
    }


def evaluate_solution(solution, calc, checker):
    parts_solution = to_parts_solution(solution)
    # Structural/route feasibility check.
    if not checker.is_solution_feasible(parts_solution):
        return False, float("inf")

    # Timing-based feasibility (captures range violations with synchronization waiting).
    total_time, _, _, calc_feasible = calc.calculate_total_waiting_time(parts_solution)
    if not calc_feasible:
        return False, float("inf")

    return True, total_time


def solution_key(solution):
    truck, drone1, drone2 = solution
    return (tuple(truck), tuple(drone1), tuple(drone2))


def fast_precheck_solution(solution):
    """
    Cheap filter to reject clearly invalid candidates before expensive checks.
    Uses internal (0-based) tuple representation.
    """
    truck, drone1, drone2 = solution
    truck_len = len(truck)

    if truck_len < 2 or truck[0] != depot or truck[-1] != depot:
        return False

    truck_customers = [node for node in truck if node != depot]
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
            if not (0 <= launch_idx < land_idx < truck_len):
                return False
            if launch_idx in used_launch or land_idx in used_land:
                return False
            if launch_idx < prev_land:
                return False

            launch_node = truck[launch_idx]
            land_node = truck[land_idx]
            trip_time = D[launch_node][node] + D[node][land_node]
            if trip_time > flight_limit:
                return False

            used_launch.add(launch_idx)
            used_land.add(land_idx)
            prev_land = land_idx
            drone_nodes.append(node)

    if len(drone_nodes) != len(set(drone_nodes)):
        return False

    all_served = truck_customers + drone_nodes
    if len(all_served) != n_customers:
        return False
    if len(set(all_served)) != n_customers:
        return False

    return True


def local_search(initial_solution, iterations=10000):
    """
    Local search (1-reinsert neighborhood), following pseudocode:
      BestSolution <- s0
      for iter in 1..N:
          NewSolution <- Operator(BestSolution)
          if NewSolution feasible and f(NewSolution) < f(BestSolution):
              BestSolution <- NewSolution
    """
    calc = CalCulateTotalArrivalTime()
    calc.truck_times = T
    calc.drone_times = D
    calc.flight_range = flight_limit
    calc.depot_index = depot
    # Ensure operator uses the same active instance as evaluator/checker.
    set_operator_context(T, D, flight_limit, depot)

    checker = SolutionFeasibility(
        n_nodes=n_customers + 1,
        n_drones=2,
        depot_index=depot,
        drone_times=D,
        flight_range=flight_limit,
    )

    best_solution = copy.deepcopy(initial_solution)
    best_key = solution_key(best_solution)
    eval_cache = {}
    best_feasible, best_cost = evaluate_solution(best_solution, calc, checker)
    eval_cache[best_key] = (best_feasible, best_cost)
    if not best_feasible:
        raise ValueError("Initial solution is not feasible.")
    current_best_key = solution_key(best_solution)
    seen_neighbors = set()

    for _ in range(iterations):
        new_solution = operator(best_solution)
        new_key = solution_key(new_solution)

        # Skip repeated sampled move outcome from the same BestSolution.
        if new_key in seen_neighbors:
            continue
        seen_neighbors.add(new_key)

        if not fast_precheck_solution(new_solution):
            continue

        if new_key in eval_cache:
            feasible, new_cost = eval_cache[new_key]
        else:
            feasible, new_cost = evaluate_solution(new_solution, calc, checker)
            eval_cache[new_key] = (feasible, new_cost)

        if feasible and new_cost < best_cost:
            best_solution = new_solution
            best_cost = new_cost
            new_best_key = solution_key(best_solution)
            if new_best_key != current_best_key:
                current_best_key = new_best_key
                seen_neighbors = set()

    # Safety: never return an infeasible best.
    best_parts = to_parts_solution(best_solution)
    assert checker.is_solution_feasible(best_parts)
    assert calc.calculate_total_waiting_time(best_parts)[3]

    return best_solution, best_cost


def run_statistics(initial_solution, runs=10, iterations=10000):
    calc = CalCulateTotalArrivalTime()
    calc.truck_times = T
    calc.drone_times = D
    calc.flight_range = flight_limit
    calc.depot_index = depot

    checker = SolutionFeasibility(
        n_nodes=n_customers + 1,
        n_drones=2,
        depot_index=depot,
        drone_times=D,
        flight_range=flight_limit,
    )

    init_feasible, init_cost = evaluate_solution(initial_solution, calc, checker)
    if not init_feasible:
        raise ValueError("Initial solution is not feasible; cannot compute improvement statistics.")

    run_costs = []
    run_times = []
    global_best_solution = None
    global_best_cost = float("inf")

    for _ in range(runs):
        start = time.perf_counter()
        best_solution, best_cost = local_search(copy.deepcopy(initial_solution), iterations=iterations)
        elapsed = time.perf_counter() - start

        run_costs.append(best_cost)
        run_times.append(elapsed)

        if best_cost < global_best_cost:
            global_best_cost = best_cost
            global_best_solution = best_solution

    avg_obj = sum(run_costs) / len(run_costs)
    best_obj = global_best_cost
    improvement_abs = init_cost - best_obj
    improvement_pct = (improvement_abs / init_cost * 100.0) if init_cost > 0 else 0.0
    avg_runtime_per_run = sum(run_times) / len(run_times)

    print(f"Initial objective value: {init_cost}")
    print(f"Average objective value (over {runs} runs): {avg_obj}")
    print(f"Best objective: {best_obj}")
    print(
        f"Improvement (initial -> best): {improvement_abs} "
        f"({improvement_pct:.2f}% reduction)"
    )
    print(
        f"Average runtime per iteration (1 iteration = {iterations} local-search loops): "
        f"{avg_runtime_per_run:.4f} seconds"
    )

    return global_best_solution, global_best_cost


def format_solution_pipe(solution):
    # Checker-compatible display format (1-based launch/landing cells).
    parts = to_parts_solution(solution)
    part1 = ",".join(str(x) for x in parts["part1"])
    part2 = ",".join(str(x) for x in parts["part2"])
    part3 = ",".join(str(x) for x in parts["part3"])
    part4 = ",".join(str(x) for x in parts["part4"])
    return (
        f"{part1} | "
        f"{part2} | "
        f"{part3} | "
        f"{part4}"
    )


if __name__ == "__main__":
    truck_route = [i for i in range(n_customers + 1)] + [0]
    drone1 = []
    drone2 = []
    initial_solution = [truck_route, drone1, drone2]

    # print(f"Old solution was: {initial_solution}")
    best_solution, best_cost = run_statistics(initial_solution, runs=10, iterations=10000)
    # print(f"Best solution after all runs: {best_solution}")
    print(f"Best solution (pipe format): {format_solution_pipe(best_solution)}")
