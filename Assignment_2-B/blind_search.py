# blind_search.py
#
# Blind random search for the STRPD instance.
#
# Uses:
#   - instance.py                 -> read_instance(instance_path) -> dict
#   - random_solution.py          -> generate_random_solution(instance, n_drones)
#   - FeasibiltyCheck.py          -> class SolutionFeasibility   (one level up)
#   - CalCulateTotalArrivalTime.py -> class CalCulateTotalArrivalTime (one level up)
#
# Algorithm (as in the assignment):
#   1. Read data from text file.
#   2. Construct an initial solution:
#        - all customers are served by the truck
#        - in the order of their indices: 0, 1, 2, ..., N, 0
#   3. For a fixed number of iterations:
#        - Generate a random valid (but not necessarily feasible) solution.
#        - Check feasibility.
#        - If feasible and better than current best: store as best.
#   4. Repeat the whole search several times per instance.
#   5. Report:
#        - average objective over runs
#        - best objective
#        - improvement (%) over initial solution
#        - average running time

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)
# Allow importing shared utilities that live in ../General_algorithm
sys.path.append(os.path.join(PROJECT_ROOT, "General_algorithm"))

from instance import read_instance
from random_solution import generate_random_solution
from FeasibiltyCheck import SolutionFeasibility
from CalCulateTotalArrivalTime import CalCulateTotalArrivalTime


def construct_initial_solution(instance: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initial solution as specified in the assignment:
    All customers are served by the truck in increasing index order.

    Truck route: depot -> 1 -> 2 -> ... -> N -> depot
    No drones are used (parts 2–4 empty).
    """
    n_customers: int = instance["n_customers"]
    depot: int = instance.get("depot_index", 0)

    truck_route = [depot] + list(range(1, n_customers + 1)) + [depot]

    return {
        "part1": truck_route,  # truck route
        "part2": [],           # drone sequence (customers and -1 separators)
        "part3": [],           # launch cells
        "part4": [],           # reconvene cells
    }


def blind_random_search(
    instance_path: str,
    n_iterations: int = 10_000,
    n_runs: int = 10,
    n_drones: int = 3,
) -> Dict[str, Any]:
    """
    Perform blind random search on a single instance.

    Parameters
    ----------
    instance_path : str
        Path to the instance text file, e.g. "R_100.txt".
    n_iterations : int
        Number of random solutions generated per run.
    n_runs : int
        Number of independent runs (restarts).
    n_drones : int
        Number of drones in the problem.

    Returns
    -------
    dict
        Summary statistics: initial, average, best objective, improvement,
        average runtime, and best solution.
    """

    # ------------------------------------------------------------------
    # 1) Read instance data
    # ------------------------------------------------------------------
    instance = read_instance(instance_path)

    n_customers: int = instance["n_customers"]
    depot: int = instance.get("depot_index", 0)
    flight_limit: float = instance["flight_limit"]
    truck_times = instance["truck_times"]
    drone_times = instance["drone_times"]

    # ------------------------------------------------------------------
    # 2) Set up feasibility checker and travel-time calculator
    #    (You may need to adapt argument names to your actual classes.)
    # ------------------------------------------------------------------
    feas_checker = SolutionFeasibility(
        n_nodes=n_customers + 1,
        n_drones=n_drones,
        depot_index=depot,
        drone_times=drone_times,
        flight_range=flight_limit,
    )

    travel_time_calc = CalCulateTotalArrivalTime()
    # If your class uses attributes for data, set them here:
    travel_time_calc.truck_times = truck_times
    travel_time_calc.drone_times = drone_times
    travel_time_calc.n_customers = n_customers
    travel_time_calc.depot_index = depot
    travel_time_calc.n_drones = n_drones
    travel_time_calc.flight_range = flight_limit

    # ------------------------------------------------------------------
    # 3) Construct initial solution (truck-only, ordered by indices)
    #     Only keep it if it is feasible; otherwise, we start runs with no incumbent.
    # ------------------------------------------------------------------
    initial_solution = construct_initial_solution(instance)
    initial_obj = None
    if feas_checker.is_solution_feasible(initial_solution):
        tot, _, _, feas_flag = travel_time_calc.calculate_total_waiting_time(initial_solution)
        if feas_flag:
            initial_obj = tot

    # ------------------------------------------------------------------
    # 4) Blind random search: multiple runs
    # ------------------------------------------------------------------
    best_overall_obj = float("inf")
    best_overall_solution: Dict[str, Any] | None = None

    run_best_objectives: List[float] = []
    run_times: List[float] = []

    for run in range(n_runs):
        start_time = time.time()

        # Diagnostics: count how many generated/feasible solutions include drones
        generated_total = 0
        generated_with_drones = 0
        feasible_total = 0
        feasible_with_drones = 0

        # For each run, start from the initial feasible solution (if available)
        best_run_obj = initial_obj if initial_obj is not None else float("inf")
        best_run_solution = initial_solution if initial_obj is not None else None

        for _ in range(n_iterations):
            # Step 1: Generate random valid solution
            current_solution = generate_random_solution(instance, n_drones=n_drones)
            generated_total += 1
            if current_solution.get("part2"):
                generated_with_drones += 1

            # Step 2: Check feasibility
            if not feas_checker.is_solution_feasible(current_solution):
                continue

            # Step 3: Evaluate objective (total travel time)
            tot, _, _, feas_flag = travel_time_calc.calculate_total_waiting_time(current_solution)
            if not feas_flag:
                continue
            feasible_total += 1
            if current_solution.get("part2"):
                feasible_with_drones += 1
            current_obj = tot

            # Step 4: If better, update best
            if current_obj < best_run_obj:
                best_run_obj = current_obj
                best_run_solution = current_solution

        elapsed = time.time() - start_time

        # If no feasible solution was found in this run, skip stats for this run
        if best_run_solution is None:
            print(f"Run {run + 1}/{n_runs}: no feasible solution found (skipped)")
            continue

        run_best_objectives.append(best_run_obj)
        run_times.append(elapsed)

        if best_run_obj < best_overall_obj:
            best_overall_obj = best_run_obj
            best_overall_solution = best_run_solution

        print(
            f"Run {run + 1}/{n_runs}: "
            f"best objective = {best_run_obj:.3f}, "
            f"time = {elapsed:.3f} s"
        )
        # Drone diagnostics for this run
        if generated_total > 0:
            print(
                f"  Generated with drones: {generated_with_drones}/{generated_total} "
                f"({generated_with_drones / generated_total:.1%})"
            )
        if feasible_total > 0:
            print(
                f"  Feasible with drones:  {feasible_with_drones}/{feasible_total} "
                f"({feasible_with_drones / feasible_total:.1%})"
            )
        else:
            print("  Feasible solutions:     0")

    # ------------------------------------------------------------------
    # 5) Summary statistics (only if at least one feasible run)
    # ------------------------------------------------------------------
    if not run_best_objectives:
        print("\n=== Blind Random Search Summary ===")
        print(f"Instance:              {instance_path}")
        print("No feasible solutions found in any run.")
        return {
            "initial_objective": initial_obj,
            "average_objective": None,
            "best_objective": None,
            "improvement_percent": None,
            "average_runtime": None,
            "best_solution": None,
        }

    avg_obj = sum(run_best_objectives) / len(run_best_objectives)
    avg_time = sum(run_times) / len(run_times)

    if initial_obj is not None:
        improvement_percent = 100.0 * (initial_obj - best_overall_obj) / initial_obj
    else:
        improvement_percent = None

    print("\n=== Blind Random Search Summary ===")
    print(f"Instance:              {instance_path}")
    if initial_obj is not None:
        print(f"Initial objective:     {initial_obj:.3f}")
    else:
        print("Initial objective:     (initial solution infeasible)")
    print(f"Average objective:     {avg_obj:.3f}")
    print(f"Best objective:        {best_overall_obj:.3f}")
    if improvement_percent is not None:
        print(f"Improvement (%%):       {improvement_percent:.2f}")
    else:
        print("Improvement (%%):       n/a (no feasible initial solution)")
    print(f"Average runtime (s):   {avg_time:.3f}")

    # ------------------------------------------------------------------
    # 6) Save and print best solution
    # ------------------------------------------------------------------
    if best_overall_solution is not None:
        # Convert best solution dict into the required permutation string:
        # part1 | part2 | part3 | part4  (each part comma-separated)
        def _solution_to_string(sol: Dict[str, List[int]]) -> str:
            def join_part(part: List[int]) -> str:
                return ",".join(str(x) for x in part)
            return " | ".join(
                [
                    join_part(sol.get("part1", [])),
                    join_part(sol.get("part2", [])),
                    join_part(sol.get("part3", [])),
                    join_part(sol.get("part4", [])),
                ]
            )

        best_solution_str = _solution_to_string(best_overall_solution)

        print("\nBest solution (permutation format):")
        print(best_solution_str)

        # Save to a text file next to the instance file
        base_name = os.path.splitext(os.path.basename(instance_path))[0]
        out_name = f"best_solution_{base_name}.txt"
        out_path = os.path.join(os.path.dirname(instance_path), out_name)
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(best_solution_str + "\n")
            print(f"Best solution saved to: {out_path}")
        except OSError as e:
            print(f"Warning: could not save best solution to file: {e}")

    return {
        "initial_objective": initial_obj,
        "average_objective": avg_obj,
        "best_objective": best_overall_obj,
        "improvement_percent": improvement_percent,
        "average_runtime": avg_time,
        "best_solution": best_overall_solution,
    }


def resolve_instance_path(path: str) -> str:
    """
    Try to locate the instance file in common spots:
    - absolute path (as given)
    - relative to Assignment_2-B/
    - relative to repo root
    - inside repo root/Test_files/
    Returns the first existing path; otherwise the original string.
    """
    if os.path.isabs(path) and os.path.exists(path):
        return path

    candidate_curr = os.path.join(CURRENT_DIR, path)
    if os.path.exists(candidate_curr):
        return candidate_curr

    candidate_root = os.path.join(PROJECT_ROOT, path)
    if os.path.exists(candidate_root):
        return candidate_root

    # Look inside the shared Test_files directory (accept either basename or subpath)
    candidate_tests = os.path.join(PROJECT_ROOT, "Test_files", os.path.basename(path))
    if os.path.exists(candidate_tests):
        return candidate_tests

    return path


if __name__ == "__main__":
    # Default to Test_files/F_10.txt when no path is provided; otherwise use CLI args
    if len(sys.argv) < 2:
        instance_path = os.path.join("Test_files", "F_10.txt")
    else:
        instance_path = sys.argv[1]

    n_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000
    n_runs = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    instance_path = resolve_instance_path(instance_path)

    blind_random_search(instance_path, n_iterations=n_iterations, n_runs=n_runs)
