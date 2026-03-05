from pathlib import Path
import sys

# Ensure this file can import the weighted SA module when run directly.
ALG_DIR = Path(__file__).resolve().parent
if str(ALG_DIR) not in sys.path:
    sys.path.append(str(ALG_DIR))

import simulated_annealing_3op_weighted as sa_weighted


def build_construction_initial_solution(instance_data):
    """
    Construction heuristic for the initial solution:
    - Greedy truck route
    - Seeded feasible drone assignments
    """
    return sa_weighted.build_greedy_initial_solution(
        instance_data,
        target_drone_ratio=0.18,
    )


def main():
    instance_data = sa_weighted.load_instance()
    initial_solution = build_construction_initial_solution(instance_data)

    # Same algorithm and settings as simulated_annealing_3op_weighted.py
    operator_weights = {"op1": 0.15, "op2": 0.6, "op3": 0.25}
    sa_weighted.run_statistics(
        initial_solution,
        instance_data=instance_data,
        runs=10,
        warmup_iterations=100,
        iterations=9900,
        final_temperature=0.1,
        cache_limit=200000,
        plot_best_after_all=True,
        operator_weights=operator_weights,
        verbose=True,
    )


if __name__ == "__main__":
    main()
