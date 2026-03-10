from pathlib import Path
import sys

# Ensure this file can import the weighted SA module when run directly.
ALG_DIR = Path(__file__).resolve().parent
if str(ALG_DIR) not in sys.path:
    sys.path.append(str(ALG_DIR))

import simulated_annealing_3op_weighted as sa_weighted


def build_index_order_truck_only_initial_solution(instance_data):
    n_customers = instance_data["n_customers"]
    truck_route = [i for i in range(n_customers + 1)] + [0]
    return [truck_route, [], []]


def main():
    instance_data = sa_weighted.load_instance()
    initial_solution = build_index_order_truck_only_initial_solution(instance_data)

    configs = [
        ("Balanced", {"op1": 0.15, "op2": 0.50, "op3": 0.35}),
        ("Truck-heavy", {"op1": 0.10, "op2": 0.55, "op3": 0.35}),
        ("2opt-heavy", {"op1": 0.10, "op2": 0.40, "op3": 0.50}),
    ]

    results = []
    for name, weights in configs:
        print(f"\n=== {name} ===")
        _, best_cost = sa_weighted.run_statistics(
            initial_solution,
            instance_data=instance_data,
            runs=10,
            warmup_iterations=100,
            iterations=9900,
            final_temperature=0.1,
            cache_limit=200000,
            plot_best_after_all=True,
            operator_weights=weights,
            verbose=True,
        )
        results.append((name, best_cost))

    print("\n=== Comparison (lower is better) ===")
    for name, best_cost in results:
        print(f"{name}: best={best_cost}")


if __name__ == "__main__":
    main()
