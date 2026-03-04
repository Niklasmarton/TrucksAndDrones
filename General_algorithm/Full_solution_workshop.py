"""
End-to-end runner for the truck–drone contest instance using
Construction + Simulated Annealing + arrival-time evaluator.

Uses the new instance file Truck_Drone_Contest_new.txt implicitly via
Construction.T / Construction.D.
"""

import random

from Construction import drone_route, T, D
from CalCulateTotalArrivalTime import CalCulateTotalArrivalTime
from Simulated_Annealing import simulated_annealing, to_solution_dict


def build_calculator():
    calc = CalCulateTotalArrivalTime()
    calc.truck_times = T
    calc.drone_times = D
    calc.flight_range = 5500  # range read from instance header (unchanged)
    calc.depot_index = 0
    return calc


def main():
    random.seed(42)

    calc = build_calculator()

    start_solution = drone_route()

    best_solution, best_cost = simulated_annealing(
        calc,
        start_solution=start_solution,
        T0=8000,       # good starting temperature for this instance
        alpha=0.995,
        iter_per_T=200,
        Tmin=50,
    )

    best_dict = to_solution_dict(best_solution)
    print("Best total time (min):", best_cost)
    print("Truck route:", best_dict["part1"])
    print("Drone customers:", best_dict["part2"])
    print("Launch indices:", best_dict["part3"])
    print("Landing indices:", best_dict["part4"])


if __name__ == "__main__":
    main()
