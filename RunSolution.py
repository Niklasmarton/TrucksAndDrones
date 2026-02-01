from CalCulateTotalArrivalTime import CalCulateTotalArrivalTime
from TAD import *
import numpy as np

INSTANCE_FILE = "Truck_Drone_Contest.txt"

# --- Read flight range ---
with open(INSTANCE_FILE, "r") as f:
    lines = f.readlines()

n_customers = int(lines[1].strip())          # 100
flight_range = float(lines[3].strip())       # 5500
n_nodes = n_customers + 1                    # 0..100 => 101 nodes

# --- Read truck travel time matrix (101x101) ---
truck_times = np.loadtxt(
    INSTANCE_FILE,
    delimiter="\t",
    skiprows=5,      # skip header lines + "truck matrix" label
    max_rows=n_nodes
)

drone_times = np.loadtxt(
    INSTANCE_FILE,
    delimiter="\t",
    skiprows=107,    # 5 + 101 + 1 = 107
    max_rows=n_nodes
)

calc = CalCulateTotalArrivalTime()

calc.truck_times = truck_times
calc.drone_times = drone_times
calc.flight_range = flight_range
calc.depot_index = 0  # depot is node 0 in your routes

truck_route = final_route()

solution = {
    "part1": truck_route,
    "part2": [-1],   # no drone customers yet
    "part3": [-1],   # no launches
    "part4": [-1],   # no landings
}

total_time, arrivals, departures, feasible = calc.calculate_total_waiting_time(solution)

print("Total time:", total_time)
print("Feasible:", feasible)