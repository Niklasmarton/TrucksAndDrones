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

truck, drone1, drone2, drone_launch, drone_land = drone_route()
launch_indices_1 = [truck.index(node) + 1 for node in drone_launch[1]]
landing_indices_1 = [truck.index(node) + 1 for node in drone_land[1]]
launch_indices_2 = [truck.index(node) + 1 for node in drone_launch[2]]
landing_indices_2 = [truck.index(node) + 1 for node in drone_land[2]]

drone_serving = drone1 + [-1] + drone2
drone_total_launches = launch_indices_1 + [-1] + launch_indices_2
drone_total_landings = landing_indices_1 + [-1] + landing_indices_2

solution = {
    "part1": truck,
    "part2": drone_serving,   # no drone customers yet
    "part3": drone_total_launches,   # no launches
    "part4": drone_total_landings,   # no landings
}

total_time, arrivals, departures, feasible = calc.calculate_total_waiting_time(solution)

print("Total time:", total_time)
print("Feasible:", feasible)

if feasible!= True:
    for i in drone_total_landings:
        if i not in truck:
            print(f"landing {i} is not in the truck")
    for i in drone_total_launches:
        if i not in truck:
            print(f"landing {i} is not in the truck")
    print(truck)
