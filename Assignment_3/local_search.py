from read_file import read_instance
from op import operator

absolute_path = "/Users/niklasmarton/Library/CloudStorage/OneDrive-Personlig/ITØK/Metaheuristics/TrucksAndDrones/Test_files/"
file_name = "F_100.txt"

instance = read_instance(f"{absolute_path}{file_name}")

# Unpack the fields you actually need from the returned dict.
T = instance["truck_times"]
D = instance["drone_times"]
flight_limit = instance["flight_limit"]
n_customers = instance["n_customers"]
depot = instance.get("depot_index", 0)

truck_route = [i for i in range(n_customers + 1)] + [0]
drone1 = []
drone2 = []
launch1 = []
launch2 = []
land1 = []
land2 = []

initial_solution = [truck_route, drone1, drone2, launch1, launch2, land1, land2]

print(f"Old solution was: {initial_solution}")

def run_operator_iterations(start_solution, iterations=1000):
    current_solution = start_solution
    for _ in range(iterations):
        current_solution = operator(current_solution)
    return current_solution


final_solution = run_operator_iterations(initial_solution, iterations=1000)
print(f"Final solution after 1000 iterations: {final_solution}")




