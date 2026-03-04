import random
from read_file import read_instance

absolute_path = "/Users/niklasmarton/Library/CloudStorage/OneDrive-Personlig/ITØK/Metaheuristics/TrucksAndDrones/Test_files/"
file_name = "R_20.txt"

instance = read_instance(f"{absolute_path}{file_name}")
# Operator context must be set by the algorithm that uses the operator
# (e.g., local_search.py / simulated_annealing.py).

T = instance["truck_times"]
D = instance["drone_times"]
flight_limit = instance["flight_limit"]
n_customers = instance["n_customers"]
depot = instance.get("depot_index", 0)


def _clone_solution(solution):
    # Tuples inside drone routes are immutable, so shallow-copying route lists is sufficient.
    return [solution[0][:], solution[1][:], solution[2][:]]

def calculate_worst_index(solution) -> int: #Returns the index of the node we want to reinsert
    k = 5
    truck = solution[0]
    worst = []

    for i in range(1, len(truck)-2):
        curr_node = truck[i]
        leaving_node = truck[i-1]
        entering_node = truck[i+1]
        cost = T[leaving_node][curr_node] + T[curr_node][entering_node] - T[leaving_node][entering_node]
        
        if len(worst) < k:
            worst.append((i, cost))

        else:
            min_pos = min(range(k), key=lambda p: worst[p][1])
            if cost > worst[min_pos][1]:
                worst[min_pos] = (i, cost)

    random_choice = random.randrange(len(worst))
    edge_i = worst[random_choice][0]
    node_removed = solution[edge_i]
    return edge_i, node_removed

def calculate_best_index(solution, node_to_insert) -> int: #Returns the index to reinsert the node
    k = 5
    truck = solution[0]
    best = []

    for i in range(1, len(truck)-1):
        leaving_node = truck[i]
        entering_node = truck[i+1]
        cost = T[leaving_node][node_to_insert] + T[node_to_insert][entering_node] - T[leaving_node][entering_node]
        
        if len(best) < k:
            best.append((i, cost))

        else:
            max_pos = max(range(k), key=lambda p: best[p][1])
            if cost > best[max_pos][1]:
                best[max_pos] = (i, cost)

    random_choice = random.randrange(len(best))
    edge_i = best[random_choice][0]
    return edge_i

def truck_reinsert(solution):
    solution = _clone_solution(solution)

    removal_index, removed_node = calculate_worst_index(solution)
    insertion_index = calculate_best_index(solution, removed_node)

    """
    For every drone_route in drone1, drone2:
        if removal index in drone1 or drone2:
            get drone_pair from drone1 or drone2
            if checkdronefeasibility(removal_index, insertion_index, drone_pair) == True:
                pop the removed node from the solution
                insert the node into the insertion index
                return solution
            else:
                pop the removed node from the solution
                insert the node into the insertion index
                build_new_drone_pair()
                return solution
        else:
        pop the removed node from the solution
        insert the node into the insertion index
        return solution
                
    """

    removed_node = solution[0].pop(removal_index)
    solution[0].insert(insertion_index, removed_node)

    return solution

"""
def checkdronefeasibility(removal_index, drone_pair, drone1 or drone2) -> Bool:
    if removal_index == drone_pair launch:
        if insertion_index <= drone_pair land:
            return False
    if removal_index == drone_pair land:
        if insertion_index >= drone pair launch:
            return False
    previous_index = getindex_of_drone_pair
    order copy of drone route by launch
    get new_index of the drone_pair
    if new_index != previous_index:
        return False
    order copy of drone route by landing
    if new_index != previous_index:
        return False
    return True
    
"""