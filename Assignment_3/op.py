from read_file import read_instance
import copy
import random

absolute_path = "/Users/niklasmarton/Library/CloudStorage/OneDrive-Personlig/ITØK/Metaheuristics/TrucksAndDrones/Test_files/"
file_name = "F_10.txt"

instance = read_instance(f"{absolute_path}{file_name}")

# Unpack the fields you actually need from the returned dict.
T = instance["truck_times"]
D = instance["drone_times"]
flight_limit = instance["flight_limit"]
n_customers = instance["n_customers"]
depot = instance.get("depot_index", 0)


def choose_launch_landing_greedy(drone_node, truck_route, truck_index):
    relevant_row = T[drone_node]
    index_in_truck = truck_route.index(drone_node)
    left_max = min(len(truck_route[:index_in_truck]), 5)
    right_max = min(len(truck_route[:index_in_truck]), 5)






def operator(current_solution):
    new_solution = copy.deepcopy(current_solution)
    removal_choice = random.randint(0,2)
    insert_choice = random.randint(0,2)
    print(removal_choice)
    print(insert_choice)
    
    #If we choose from the truck route
    if removal_choice == 0 and len(new_solution[removal_choice]) > 2:
        removal_index = random.randint(1, len(new_solution[0]) - 2)
        value_insertion = new_solution[removal_choice].pop(removal_index)
    
    #If we choose from the drone route
    else:
        if len(new_solution[removal_choice]) == 0:
            return new_solution
        removal_index = random.randint(0, len(new_solution[removal_choice])-1)
        
        value_insertion = new_solution[removal_choice].pop(removal_index)
        


    #If we choose to add into the truck route
    if insert_choice == 0:
        min_idx = 1
        max_idx = len(new_solution[insert_choice]) - 2
        if min_idx > max_idx:
            return new_solution
        if removal_choice == 0:
            candidate_indices = [i for i in range(min_idx, max_idx + 1) if i != removal_index]
            if candidate_indices:
                insertion_index = random.choice(candidate_indices)
            else:
                insertion_index = removal_index
        else:
            insertion_index = random.randint(min_idx, max_idx)
        new_solution[insert_choice].insert(insertion_index, value_insertion)

    #If we to add into the drone route
    else:
        if len(new_solution[insert_choice]) == 0:
            new_solution[insert_choice].append(value_insertion)
            return new_solution
        insertion_index = random.randint(0, len(new_solution[insert_choice]))
        new_solution[insert_choice].insert(insertion_index, value_insertion)
    
    return new_solution

    









