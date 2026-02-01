import numpy as np
import random

# Load distance matrix: row/col 0 is the hub.
T = np.loadtxt("Truck_Drone_Contest.txt", delimiter="\t", skiprows=5, max_rows=101)
NUM_NODES = T.shape[1]  # columns correspond to nodes (hub + 100)
D = np.loadtxt("Truck_Drone_Contest.txt", delimiter="\t", skiprows=107, max_rows=101)

current = 0
ending = 0
# Order of visited cities
truck_route = [0]
# Remaining nodes (only real node columns, exclude hub)
search_queue = [i for i in range(1, NUM_NODES)]


def next_node(current_node):
	relevant_row = T[current_node]

	# pure nearest-neighbor with random tie-break among the 3 closest unseen nodes
	target_size = 1
	candidates = []

	for index in search_queue:
		if index == current_node:
			continue
		distance = relevant_row[index]
		candidates.append((index, distance))

	if not candidates:
		return None

	candidates.sort(key=lambda x: x[1])
	top_choices = candidates[:target_size]
	return random.choice(top_choices)[0]


def truck_only_route() -> list[int]:
	current_node = 0
	while len(search_queue) > 0:
		chosen = next_node(current_node)
		if chosen is None:
			break
		truck_route.append(chosen)
		search_queue.remove(chosen)
		current_node = chosen
	truck_route.append(0)
	return truck_route


#Define two dictionaries for the launching and landing
drone_launching = {1: [], 2: []}
drone_landing = {1: [], 2: []}

def map_launching_and_landing(prev_node, next_node, drone_number, drone_launch, drone_land):
	drone_launch[drone_number].append(prev_node)
	drone_land[drone_number].append(next_node)
	return drone_launch, drone_land


def drone_route() -> list[int]:
	# reset global route state for each run
	global truck_route, search_queue
	truck_route = [0]
	search_queue = [i for i in range(1, NUM_NODES)]

	truck = truck_only_route()
	drone1 = []
	drone2 = []
	prob_first_node = 0.9
	prob_second_node = 0.5
	drone_flight_limit = 5500
	i = 1
	drone_busy = {"drone1": False, "drone2": False}
	drone_launch = {1: [], 2: []}
	drone_land = {1: [], 2: []}
	protected_nodes = set()  # nodes that must remain in truck route as landing points

	while i < len(truck) - 1:
		prev_node = truck[i - 1]
		curr_node = truck[i]
		next_node = truck[i + 1]
		flight_out = D[prev_node][curr_node]
		flight_back = D[curr_node][next_node]
		truck_leg = T[prev_node][next_node]


		##This is the code that allows the drone to fly one city and arrive at the next one

		#This ensures that the flight time restriction is valid
		# Landing nodes must remain; when reached, free the drone and move on
		if curr_node in protected_nodes:
			protected_nodes.remove(curr_node)
			drone_busy["drone1"] = False
			i += 1
			continue
		total_flight = flight_out + flight_back
		#Ensure feasible: evaluator flags if either drone flight or truck leg exceeds range
		if max(total_flight, truck_leg) <= drone_flight_limit:
			if drone_busy["drone1"] is False:
				if random.random() < prob_first_node:
					drone_busy["drone1"] = True
					drone1.append(curr_node)
					drone_launch, drone_land = map_launching_and_landing(prev_node, next_node, 1, drone_launch, drone_land)
					protected_nodes.add(next_node)
					truck.pop(i)
					continue
			if drone_busy["drone2"] is False:
				if random.random() < prob_second_node:
					drone_busy["drone2"] = True
					drone2.append(curr_node)
					drone_launch, drone_land = map_launching_and_landing(prev_node, next_node, 2, drone_launch, drone_land)
					protected_nodes.add(next_node)
					truck.pop(i)
					continue
		i += 1
	return truck, drone1, drone2, drone_launch, drone_land



if __name__ == "__main__":
	truck, drone1, drone2, drone_launch, drone_land = drone_route()
	launch_indices_1 = [truck.index(node) + 1 for node in drone_launch[1]]
	landing_indices_1 = [truck.index(node) + 1 for node in drone_land[1]]
	launch_indices_2 = [truck.index(node) + 1 for node in drone_launch[2]]
	landing_indices_2 = [truck.index(node) + 1 for node in drone_land[2]]
	print(f"Truck route is: {truck}")
	print(f"Drone1 visits: {drone1}")
	print(f"Drone2 visits: {drone2}")
	print(f"launching sites: {launch_indices_1, -1, landing_indices_2}")
	print(f"landing sites: {landing_indices_1, -1, landing_indices_2}")
	print(f"Length of truck route is: {len(truck)}")

	def print_result():
		print(f"{truck}|{drone1}-1{drone2}|{drone_launch[1]}-1{drone_launch[2]}|{drone_land[1]}-1{drone_land[2]}")
	
