import numpy as np
import random

# Load distance matrix: row/col 0 is the hub.
D = np.loadtxt("Truck_Drone_Contest.txt", delimiter="\t", skiprows=5)
NUM_NODES = D.shape[1]  # columns correspond to nodes (hub + 100)

current = 0
ending = 0
# Order of visited cities
truck_route = [0]
# Remaining nodes (only real node columns, exclude hub)
search_queue = [i for i in range(1, NUM_NODES)]


def next_node(current_node):
	relevant_row = D[current_node]
	hub_row = D[0]

	# Weight between current-node distance and hub distance; closer to hub when few nodes remain.
	remaining = len(search_queue)
	total = NUM_NODES - 1
	frac = remaining / total if total else 1
	alpha = 0.7 + 0.2 * frac

	target_size = min(3, len(search_queue))
	candidates = []

	for index in search_queue:
		if index == current_node:
			continue
		distance = relevant_row[index]
		hub_distance = hub_row[index]
		score = alpha * distance + (1 - alpha) * hub_distance
		candidates.append((index, score))

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

def map_launching_and_landing(prev_node, next_node, drone_number):
	drone_launching[drone_number].append(prev_node)
	drone_landing[drone_number].append(next_node)
	return drone_launching, drone_landing


def drone_route() -> list[int]:
	truck = truck_only_route()
	drone1 = []
	drone2 = []
	prob_first_node = 0.3
	prob_second_node = 0.2
	drone_flight_limit = 5500
	i = 1
	while i < len(truck_route) - 1:
		prev_node = truck_route[i - 1]
		curr_node = truck_route[i]
		next_node = truck_route[i + 1]
		drone1_bool = False
		prev_distance = D[prev_node][curr_node]
		next_distance = D[curr_node][next_node]
		# Simple check: total flight for outbound+return must fit limit
		if prev_distance + next_distance <= drone_flight_limit:
			if random.random() < prob_first_node:
				drone1_bool = True
				drone1.append(i)
				drone_launch, drone_land = map_launching_and_landing(prev_node, next_node, 1)
				i += 1
				truck.remove(i)
			#This is for drone 2
			if random.random() < prob_first_node and drone1_bool == False:
				drone2.append(i)
				drone_launch, drone_land = map_launching_and_landing(prev_node, next_node, 2)
				i += 1
				truck.remove(i)
		if i >= 2 and random.random() < prob_second_node:
			prev_node = truck_route[i - 2]
			prev_distance = D[prev_node][curr_node]
			next_distance = D[curr_node][next_node]
			if prev_distance + next_distance <= drone_flight_limit:
				if random.random() < prob_first_node:
					drone1_bool = True
					drone1.append(i)
					drone_launch, drone_land = map_launching_and_landing(prev_node, next_node, 1)
					i += 1
					truck.remove(i)
				if random.random() < prob_first_node and drone1_bool == False:
					drone2.append(i)
					i += 1
					drone_launch, drone_land = map_launching_and_landing(prev_node, next_node, 2)
					truck.remove(i)	
				
		i += 1
	return truck, drone1, drone2, drone_launch, drone_land
		

truck, drone1, drone2, drone_launch, drone_land = drone_route()
print(f"Truck route is: {truck}")
print(f"Drone1 visits: {drone1}")
print(f"Drone2 visits: {drone2}")
print(f"launching sites: {drone_launch}")
print(f"launching sites: {drone_land}")

def full_format_solution():
	print(f"{truck}|{drone1}-1{drone2}|{drone_launch[1]}-1{drone_launch[2]}|{drone_land[1]}-1{drone_land[2]}")



