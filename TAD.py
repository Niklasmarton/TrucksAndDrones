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


def final_route():
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


print(final_route())


