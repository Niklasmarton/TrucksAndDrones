import numpy as np
import random

D = np.loadtxt("Truck_Drone_Contest.txt", 
               delimiter="\t",
               skiprows=5)

current = 0
ending = 0
#A list which will contain the order of visited cities
truck_route = [0]
#The search queue is a list of all nodes from 1 to 100
search_queue = [i for i in range(1, 101)]
three_nearest = []


def next_node(current_node):
	relevant_row = D[current_node]
	three_nearest = []
	target_size = min(3, len(search_queue))
	i = 0
	# collect the first `target_size` available neighbors
	while len(three_nearest) < target_size and i < len(relevant_row):
		if i == current_node or i not in search_queue:
			i += 1
			continue
		three_nearest.append((i, relevant_row[i]))
		i += 1

	# if only one node is left we must return it
	if not three_nearest:
		return None

	three_nearest.sort(key=lambda x: x[1])
	for index, distance in enumerate(relevant_row):
		if index == current_node or index not in search_queue:
			continue
		if distance < three_nearest[-1][1]:
			three_nearest.append((index, distance))
			three_nearest.sort(key=lambda x: x[1])
			three_nearest.pop()
	random_index = random.randint(0, len(three_nearest) - 1)
	return three_nearest[random_index][0]

def final_route():
	current_node = 0
	while search_queue:
		chosen = next_node(current_node)
		if chosen is None:
			break
		truck_route.append(chosen)
		search_queue.remove(chosen)
		current_node = chosen
	truck_route.append(0)
	return truck_route

print(final_route())

        







