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
	hub_row = D[0]
	three_nearest = []
	# collect the first `target_size` available neighbors
	# while len(three_nearest) < target_size and i < len(relevant_row):
	# 	if i == current_node or i not in search_queue:
	# 		i += 1
	# 		continue
	# 	three_nearest.append((i, relevant_row[i]))
	# 	i += 1

	# if only one node is left we must return it
	# if not three_nearest:
	# 	return None

	#We want to weigh the distance to the hub increasingly in strength as we search more nodes
	#Want to establish an alpha, which is how much we weight the distance to the hub
	three_nearest.sort(key=lambda x: x[1])
	remaining = len(search_queue)
	total = 100
	frac = remaining / total
	alpha = 0.7 + 0.2 * frac
	nearest_to_node = []
	target_size = min(3, len(search_queue))

	for index, distance in enumerate(relevant_row):
		if index == current_node or index not in search_queue:
			continue
		#Here we have the distances to the remaining nodes
		nearest_to_node.append((index, distance))
	
	#Create another list of those elements with the new combines heuristic
	three_best_scores = []
	if current_node != 0:
		#This iterates through the distances to the hub node
		for (index, distance) in nearest_to_node:
			for (index_h, distance_h) in hub_row:
				if index == index_h:
					score = distance * alpha + distance_h * (1-alpha)
					#This then is the list with the updated heuristics
					while len(three_best_scores) < target_size:
						three_best_scores.append((index, score))
					if score < three_best_scores[-1][1]:
						three_best_scores.append((index, distance))
						three_best_scores.sort(key=lambda x: x[1])
						three_best_scores.pop()
	
	if not three_best_scores:
		return None
					

	random_index = random.randint(0, len(three_nearest) - 1)
	next_node_num = three_best_scores[random_index][0]
	return next_node_num

def final_route():
	current_node = 0
	while len(search_queue) > 10:
		chosen = next_node(current_node)
		if chosen is None:
			break
		truck_route.append(chosen)
		search_queue.remove(chosen)
		current_node = chosen
	truck_route.append(0)

	#Run the local search to add and remove one branch
	return truck_route

print(final_route())



        






