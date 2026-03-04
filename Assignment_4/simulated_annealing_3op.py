"""
Solution representation:
My solution representation has changed a bit, and is on the form:
solution: [(truck_visited), (drone1), (drone2)]
where drone1 and drone2 are arrays of tuples of visisted nodes on the form:
(visited_node, launch_idx, land_idx)
This was done to make it easier to create new feasible drone routes with the operators

Operator 1: Truck reinsert
This operator chooses one among 5 of the worst edges in the current solution, takes the leaving node and reinserts it where the travel time is the least for that edge.
By removing the worst edge, we should hopefully produce a better truck route. By choosing among the five worstedges, we introduce a bit of randomness to the operator, letting it explore more. 
"""