## Simulated Annealing Setup

The simulated annealing algorithm uses **500 warm-up iterations**.  
Through testing, this gave stable temperature estimates and consistently good solution quality.

## Operators Used

For this assignment, I implemented three operators:

1. **Truck-Drone-Reinsert**
2. **Destroy-Repair**
3. **Or-Opt**

Detailed implementation notes and design choices are documented in the corresponding files in the `operators/` folder.

## Operator Roles in the Search

### 1. Truck-Drone-Reinsert (`op1`)

- Small, flexible reassignment move.
- Removes one customer from either truck or drone and reinserts it into truck or drone.
- Main role: fine-grained truck-drone synchronization and workload balancing.

### 2. Destroy-Repair (`op2`)

- Medium/large move to shake up the solution
- Removes a small set of costly truck customers and repairs by reinserting into truck or drone routes.
- Main role: escape poor local structures and open new promising neighborhoods.

### 3. Or-Opt (`op3`)

- Truck-route intensification move.
- Relocates short truck segments to better positions, then remaps/repairs drone routes.
- Main role: improve truck backbone order and remove poor edge patterns while preserving feasibility.

### Comment about the building of drone pairs in the drone_route_utils.py

- Pair construction strategy:
- Reuse preferred_pair first when valid (stability across neighbor moves).
- Search promising truck indices near the node (cheap truck proximity heuristic).
- Keep only timeline-feasible, flight-feasible pairs.
- Score by synchronization quality (waiting/mismatch penalties) and pick best,
  with small top-k randomness for diversification.
- If nearest neighborhood fails, sample a wider set as fallback.
