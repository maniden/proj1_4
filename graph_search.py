import dis
from heapq import heappush, heappop
from hmac import new
from math import dist  # Recommended.
import numpy as np

from flightsim.world import World

from .occupancy_map import OccupancyMap # Recommended.

def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        return a tuple (path, nodes_expanded)
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
        nodes_expanded, the number of nodes that have been expanded

    """

    # While not required, we have provided an occupancy map you may use or modify.
    occ_map = OccupancyMap(world, resolution, margin)

    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))

    # Check that the start and goal nodes are free of obstacles and within the map
    # if occ_map.is_occupied_index(start_index) or occ_map.is_occupied_index(goal_index):
    #     print("Occupied")
    #     return None, 0
    path = None
    nodes_expanded = 0
    
    if not astar:
        # Dijkstra's algorithm
        # Initialize the distance and parent arrays
        distance = np.inf*np.ones(occ_map.map.shape)
        distance[start_index] = 0
        parents = np.full(occ_map.map.shape, None, dtype=object)
        Q = np.zeros(occ_map.map.shape)
        index = np.full(occ_map.map.shape, None, dtype=object)
        # for i in range(occ_map.map.shape[0]):
        #     for j in range(occ_map.map.shape[1]):
        #         for k in range(occ_map.map.shape[2]):
        #             index[i, j, k] = (i, j, k)
        sum_max = occ_map.map.shape[0]*occ_map.map.shape[1]*occ_map.map.shape[2]

        # Main loop
        
        idx_arr = [
                (-1,-1,-1),(-1,-1,0),(-1,-1,1),
                (-1,0,-1),(-1,0,0),(-1,0,1),
                (-1,1,-1),(-1,1,0),(-1,1,1),
                (0,-1,-1),(0,-1,0),(0,-1,1),
                (0,0,-1),(0,0,1),
                (0,1,0),(0,1,1),(0,1,-1),
                (1,-1,0),(1,-1,1),(1,-1,-1),
                (1,0,1),(1,0,-1),(1,0,0),
                (1,1,1),(1,1,-1),(1,1,0)]
        cnt = 0
        while np.any(Q > 0):
            # Find the node with the smallest distance
            # reduced_index = index[Q<1]
            # current_index = reduced_index[np.unravel_index(np.argmin(distance[Q < 1]), reduced_index.shape)]
            current_index = np.unravel_index(np.argmin(distance + 1e5*Q), distance.shape)
            # If the goal is reached, break

            if current_index == goal_index:
                print("Goal")
                break

            # If the distance is infinity, no path exists
            if Q[current_index] == 1:
                print("Inf")
                return None, 0

            Q[current_index] = 1
            
            # Get the neighbors of the current node
            neighbors = []
            current_distance = distance[current_index]
            
            # for i in range(-1, 2):
            #     for j in range(-1, 2):
            #         for k in range(-1, 2):
            #             if i == 0 and j == 0 and k == 0:
            #                 continue
            for idx in idx_arr:
                        i, j, k = idx[0], idx[1], idx[2]
                        neighbor = (current_index[0] + i, current_index[1] + j, current_index[2] + k)
                        if not occ_map.is_occupied_index(neighbor) and Q[neighbor] == 0:
                            neighbors.append([neighbor, (i+1,j+1,k+1)])


            # print("Neighbors", neighbors)
            # cnt_1 = 0
            # Update neighbor distances

            # current_distance = distance[current_index]
            for neighbor in neighbors:
                # if Q[neighbor[0]] == 0:
                    # Calculate the new distance
                    new_distance = current_distance + occ_map.neighbor_dist[neighbor[1]]
                    if new_distance < distance[neighbor[0]]:
                        distance[neighbor[0]] = new_distance
                        parents[neighbor[0]] = current_index
            # Q_n = Q[neighbors]
            # distance_n = distance[neighbors]
            # new_distance = distance[current_index] + np.array([occ_map.euclidean_distance(current_index, neighbor) for neighbor in neighbors])
            # cnt += 1

            # # If the number of iterations is too large, return None
            # if cnt > 1000000:
            #     print("Timeout")
            #     return None, 0

        # Reconstruct the path
        if distance[goal_index] == np.inf:
            print("Inf")
            return None, 0
        path = []
        current_index = goal_index
        path.append(goal)
        while current_index != start_index:
            path.append(occ_map.index_to_metric_center(current_index))
            current_index = parents[current_index]
        path.append(start)
        path.reverse()
        path = np.array(path)
        nodes_expanded = np.sum(Q)

        return path, nodes_expanded

    else:
        # A* algorithm
        distance = np.inf*np.ones(occ_map.map.shape)
        f_score = np.inf*np.ones(occ_map.map.shape)
        distance[start_index] = 0
        f_score[start_index] = occ_map.euclidean_distance(start_index, goal_index)
        parents = np.full(occ_map.map.shape, None, dtype=object)
        Q = np.zeros(occ_map.map.shape)
        index = np.full(occ_map.map.shape, None, dtype=object)
        for i in range(occ_map.map.shape[0]):
            for j in range(occ_map.map.shape[1]):
                for k in range(occ_map.map.shape[2]):
                    index[i, j, k] = (i, j, k)
        Q_sum = 0
                    

        Q[start_index] = 1
        Q_sum += 1
        

        cnt = 0
        # Main loop
        while Q_sum > 0:
            # Find the node with the smallest distance
            # reduced_index = index[Q>0]
            # current_index = reduced_index[np.unravel_index(np.argmin(f_score[Q > 0]), reduced_index.shape)]
            current_index = np.unravel_index(np.argmin(f_score - 1e5*Q), f_score.shape)
            # If the goal is reached, break
            if current_index == goal_index:
                print("Goal")
                break
            # If the distance is infinity, no path exists

            if distance[current_index] == np.inf:
                print("Inf")
                return None, 0
            Q[current_index] = 0
            Q_sum -= 1

            # Get the neighbors of the current node
            neighbors = []
            current_distance = distance[current_index]
            for i in range(-1, 2):
                for j in range(-1, 2):
                    for k in range(-1, 2):
                        if i == 0 and j == 0 and k == 0:
                            continue
                        neighbor = (current_index[0] + i, current_index[1] + j, current_index[2] + k)
                        if not occ_map.is_occupied_index(neighbor) and Q[neighbor] == 0:
                            neighbors.append([neighbor, (i+1,j+1,k+1)])
                            

            
            # Update neighbor distances
            for neighbor in neighbors:
                # if Q[neighbor[0]] == 0:
                    # Calculate the new distance
                    new_distance = current_distance + occ_map.neighbor_dist[neighbor[1]]
                    if new_distance < distance[neighbor[0]]:
                        distance[neighbor[0]] = new_distance
                        parents[neighbor[0]] = current_index
                        f_score[neighbor[0]] = new_distance + occ_map.euclidean_distance(neighbor[0], goal_index)
                        Q[neighbor[0]] = 1
                        Q_sum += 1
            cnt += 1

            # If the number of iterations is too large, return None
            if cnt > 1000000:
                print("Timeout")
                return None, 0


        # Reconstruct the path
        if distance[goal_index] == np.inf:
            return None, 0

        path = []
        current_index = goal_index
        path.append(goal)
        while current_index != start_index:
            path.append(occ_map.index_to_metric_center(current_index))
            current_index = parents[current_index]
        path.append(start)
        path.reverse()
        path = np.array(path)
        nodes_expanded = np.sum(Q)
        return path, nodes_expanded

    # Return a tuple (path, nodes_expanded)

    return None, 0



