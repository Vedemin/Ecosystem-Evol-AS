import numpy as np

# def RayCast(env, a_x, a_y, a_d, b_x, b_y, b_d, sight_distance):
#     """
#     Optimized raycasting function to check LOS (line of sight) between two points.
#     Focuses only on a fragment of the map and evaluates potential obstructions selectively.

#     Args:
#         env (2D np.array): The terrain height map.
#         a_x, a_y, a_d (float): Coordinates and height of point A (observer).
#         b_x, b_y, b_d (float): Coordinates and height of point B (target).
#         sight_distance (float): Maximum visible distance.

#     Returns:
#         (bool, float): A tuple where:
#                        - bool is True if LOS is clear, False if blocked.
#                        - float is the distance between A and B.
#     """
#     # Calculate the distance between the two points
#     dist = GetDistance(a_x, a_y, a_d, b_x, b_y, b_d)
    
#     # If the target is beyond the sight distance, return False
#     if dist > sight_distance:
#         return False, dist
    
#     # If the points are effectively the same, return True
#     if abs(dist) < 0.001:
#         return True, dist

#     # Determine the bounding box of the relevant fragment
#     x_min, x_max = int(min(a_x, b_x)), int(max(a_x, b_x)) + 1
#     y_min, y_max = int(min(a_y, b_y)), int(max(a_y, b_y)) + 1

#     # Extract the fragment of the terrain map
#     fragment = env.map[y_min:y_max, x_min:x_max]

#     # Get the minimum height along the ray
#     lower_point = min(a_d, b_d)

#     # Quickly check if any part of the terrain is above the lower point
#     if np.max(fragment) <= lower_point:
#         return True, dist  # No obstruction possible

#     # Identify terrain points above the ray's lower depth
#     potential_obstacles = np.argwhere(fragment > lower_point)
#     if len(potential_obstacles) == 0:
#         return True, dist  # No blocking terrain
    
#     # Shift potential obstacle coordinates to the global map
#     potential_obstacles += [y_min, x_min]

#     # Check only terrain points above the ray's depth
#     for obs_y, obs_x in potential_obstacles:
#         # Get the normalized position along the ray
#         t = ((obs_x - a_x) / (b_x - a_x)) if abs(b_x - a_x) > abs(b_y - a_y) else ((obs_y - a_y) / (b_y - a_y))

#         if 0 <= t <= 1:  # Only consider points between A and B
#             interpolated_height = a_d + t * (b_d - a_d)
#             if env.map[obs_y, obs_x] > interpolated_height:
#                 return False, dist  # LOS is blocked

#     return True, dist  # LOS is clear


def RayCast(env, a_x, a_y, a_d, b_x, b_y, b_d, sight_distance):
    dist = GetDistance(a_x, a_y, a_d, b_x, b_y, b_d)
    if dist > sight_distance:
        return False, dist
    if abs(dist) < 0.001:
        return True, dist
    vec = [b_x - a_x, b_y - a_y, b_d - a_d]
    mini = [vec[0] / dist, vec[1] / dist, vec[2] / dist]
    c_x = a_x
    c_y = a_y
    c_d = a_d

    for i in range(round(dist) + 1):
        if env.map[round(c_x)][round(c_y)] < c_d:
            if round(c_x) == b_x and round(c_y) == b_y:
                return True, dist
            else:
                return False, dist
        c_x += mini[0]
        c_y += mini[1]
        c_d += mini[2]

        if b_x < 0:
            if c_x <= b_x:
                return True, dist
        else:
            if c_x >= b_x:
                return True, dist
        if b_y < 0:
            if c_y <= b_y:
                return True, dist
        else:
            if c_y >= b_y:
                return True, dist

    return True, dist

def GetDistance(a_x, a_y, a_d, b_x, b_y, b_d):
    return np.sqrt((b_x - a_x) ** 2 + (b_y - a_y) ** 2 + (b_d - a_d) ** 2)