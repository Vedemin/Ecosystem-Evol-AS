import numpy as np
import colorsys
import math
from numba import njit

def getChunksIntersected(a_x, a_y, b_x, b_y, chunk_size):
    """
    Calculates the chunks intersected by the line connecting two points (a_x, a_y) and (b_x, b_y) on a grid.

    Parameters:
    - a_x, a_y: Coordinates of the starting point of the line.
    - b_x, b_y: Coordinates of the ending point of the line.
    - chunk_size: The size of the chunks (grid cells) to consider.

    Returns:
    - List of tuples: The list of chunk coordinates intersected by the line.
    """
    chunks = []
    x0, y0 = int(a_x // chunk_size), int(a_y // chunk_size)
    x1, y1 = int(b_x // chunk_size), int(b_y // chunk_size)
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        chunks.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return chunks

# @njit
def RayCast(env, a_x, a_y, a_d, b_x, b_y, b_d, sight_distance):
    """
    Simulates a ray cast between two points in a 3D space to determine visibility and distance, considering obstacles.

    Parameters:
    - env: The simulation environment, which includes map and depth data.
    - a_x, a_y, a_d: Starting point coordinates (x, y, depth) of the ray.
    - b_x, b_y, b_d: Ending point coordinates (x, y, depth) of the ray.
    - sight_distance: The maximum distance the ray can travel.

    Returns:
    - Tuple:
    - Boolean: Whether the endpoint is visible from the start point.
    - Float: The distance between the two points.
    """
    if abs(a_x - b_x) + abs(a_y - b_y) + abs(a_d - b_d) > sight_distance * math.sqrt(3):
        return False, 10000

    dist = GetDistance(a_x, a_y, a_d, b_x, b_y, b_d)
    if dist > sight_distance:
        return False, dist
    if dist < 0.001:
        return True, dist
    if env.min_depth > max(a_d, b_d):
        return True, dist

    chunk_size = env.mapsize[0] // env.depth_partition_precision
    intersected_chunks = getChunksIntersected(a_x, a_y, b_x, b_y, chunk_size)

    for chunk_x, chunk_y in intersected_chunks:
        if chunk_y < 0 or chunk_x < 0 or chunk_y >= len(env.chunk_min_depth) or chunk_x >= len(env.chunk_min_depth[0]):
            continue

        chunk_min_depth = env.chunk_min_depth[chunk_y][chunk_x]

        if max(a_d, b_d) >= chunk_min_depth:
            vec = [b_x - a_x, b_y - a_y, b_d - a_d]
            mini = [vec[0] / dist, vec[1] / dist, vec[2] / dist]
            c_x, c_y, c_d = a_x, a_y, a_d

            for _ in range(round(dist) + 1):
                cell_x, cell_y = int(c_x), int(c_y)
                if (
                    chunk_x * chunk_size <= cell_x < (chunk_x + 1) * chunk_size
                    and chunk_y * chunk_size <= cell_y < (chunk_y + 1) * chunk_size
                ):
                    if env.map[cell_y][cell_x] < c_d:
                        return False, dist

                c_x += mini[0]
                c_y += mini[1]
                c_d += mini[2]

    return True, dist

@njit
def GetDistance(a_x, a_y, a_d, b_x, b_y, b_d):
    """
    Calculates the Euclidean distance between two points in 3D space.

    Parameters:
    - a_x, a_y, a_d: Coordinates (x, y, depth) of the first point.
    - b_x, b_y, b_d: Coordinates (x, y, depth) of the second point.

    Returns:
    - Float: The Euclidean distance between the two points.
    """
    return math.sqrt((b_x - a_x) ** 2 + (b_y - a_y) ** 2 + (b_d - a_d) ** 2)

@njit
def VectorMagnitude(vector):
    """
    Calculates the magnitude (length) of a vector in 3D space.

    Parameters:
    - vector: A list or tuple representing the vector components (x, y, z).

    Returns:
    - Float: The magnitude of the vector.
    """
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)

def NormalizedVector(vector):
    """
    Normalizes a vector in 3D space, scaling it to have a magnitude of 1.

    Parameters:
    - vector: A list or tuple representing the vector components (x, y, z).

    Returns:
    - List: A normalized vector with the same direction but a magnitude of 1.
    """
    mag = VectorMagnitude(vector)
    return [
        vector[0] / mag,
        vector[1] / mag,
        vector[2] / mag
    ]

def GetFoodVector(agent, target):
    """
    Calculates a vector pointing from the agent to the food target.

    Parameters:
    - agent: The agent whose position is considered as the starting point.
    - target: The food target position as a list or tuple (x, y, depth).

    Returns:
    - List: A vector from the agent to the target.
    """
    return [
        target[0] - agent.x,
        target[1] - agent.y,
        target[2] - agent.depth,
    ]

def GetAgentVector(agent, target):
    """
    Calculates a vector pointing from one agent to another agent.

    Parameters:
    - agent: The starting agent.
    - target: The target agent whose position is the destination.

    Returns:
    - List: A vector from the first agent to the second agent.
    """
    return [
        target.x - agent.x,
        target.y - agent.y,
        target.depth - agent.depth,
    ]

def CalculateMovementCost(genome):
    """
    Calculates the movement cost of an agent based on its genome attributes. 
    Factors like speed, health, and armor contribute to the cost.

    Parameters:
    - genome: A dictionary representing the agent's genome, with numeric attributes like speed, health, and armor.

    Returns:
    - Float: The calculated movement cost, clamped to a minimum of -10.
    """

    base_cost = -0.1
    weights = {
        "speed": 0.4,
        "health": 0.0003,
        "stomach_size": 0.001,
        "armor": 0.05,
        "bite_damage": 0.003,
        "eyesight_range": 0.002,
        "feed_range": 0.002,
        "bite_range": 0.002,
        "memory": 0.001,
        "depth_tolerance_range": 0.01
    }

    additional_cost = 0.0

    for stat_name, weight in weights.items():
        if stat_name in genome and isinstance(genome[stat_name], (int, float)):
            additional_cost += genome[stat_name] * weight

    movement_cost = base_cost - additional_cost

    movement_cost = max(movement_cost, -10.0)

    return movement_cost

def hsv2rgb(h, s, v):
    """
    Converts an HSV color to an RGB color. The HSV input ranges are [0..180] for hue and [0..1] for saturation and value.

    Parameters:
    - h: The hue component of the color, in the range [0..180].
    - s: The saturation component of the color, in the range [0..1].
    - v: The value (brightness) component of the color, in the range [0..1].

    Returns:
    - Tuple: An RGB color with each component in the range [0..255].
    """
    h_normalized = h / 180.0
    rgb_fractional = colorsys.hsv_to_rgb(h_normalized, s, v)
    return tuple(round(channel * 255) for channel in rgb_fractional)
    