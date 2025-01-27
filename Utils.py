import numpy as np
import colorsys
import math
from numba import njit

def getChunksIntersected(a_x, a_y, b_x, b_y, chunk_size):
        """Calculate the chunks intersected by the line from (a_x, a_y) to (b_x, b_y)."""
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
    if abs(a_x - b_x) + abs(a_y - b_y) + abs(a_d - b_d) > sight_distance * math.sqrt(3):
        return False, 10000

    dist = GetDistance(a_x, a_y, a_d, b_x, b_y, b_d)
    if dist > sight_distance:
        return False, dist
    if dist < 0.001:
        return True, dist
    if env.min_depth > max(a_d, b_d):
        return True, dist

    # Calculate which chunks the line passes through
    chunk_size = env.mapsize[0] // env.depth_partition_precision
    intersected_chunks = getChunksIntersected(a_x, a_y, b_x, b_y, chunk_size)

    # Get depths of intersected chunks
    for chunk_x, chunk_y in intersected_chunks:
        if chunk_y < 0 or chunk_x < 0 or chunk_y >= len(env.chunk_min_depth) or chunk_x >= len(env.chunk_min_depth[0]):
            continue  # Skip chunks outside bounds

        chunk_min_depth = env.chunk_min_depth[chunk_y][chunk_x]

        # If the higher depth of the endpoints is >= chunk's minimum depth, investigate further
        if max(a_d, b_d) >= chunk_min_depth:
            vec = [b_x - a_x, b_y - a_y, b_d - a_d]
            mini = [vec[0] / dist, vec[1] / dist, vec[2] / dist]
            c_x, c_y, c_d = a_x, a_y, a_d

            for _ in range(round(dist) + 1):
                # Check if the ray point is within the current chunk
                cell_x, cell_y = int(c_x), int(c_y)
                if (
                    chunk_x * chunk_size <= cell_x < (chunk_x + 1) * chunk_size
                    and chunk_y * chunk_size <= cell_y < (chunk_y + 1) * chunk_size
                ):
                    # Check if the depth at this point is higher than the map depth
                    if env.map[cell_y][cell_x] < c_d:
                        return False, dist

                # Advance ray
                c_x += mini[0]
                c_y += mini[1]
                c_d += mini[2]

    return True, dist

@njit
def GetDistance(a_x, a_y, a_d, b_x, b_y, b_d):
    return math.sqrt((b_x - a_x) ** 2 + (b_y - a_y) ** 2 + (b_d - a_d) ** 2)

@njit
def VectorMagnitude(vector):
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)

def NormalizedVector(vector):
    mag = VectorMagnitude(vector)
    return [
        vector[0] / mag,
        vector[1] / mag,
        vector[2] / mag
    ]

def GetFoodVector(agent, target):
    return [
        target[0] - agent.x,
        target[1] - agent.y,
        target[2] - agent.depth,
    ]

def GetAgentVector(agent, target):
    return [
        target.x - agent.x,
        target.y - agent.y,
        target.depth - agent.depth,
    ]

def CalculateMovementCost(genome):
    """
    Calculates the movement cost of an agent based on numeric genome stats,
    excluding 'lifespan' but including 'depth_tolerance_range'.
    Weights are chosen to scale reasonably with each stat's typical range.
    """

    # Base movement cost (negative = cost to move)
    base_cost = -0.1

    # Approximate weights for each numeric stat.
    # Adjusted to ensure health doesn't inflate cost too much,
    # and speed/armor remain significant factors.
    weights = {
        "speed": 0.4,                  # High penalty for high speed
        "health": 0.0003,             # Very small weight, as health can be in hundreds
        "stomach_size": 0.001,        # Small impact
        "armor": 0.05,                # Medium impact
        "bite_damage": 0.003,         # Small impact
        "eyesight_range": 0.002,      # Slight impact
        "feed_range": 0.002,          # Slight impact
        "bite_range": 0.002,          # Slight impact
        "memory": 0.001,              # Very minimal impact
        "depth_tolerance_range": 0.001  # Slight penalty for wide depth flexibility
    }

    additional_cost = 0.0

    # Sum up the additional cost from each relevant numeric stat
    for stat_name, weight in weights.items():
        if stat_name in genome and isinstance(genome[stat_name], (int, float)):
            additional_cost += genome[stat_name] * weight

    # Combine with base cost (the more you invest in stats, the higher your negative cost)
    movement_cost = base_cost - additional_cost

    # Clamp so it doesn't go below -10
    movement_cost = max(movement_cost, -10.0)

    return movement_cost

def hsv2rgb(h, s, v):
    """
    Converts an HSV color (where h in [0..180], s in [0..1], v in [0..1])
    to an RGB tuple in [0..255].
    """
    # Convert hue from [0..180] to [0..1] for Python's colorsys
    h_normalized = h / 180.0
    
    # Use the standard colorsys, which expects h in [0..1], s in [0..1], v in [0..1]
    rgb_fractional = colorsys.hsv_to_rgb(h_normalized, s, v)
    
    # Scale from [0..1] to [0..255]
    return tuple(round(channel * 255) for channel in rgb_fractional)
    