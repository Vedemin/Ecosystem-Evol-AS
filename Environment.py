import numpy as np
import random
from copy import copy
import cv2
import colorsys
from Utils import *
from Agent import Agent

default_genome = {
  "speed": 1.0,
  "health": 100.0,
  "stomach_size": 100.0,
  "type": "herbivore",
  "armor": 0.0,
  "bite_damage": 40.0,
  "eyesight_range": 32.0,
  "feed_range": 3.0,
  "bite_range": 3.0,
  "memory": 20,
  "depth_point": 20.0,  # Mean depth where the fish lives
  "depth_tolerance_range": 15.0,  # Range above and below the mean depth
}

default_params = {
  "mapsize": 256,
  "min_depth": 10,
  "max_depth": 30,
  "starting_population": 10,
  "food_value": 200,
  "max_timesteps": 200000,
  "movement_cost_factor": 0.5,
  "food_per_agent": 4,
  "egg_incubation_time": 100,
  "mutation_factor": 2
}

class Ecosystem():
    def __init__(self, render_mode, params=default_params, image_path="", debug=False):
        self.render_mode = render_mode
        self.debug = debug
        self.params = {
            "mapsize": 128,
            "min_depth": 10,
            "max_depth": 30,
            "starting_population": 15,
            "food_value": 200,
            "movement_cost": -1,
            "food_per_agent": 4,
            "max_timesteps": 200000,
            "movement_cost_factor": 1.0,
            "egg_incubation_time": 100,
            "mutation_factor": 2
        }
        for key in params:
            if key in self.params:
                self.params[key] = params[key]
            else:
                print("Parameter", key, "is invalid")

        self.mapsize = (self.params["mapsize"], self.params["mapsize"])
        self.min_depth = self.params["min_depth"]
        self.max_depth = self.params["max_depth"]
        if image_path == "":
            self.filename = 'small_depth.png'
        else:
            self.filename = image_path
        self.depth_map = cv2.imread(self.filename)
        self.depth_map = cv2.cvtColor(self.depth_map, cv2.COLOR_BGR2GRAY)
        self.depth_map = cv2.resize(
            self.depth_map, self.mapsize, interpolation=cv2.INTER_CUBIC)
        self.display_map = cv2.normalize(
            self.depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        self.cross = np.sqrt(
            self.mapsize[0] ** 2 + self.mapsize[1] ** 2 + self.max_depth ** 2)
        self.depth_partition_precision = 10  # Default precision value

        self.possible_agents = [f"agent_{i}" for i in range(
            self.params["starting_population"])]
        self.foodAmount = self.params["starting_population"] * self.params["food_per_agent"]
        self.egg_incubation_time = self.params["egg_incubation_time"]
        self.agents = {}
        self.foods = []
        self.agentColors = {
            "h1": 90,
            "p1": 0,
            "np": 150
        }

        self.timestep = 0  # Resets the timesteps

    def reset(self, seed=None, options=None):
        self.agentNames = copy(self.possible_agents)
        self.timestep = 0
        self.map = np.zeros(self.mapsize)
        self.foods = []
        # This function starts the world:
        # loads the map from image,
        # spawns agents and generates food
        self.generateMap()
        return

    def render(self):  # Simple OpenCV display of the environment
        image = self.toImage((400, 400))
        scale_x = 400 / self.mapsize[0]
        scale_y = 400 / self.mapsize[1]
        for agent in self.agentNames:
            ag = self.agents[agent]
            agent_rgb = hsv2rgb(
                    self.agentColors[ag.stats["species"]] / 180,
                    1,
                    0.25 + (1 - ag.life / ag.lifespan) * 0.75
                )
            color = (agent_rgb[2], agent_rgb[1], agent_rgb[0])
            org = (int(self.agents[agent].y * scale_y),
                   int(self.agents[agent].x * scale_x - 10))
            image = cv2.putText(image, str(int(self.agents[agent].depth)), org, cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 2)
        cv2.imshow("map", image)
        cv2.waitKey(1)

    def toImage(self, window_size):  # Converts the map to a ready to display image
        food_color = [0, 255, 0]

        img = cv2.bitwise_not(self.display_map)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for food in self.foods:
            img[int(food[0])][int(food[1])][0] = food_color[0]
            img[int(food[0])][int(food[1])][1] = food_color[1]
            img[int(food[0])][int(food[1])][2] = food_color[2]

        for agent in self.agentNames:
            ag = self.agents[agent]
            a_x = ag.x
            a_y = ag.y
            agent_rgb = hsv2rgb(
                    self.agentColors[ag.stats["species"]] / 180,
                    1,
                    0.25 + (1 - ag.life / ag.lifespan) * 0.75
                )
            img[int(a_x)][int(a_y)][0] = agent_rgb[2]
            img[int(a_x)][int(a_y)][1] = agent_rgb[1]
            img[int(a_x)][int(a_y)][2] = agent_rgb[0]

        return cv2.resize(img, window_size, interpolation=cv2.INTER_NEAREST)

    def close(self):
        print(f"Closing environment Ecosystem at {self.timestep} timestep.")
        cv2.destroyAllWindows()

    def stepper(self):
        agents = copy(self.agents)
        for name, agent in agents.items():
            agent.Activate(self)
        if self.render_mode == "human":
            self.render()
        self.timestep += 1
 
    def getDistance(self, a_x, a_y, a_d, b_x, b_y, b_d):
        return np.sqrt((b_x - a_x) ** 2 + (b_y - a_y) ** 2 + (b_d - a_d) ** 2)
    
    def partitionDepthMap(self):
        """Partition the map into chunks and compute min and max depth for each chunk."""
        chunk_size = self.mapsize[0] // self.depth_partition_precision
        self.chunk_min_depth = []
        self.chunk_max_depth = []

        for y in range(0, self.mapsize[1], chunk_size):
            min_row = []
            max_row = []
            for x in range(0, self.mapsize[0], chunk_size):
                chunk = self.map[y:y + chunk_size, x:x + chunk_size]
                min_row.append(chunk.min())
                max_row.append(chunk.max())
            self.chunk_min_depth.append(min_row)
            self.chunk_max_depth.append(max_row)

    def getChunkDepth(self, x, y):
        """Get the min and max depth of the chunk containing (x, y)."""
        chunk_size = self.mapsize[0] // self.depth_partition_precision
        chunk_x = x // chunk_size
        chunk_y = y // chunk_size
        if chunk_y < len(self.chunk_min_depth) and chunk_x < len(self.chunk_min_depth[0]):
            return self.chunk_min_depth[chunk_y][chunk_x], self.chunk_max_depth[chunk_y][chunk_x]
        else:
            raise ValueError("Coordinates out of bounds!")

    def getNeighboringChunkDepths(self, x, y):
        """Get the min and max depths of neighboring chunks around the chunk containing (x, y)."""
        chunk_size = self.mapsize[0] // self.depth_partition_precision
        chunk_x = x // chunk_size
        chunk_y = y // chunk_size

        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = chunk_x + dx, chunk_y + dy
                if 0 <= ny < len(self.chunk_min_depth) and 0 <= nx < len(self.chunk_min_depth[0]):
                    neighbors.append(
                        (self.chunk_min_depth[ny][nx], self.chunk_max_depth[ny][nx])
                    )
        return neighbors

############################################################################################################

    def cstCoord(self, x, y):  # Constrain the passed coordinates so they don't exceed the map
        if x > self.mapsize[0] - 1:
            x = self.mapsize[0] - 1
        elif x < 0:
            x = 0
        if y > self.mapsize[1] - 1:
            y = self.mapsize[1] - 1
        elif y < 0:
            y = 0
        return x, y

############################################################################################################

    def gridRInt(self, xy):  # Returns random int in the map axis limit
        if xy == "y":
            return float(random.randint(0, self.mapsize[0] - 1))
        else:
            return float(random.randint(0, self.mapsize[1] - 1))
        
    def generateMap(self):
        # Loads map file and converts it to a discrete terrain map
        scaling_factor = (self.max_depth - self.min_depth) / 255.0
        self.map = (self.display_map * scaling_factor + self.min_depth).astype(np.int16)
        self.chunk_min_depth = []
        self.chunk_max_depth = []
        print(self.map)

        self.generateNewFood()

        # Generate agents
        for agentID in self.agentNames:
            x = self.gridRInt("x")
            y = self.gridRInt("y")
            depth_point = self.map[int(x)][int(y)]
            depth = random.randint(0, depth_point - 1)
            self.createNewAgent(x, y, depth, agentID, False, self.RandomGenome())

############################################################################################################

    # Check how much food is present and generate what is missing
    def generateNewFood(self, fullGrown=True):
        currentFood = len(self.foods)
        while currentFood < self.foodAmount:
            x = self.gridRInt("x")
            y = self.gridRInt("y")
            depth_point = float(self.map[int(x)][int(y)]) - 2
            self.foods.append([x, y, depth_point, self.params["food_value"]])
            if self.debug:
                print([x, y, depth_point], self.map[int(x)][int(y)])
            currentFood = len(self.foods)

############################################################################################################
    
    def createNewAgent(self, x, y, d, agentID, is_new, genome=default_genome):
        life_start = 0
        if is_new:
            life_start = np.random.uniform(0.1, 0.5)
            self.agentNames.append(agentID)
        self.agents[agentID] = Agent(agentID, [x, y, d], genome, life_start_point=life_start, debug=self.debug)

    def RandomGenome(self):
        return {
            "species": "h1", # Present: h1, c1, np - Nile Perch
            "speed": round(random.uniform(0.5, 3.0), 2),  # Speed between 0.5 and 3.0
            "health": round(random.uniform(50.0, 150.0), 1),  # Health between 50 and 150
            "stomach_size": round(random.uniform(50.0, 200.0), 1),  # Stomach size between 50 and 200
            "type": "herbivore",  # Randomly choose type
            "armor": max(round(random.uniform(0.0, 10.0) - 5, 2), 0),  # Armor between 0 and 10
            "lifespan": 5000,
            "bite_damage": round(random.uniform(10.0, 60.0), 1),  # Bite damage between 10 and 60
            "eyesight_range": round(random.uniform(20.0, 150.0), 1),  # Eyesight range between 20 and 50
            "feed_range": round(random.uniform(2.0, 5.0), 1),  # Feed range between 2 and 5
            "bite_range": round(random.uniform(2.0, 5.0), 1),  # Bite range between 2 and 5
            "memory": random.randint(10, 30),  # Memory between 10 and 30
            "depth_point": random.uniform(self.min_depth / 2,  self.max_depth * 0.75),  # Mean depth where the fish lives
            "depth_tolerance_range": random.uniform(5, 20),  # Range above and below the mean depth
        }
    
############################################################################################################
    
    def KillAgent(self, agentID):
        self.agents.pop(agentID)
        self.agentNames.remove(agentID)