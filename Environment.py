import numpy as np
import random
from copy import copy
import cv2
import colorsys
import pygame
from Utils import *
from Agent import Agent
import matplotlib.pyplot as plt
import io


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
    "mapsize": 128,
    "min_depth": 10,
    "max_depth": 30,
    "starting_population": 15,
    "food_value": 200,
    "movement_cost": -1,
    "food_per_agent": 2,
    "max_timesteps": 200000,
    "movement_cost_factor": 1.0,
    "egg_incubation_time": 100,
    "mutation_factor": 2,
    "starting_plant_population": 100,
    "plant_growth_speed": 0.01,
    "plant_minimum_growth_percentage": 0.5,
    "plant_spread_amount": 2,
    "plant_spread_radius": 400,
    "plant_spread_interval": 500
}

default_species = {
    "hb1": {
        "name": "generic herbivore",
        "type": "herbivore",
        "population": 50,
        "color": 90,
        "rgb": hsv2rgb(90, 1, 1),
        "genome": {
            "speed": {"min": 1.0, "max": 6.0},
            "health": {"min": 170.0, "max": 340.0},
            "stomach_size": {"min": 100.0, "max": 150.0},  # Moderate stomach size
            "armor": {"min": 0.0, "max": 5.0},  # Relatively low armor
            "bite_damage": {"min": 5.0, "max": 20.0},  # Low bite damage for herbivores
            "eyesight_range": {"min": 30.0, "max": 80.0},  # Moderate eyesight
            "feed_range": {"min": 3.0, "max": 4.0},  # Moderate feeding range
            "bite_range": {"min": 2.0, "max": 3.0},  # Moderate bite range
            "memory": {"min": 15, "max": 25},  # Moderate memory
            "depth_tolerance_range": {"min": 5, "max": 15}  # Moderate depth tolerance
        }
    },
    "cv1": {
        "name": "generic carnivore",
        "type": "carnivore",
        "population": 60,
        "color": 0,
        "rgb": hsv2rgb(0, 1, 1),
        "genome": {
            "speed": {"min": 2.0, "max": 4.0},  # Carnivores generally faster
            "health": {"min": 60.0, "max": 110.0},
            "stomach_size": {"min": 80.0, "max": 120.0},  # Moderate stomach size
            "armor": {"min": 2.0, "max": 7.0},  # Moderate armor
            "bite_damage": {"min": 20.0, "max": 40.0},  # Higher bite damage for carnivores
            "eyesight_range": {"min": 10.0, "max": 50.0},  # Good eyesight
            "feed_range": {"min": 2.0, "max": 3.0},  # Moderate feeding range
            "bite_range": {"min": 2.0, "max": 4.0},  # Moderate bite range
            "memory": {"min": 10, "max": 20},  # Moderate memory
            "depth_tolerance_range": {"min": 5, "max": 15}  # Moderate depth tolerance
        }
    },
    "n_p": {
        "name": "Nile Perch",
        "type": "carnivore",
        "population": 10,
        "color": 150,
        "rgb": hsv2rgb(150, 1, 1),
        "genome": {
            "speed": {"min": 3.0, "max": 10.0},  # Very fast
            "health": {"min": 580.0, "max": 1150.0},  # High health
            "stomach_size": {"min": 120.0, "max": 200.0},  # Large stomach size
            "armor": {"min": 5.0, "max": 10.0},  # High armor
            "bite_damage": {"min": 40.0, "max": 70.0},  # Very high bite damage
            "eyesight_range": {"min": 30.0, "max": 70.0},  # Excellent eyesight
            "feed_range": {"min": 1.0, "max": 2.0},  # Short feeding range (aggressive)
            "bite_range": {"min": 3.0, "max": 5.0},  # High bite range
            "memory": {"min": 20, "max": 30},  # High memory
            "depth_tolerance_range": {"min": 3, "max": 10}  # Narrow depth tolerance (specialization)
        }
    },
}


class Ecosystem:
    def __init__(self, render_mode, params=default_params, species=default_species, image_path="", debug=False):
        self.render_mode = render_mode
        self.debug = debug
        self.params = {
            "mapsize": 128,
            "min_depth": 10,
            "max_depth": 30,
            "starting_population": 15,
            "food_value": 200,
            "movement_cost": -1,
            "food_per_agent": 2,
            "max_timesteps": 200000,
            "movement_cost_factor": 1.0,
            "egg_incubation_time": 100,
            "mutation_factor": 2,
            "starting_plant_population": 100,
            "plant_growth_speed": 0.01,
            "plant_minimum_growth_percentage": 0.5,
            "plant_spread_amount": 2,
            "plant_spread_radius": 400,
            "plant_spread_interval": 500
        }
        self.species = species

        for key in params:
            if key in self.params:
                self.params[key] = params[key]
            else:
                print("Parameter", key, "is invalid")

        
        if "max_plants_global" not in params:
            params["max_plants_global"] = 200
        self.params["max_plants_global"] = params["max_plants_global"]  

        self.mapsize = (self.params["mapsize"], self.params["mapsize"])
        self.min_depth = self.params["min_depth"]
        self.max_depth = self.params["max_depth"]
        if image_path == "":
            self.filename = "small_depth.png"
        else:
            self.filename = image_path
        self.depth_map = cv2.imread(self.filename)
        self.depth_map = cv2.cvtColor(self.depth_map, cv2.COLOR_BGR2GRAY)
        self.depth_map = cv2.resize(
            self.depth_map, self.mapsize, interpolation=cv2.INTER_CUBIC
        )
        self.display_map = cv2.normalize(
            self.depth_map,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )
        
        self.cross = np.sqrt(
            self.mapsize[0] ** 2 + self.mapsize[1] ** 2 + self.max_depth**2
        )
        self.depth_partition_precision = 4

        self.possible_agents = [
            f"a_{i}" for i in range(self.params["starting_population"])
        ]
        self.possible_agents = []
        for species, data in self.species.items():
            for i in range(data["population"]):
                self.possible_agents.append(f"{species}_agent_{i}")
        
        self.foodAmount = (
            self.params["starting_population"] * self.params["food_per_agent"]
        )
        self.egg_incubation_time = self.params["egg_incubation_time"]
        self.agents = {}
        self.foods = []

        self.timestep = 0
        self.map_surface = None
        if self.render_mode == "human":
            pygame.init()
            self.window_size = 800
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Ecosystem Simulation")
            self.clock = pygame.time.Clock()
            # self.population_history = {species: [] for species in self.species.keys()}
            # self.graph_surface = None

            self.window_width = 800
            self.simulation_height = 800
            self.graph_height = 200
            self.window_height = self.simulation_height + self.graph_height
            self.graph_offset_y = self.simulation_height
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.population_history = {species: [0] * 200 for species in self.species.keys()}
        self.agent_history = []
        self.plant_history = []
        self.death_log = {}
        self.finished = False
        self.full_population_history = {species: [(0, 0)] for species in self.species.keys()}
        self.max_history_length = 200

    def reset(self, seed=None, options=None):
        self.agentNames = copy(self.possible_agents)
        self.timestep = 0
        self.map = np.zeros(self.mapsize)
        self.foods = []
        self.finished = False
        # This function starts the world:
        # loads the map from image,
        # spawns agents and generates food
        self.generateMap()
        return

    def render(self):
        self.screen.fill((0, 0, 0))
        if self.map_surface is None:
            self.map_surface = pygame.Surface(self.mapsize)
            
            # Ensure display_map is in the correct format
            if len(self.display_map.shape) == 2:  # Grayscale map
                display_map_rgb = np.stack([self.display_map] * 3, axis=-1)  # Convert to RGB
            elif len(self.display_map.shape) == 3 and self.display_map.shape[2] == 3:
                display_map_rgb = self.display_map  # Already in RGB
            else:
                raise ValueError("Unexpected shape for display_map: " + str(self.display_map.shape))

            display_map_uint8 = display_map_rgb.astype(np.uint8)
            pygame.surfarray.blit_array(self.map_surface, display_map_uint8)
        self.screen.blit(pygame.transform.scale(self.map_surface, (self.window_size, self.window_size)), (0, 0))


        # Draw all foods
        for food in self.foods:
            pygame.draw.circle(
                self.screen, 
                (0, 255, 0),  # Green color for food
                (int(food[1]), int(food[0])),  # Flip x and y for correct positioning
                3,  # Food size
            )

        # Initialize font (outside the loop, do this once in your setup code)
        if not hasattr(self, 'font'):
            self.font = pygame.font.Font(None, 24)

        # Draw all agents
        for agent in self.agentNames:
            ag = self.agents[agent]
            agent_rgb = hsv2rgb(
                self.species[ag.stats["species"]]["color"] / 180,
                1,
                0.25 + (1 - ag.life / ag.lifespan) * 0.75,
            )

            # Convert to integers (assuming hsv2rgb is already in [0, 255])
            agent_color = tuple(int(channel) for channel in agent_rgb)

            pygame.draw.circle(
                self.screen,
                agent_color,  # Pass the fixed color
                (int(ag.y), int(ag.x)),
                ag.health / ((self.species[ag.stats["species"]]["genome"]["health"]["min"] + self.species[ag.stats["species"]]["genome"]["health"]["max"]) / 2),
            )
            
            depth_text = str(round(ag.depth))  # Format depth to 2 decimal places

            # Render the text
            text_surface = self.font.render(depth_text, True, agent_color)
            # Blit the text above the agent
            text_position = (int(ag.y) - 10, int(ag.x) - 20)  # Offset to position above the agent
            # Render the population graph
            # if self.graph_surface:
            #     self.screen.blit(self.graph_surface, (10, 10))  # Position graph in the window
            self.screen.blit(text_surface, text_position)

        self.draw_population_graph()
        pygame.display.flip()

    def close(self):
        print(f"Closing environment Ecosystem at {self.timestep} timestep.")
        pygame.quit()
        self.finished = True
        return self.full_population_history

    def stepper(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        if len(self.agents) == 0:
            self.close()
            return

        agents_copy = copy(self.agents)
        for name, agent in agents_copy.items():
            if name in self.agents:
                agent.Activate(self)

        self.agent_history.append(copy(self.agents))
        self.plant_history.append(copy(self.foods))
        for species in self.species.keys():
            population = sum(1 for agent in self.agents.values() if agent.stats["species"] == species)
            total_movement_cost = sum(
                agent.movement_cost for agent in self.agents.values()
                if agent.stats["species"] == species
            )
            avg_movement_cost = total_movement_cost/population if population else 0
            history = self.population_history[species]
            history.append(population)
            self.full_population_history[species].append((population, avg_movement_cost))
            if len(history) > self.max_history_length:
                history.pop(0)

        ## Changes start
        # Update plant growth + spreading each step
        self.updatePlants()
        ## Changes end

        if self.render_mode == "human":
            self.render()
            self.clock.tick(120)

        self.timestep += 1


    def draw_population_graph(self):
        pygame.draw.rect(
            self.screen,
            (50, 50, 50),
            (0, self.graph_offset_y, self.window_width, self.graph_height),
        )

        pygame.draw.line(
            self.screen,
            (200, 200, 200),
            (50, self.graph_offset_y),
            (50, self.graph_offset_y + self.graph_height),
            2,
        )
        pygame.draw.line(
            self.screen,
            (200, 200, 200),
            (50, self.graph_offset_y + self.graph_height - 10),
            (self.window_width - 10, self.graph_offset_y + self.graph_height - 10),
            2,
        )

        max_population = max(max(history) for history in self.population_history.values())
        max_population = max_population or 1
        y_scale = (self.graph_height - 20) / max_population
        x_scale = (self.window_width - 60) / self.max_history_length

        for species, data in self.species.items():
            hsv_color = data["color"] / 180, 1, 1
            rgb_color = hsv2rgb(*hsv_color)
            rgb_color = tuple(int(c) for c in rgb_color)

            history = self.population_history[species]

            for i in range(1, len(history)):
                x1 = 50 + (i - 1) * x_scale
                y1 = self.graph_offset_y + self.graph_height - 10 - history[i - 1] * y_scale
                x2 = 50 + i * x_scale
                y2 = self.graph_offset_y + self.graph_height - 10 - history[i] * y_scale
                pygame.draw.line(self.screen, rgb_color, (x1, y1), (x2, y2), 2)

    def updatePlants(self):
        """
        ## Changes start
        # For each plant, increase growth_percentage, decrement spread timer,
        # and if conditions are met, spread.
        # Agents will use self.params["food_value"] * growth_percentage to get
        # the actual "food" when they eat (handled in Agent class).
        ## Changes end
        """
        for i, plant in enumerate(self.foods):
            x, y, depth, growth_pct, spread_t = plant

            growth_pct += self.params["plant_growth_speed"]
            if growth_pct > 1.0:
                growth_pct = 1.0

            spread_t -= 1

            if (spread_t <= 0 and
                growth_pct >= self.params["plant_minimum_growth_percentage"]):
                self.spreadPlants(x, y)
                spread_t = self.params["plant_spread_interval"]

            self.foods[i] = [x, y, depth, growth_pct, spread_t]


    def update_population_graph(self):
        fig, ax = plt.subplots(figsize=(4, 3))
        for species, history in self.population_history.items():
            ax.plot(history, label=self.species[species]["name"])
        
        ax.set_title("Species Population Balance")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Population")
        ax.legend()
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format="PNG")
        buf.seek(0)
        image = pygame.image.load(buf)
        buf.close()
        plt.close(fig)
        
        self.graph_surface = pygame.transform.scale(image, (400, 300))

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
                chunk = self.map[y : y + chunk_size, x : x + chunk_size]
                min_row.append(chunk.min())
                max_row.append(chunk.max())
            self.chunk_min_depth.append(min_row)
            self.chunk_max_depth.append(max_row)

    def getChunkDepth(self, x, y):
        """Get the min and max depth of the chunk containing (x, y)."""
        chunk_size = self.mapsize[0] // self.depth_partition_precision
        chunk_x = x // chunk_size
        chunk_y = y // chunk_size
        if chunk_y < len(self.chunk_min_depth) and chunk_x < len(
            self.chunk_min_depth[0]
        ):
            return (
                self.chunk_min_depth[chunk_y][chunk_x],
                self.chunk_max_depth[chunk_y][chunk_x],
            )
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
                if 0 <= ny < len(self.chunk_min_depth) and 0 <= nx < len(
                    self.chunk_min_depth[0]
                ):
                    neighbors.append(
                        (self.chunk_min_depth[ny][nx], self.chunk_max_depth[ny][nx])
                    )
        return neighbors

    ############################################################################################################

    def cstCoord(
        self, x, y
    ):  # Constrain the passed coordinates so they don't exceed the map
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
        self.map = ((255 - self.display_map) * scaling_factor + self.min_depth).astype(np.int16)
        self.chunk_min_depth = []
        self.chunk_max_depth = []
        print(self.map)

        self.generateNewFood()

        # Generate agents
        for agentID in self.agentNames:
            x = self.gridRInt("x")
            y = self.gridRInt("y")
            species = agentID[:3]
            genome = self.RandomGenome(species=species)
            depth_point = self.map[int(x)][int(y)]
            depth = np.clip(
                random.randint(0, depth_point - 1),
                genome["depth_point"] - genome["depth_tolerance_range"] + 1,
                genome["depth_point"] + genome["depth_tolerance_range"] - 1,
            )
            self.createNewAgent(x, y, depth, agentID, False, genome=genome)

    ############################################################################################################

    # Check how much food is present and generate what is missing
    def generateNewFood(self):
        """
        Spawns self.params["starting_plant_population"] plants (global limit).
        One-third of them at a random initial growth percentage, the remaining
        two-thirds at full (1.0) growth.
        We stop spawning if we reach or exceed self.params["max_plants_global"].
        """
        total_plants = self.params["starting_plant_population"]
        threshold = total_plants // 3  # One-third

        for i in range(total_plants):
            # Check global maximum
            if len(self.foods) >= self.params["max_plants_global"]:
                break

            x = self.gridRInt("x")
            y = self.gridRInt("y")
            depth_point = float(self.map[int(x)][int(y)])

            # Decide growth percentage
            if i < threshold:
                initial_growth = random.uniform(0.0, 1.0)
            else:
                initial_growth = 1.0

            # Random spread timer so not all plants spread simultaneously
            spread_timer = random.randint(1, self.params["plant_spread_interval"])

            # Append the plant if we're under the global limit
            self.foods.append([x, y, depth_point, initial_growth, spread_timer])


    def spreadPlants(self, px, py):
        """
        Attempts to spawn self.params["plant_spread_amount"] new plants
        around (px, py), each with 0 growth initially. Respects the global
        maximum (max_plants_global).
        """
        for _ in range(self.params["plant_spread_amount"]):
            # Check global maximum
            if len(self.foods) >= self.params["max_plants_global"]:
                break

            r = random.uniform(0, self.params["plant_spread_radius"])
            theta = random.uniform(0, 2 * np.pi)
            nx = px + r * np.cos(theta)
            ny = py + r * np.sin(theta)

            # Must be in map bounds
            if 0 < nx < self.mapsize[0] - 1 and 0 < ny < self.mapsize[1] - 1:
                # Depth is the local terrain minus 1
                new_depth = float(self.map[int(nx)][int(ny)]) - 1

                # Add the new plant
                self.foods.append([
                    nx, ny, new_depth,
                    0.0,  # starts at zero growth
                    self.params["plant_spread_interval"]
                ])




    ############################################################################################################

    def createNewAgent(self, x, y, d, agentID, is_new, genome=default_genome):
        life_start = 0
        if is_new:
            life_start = np.random.uniform(0.1, 0.5)
            self.agentNames.append(agentID)
        species_data = self.species[genome["species"]]
        self.agents[agentID] = Agent(
            agentID, [x, y, d], genome, avg_lifespan=species_data["genome"]["lifespan"], life_start_point=life_start, debug=self.debug
        )

    def RandomGenome(self, species):
        genome_ranges = self.species[species].get("genome", {}) 

        return {
            "species": species,
            "speed": round(random.uniform(
                genome_ranges.get("speed", {"min": 0.5, "max": 3.0})["min"], 
                genome_ranges.get("speed", {"min": 0.5, "max": 3.0})["max"]
            ), 2),
            "health": round(random.uniform(
                genome_ranges.get("health", {"min": 50.0, "max": 150.0})["min"], 
                genome_ranges.get("health", {"min": 50.0, "max": 150.0})["max"]
            ), 1),
            "stomach_size": round(random.uniform(
                genome_ranges.get("stomach_size", {"min": 50.0, "max": 200.0})["min"], 
                genome_ranges.get("stomach_size", {"min": 50.0, "max": 200.0})["max"]
            ), 1),
            "type": self.species[species]["type"],
            "armor": max(
                round(random.uniform(
                    genome_ranges.get("armor", {"min": 0.0, "max": 10.0})["min"], 
                    genome_ranges.get("armor", {"min": 0.0, "max": 10.0})["max"]
                ) - 5, 2), 0
            ),
            "lifespan": genome_ranges.get("lifespan", 5000),  # You might want to make this configurable in the genome
            "egg_lifespan_required": genome_ranges.get("egg_lifespan_required", 0.2),
            "bite_damage": round(random.uniform(
                genome_ranges.get("bite_damage", {"min": 10.0, "max": 60.0})["min"], 
                genome_ranges.get("bite_damage", {"min": 10.0, "max": 60.0})["max"]
            ), 1),
            "eyesight_range": round(random.uniform(
                genome_ranges.get("eyesight_range", {"min": 20.0, "max": 150.0})["min"], 
                genome_ranges.get("eyesight_range", {"min": 20.0, "max": 150.0})["max"]
            ), 1),
            "feed_range": round(random.uniform(
                genome_ranges.get("feed_range", {"min": 2.0, "max": 5.0})["min"], 
                genome_ranges.get("feed_range", {"min": 2.0, "max": 5.0})["max"]
            ), 1),
            "bite_range": round(random.uniform(
                genome_ranges.get("bite_range", {"min": 2.0, "max": 5.0})["min"], 
                genome_ranges.get("bite_range", {"min": 2.0, "max": 5.0})["max"]
            ), 1),
            "memory": random.randint(
                genome_ranges.get("memory", {"min": 10, "max": 30})["min"], 
                genome_ranges.get("memory", {"min": 10, "max": 30})["max"]
            ),
            "depth_point": random.uniform(
                self.min_depth / 2, self.max_depth * 0.75
            ),
            "depth_tolerance_range": random.uniform(
                genome_ranges.get("depth_tolerance_range", {"min": 5, "max": 70})["min"], 
                genome_ranges.get("depth_tolerance_range", {"min": 5, "max": 70})["max"]
            ),
        }

    ############################################################################################################

    def KillAgent(self, agentID, reason="unknown"):
        """
        Removes an agent from the simulation and logs its cause of death/timestep.
        """
        if agentID in self.agents:
            self.death_log[agentID] = {
                "reason": reason,
                "time": self.timestep
            }
            self.agents.pop(agentID)
            self.agentNames.remove(agentID)