import numpy as np
from Utils import *

default_genome = {
    "species": "h1",  # Present: h1, c1, np - Nile Perch
    "lifespan": 5000,
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


class Agent:
    def __init__(
        self, name, position, genome=default_genome, life_start_point=0, debug=False
    ):
        self.debug = debug
        if self.debug:
            print(genome)
        self.name = name
        self.stats = genome
        self.health = self.stats["health"]
        self.lifespan = np.random.normal(
            self.stats["lifespan"], 0.15 * self.stats["lifespan"]
        )
        self.life = np.round(life_start_point * self.lifespan)
        self.life_factor = self.LifeFactor()
        self.speed = self.stats["speed"] * self.life_factor
        self.stomach_size = self.stats["stomach_size"] * self.life_factor
        self.food = self.stomach_size
        self.type = self.stats["type"]
        self.depth_min = genome["depth_point"] - genome["depth_tolerance_range"]
        self.depth_max = genome["depth_point"] + genome["depth_tolerance_range"]
        self.movement_cost = CalculateMovementCost(genome)
        if self.debug:
            print(self.movement_cost)
        self.x = position[0]
        self.y = position[1]
        self.depth = position[2]
        self.distances = {}
        self.closest_food_distance = 100000
        self.closest_food = [
            self.closest_food_distance,
            self.closest_food_distance,
            self.closest_food_distance,
        ]
        self.possible_mate = None
        self.rememberedFood = self.closest_food
        self.memory = 0
        self.egg_permitted = 0
        self.patrolPoint = [-1, -1, -1]

    def LifeFactor(self):
        if self.life >= self.lifespan:
            return 0

        # Calculate the thresholds
        t1 = 0.2 * self.lifespan  # 20% of X
        t2 = 0.6 * self.lifespan  # 60% of X

        if 0 <= self.life < t1:
            return 0.5 + (0.5 * (self.life / t1))
        elif t1 <= self.life < t2:
            return 1.0
        elif t2 <= self.life < self.lifespan:
            return 1.0 - (0.9 * ((self.life - t2) / (self.lifespan - t2)))
        return 0

    def Activate(self, env):
        self.life_factor = self.LifeFactor()
        self.speed = self.stats["speed"] * self.life_factor
        self.stomach_size = self.stats["stomach_size"] * self.life_factor
        if self.debug:
            print(f"Activate {self.name}")
        self.CalculateDistances(env)
        self.SelectAction(env)
        self.ApplyDepthDamage(env)
        self.egg_permitted += 1
        self.life += 1

    def CalculateDistances(self, env):
        """Calculate distances to other agents and food items with explanations for target selection."""
        eyesight_range = self.stats["eyesight_range"] * self.life_factor

        # Calculate distances and visibility to other agents
        for name, agent in env.agents.items():
            if name == self.name or name not in env.agents:
                continue  # Skip self
            visible, distance = RayCast(
                env,
                self.x,
                self.y,
                self.depth,
                agent.x,
                agent.y,
                agent.depth,
                eyesight_range,
            )

            # Record distances and visibility status for each agent
            self.distances[agent.name] = {"distance": distance, "visible": visible}

        # Filter visible agents and sort them by proximity
        self.visible_agents = {
            name: data for name, data in self.distances.items() if data["visible"] and name in env.agents and env.agents[name].stats["species"] != self.stats["species"]
        }
        self.sorted_agents = sorted(
            self.visible_agents.keys(), key=lambda name: self.visible_agents[name]["distance"]
        )
        if (len(self.sorted_agents) > 0):
            self.closest_agent = env.agents[self.sorted_agents[0]]
            self.closest_agent_distance = self.visible_agents[self.sorted_agents[0]]["distance"]
        else:
            self.closest_agent_distance = 100000
            self.closest_agent = None

        # If the agent is ready for breeding, look for a potential mate
        if self.IsBreeding(env):
            self.possible_mate = next(
                (
                    agent
                    for agent in self.sorted_agents
                    if agent in env.agentNames and env.agents[agent].IsBreeding(env) and env.agents[agent].stats["species"] == self.stats["species"]
                ),
                None,
            )
            if self.debug:
                if self.possible_mate:
                    print(
                        f"[{self.name}] Potential mate selected: {self.possible_mate}. Closest visible mate ready for breeding."
                    )
                else:
                    print(
                        f"[{self.name}] No suitable mates visible within eyesight range."
                    )

        if self.type == "carnivore":
            if len(self.visible_agents) == 0:
                if self.memory > 0 and self.remembered_agent:
                    # Use memory to target the last seen agent
                    if self.debug:
                        print(
                            f"[{self.name}] No visible agents. Moving to remembered agent {self.remembered_agent.name}."
                        )
                    self.memory -= 1
                    up_and_towards = GetFoodVector(self, [
                        self.remembered_agent.x,
                        self.remembered_agent.y,
                        self.remembered_agent.depth
                    ])
                    up_and_towards[2] = self.speed * 2
                    self.Move(self.MovementVector(up_and_towards), env)
                else:
                    # Forget the agent if memory is depleted
                    if self.debug:
                        print(f"[{self.name}] Memory depleted. Forgetting agent.")
                    self.closest_agent_distance = 100000
                    self.closest_agent = None
                    self.remembered_agent = None
            return
        
        # Reset closest food memory if memory is exhausted or the remembered food is no longer available
        if self.memory <= 0 or self.closest_food not in env.foods:
            if self.debug:
                print(
                    f"[{self.name}] Forgetting closest food: Memory {self.memory} or food no longer available."
                )
            self.memory = 0
            self.closest_food_distance = 100000  # Effectively "infinite" distance
            self.closest_food = [
                self.closest_food_distance,
                self.closest_food_distance,
                self.closest_food_distance,
            ]

        if self.stats["type"] == "herbivore":
            # Find the closest visible food item
            for food in env.foods:
                visible, distance = RayCast(
                    env,
                    self.x,
                    self.y,
                    self.depth,
                    food[0],
                    food[1],
                    food[2],
                    eyesight_range,
                )
                if visible and distance < self.closest_food_distance:
                    if self.debug:
                        print(
                            f"[{self.name}] New food target selected at {food} with distance {distance}."
                        )
                    self.closest_food_distance = distance
                    self.closest_food = food
                    self.memory = int(
                        np.round(self.stats["memory"] * self.life_factor)
                    )  # Reset memory when new food is visible

            if self.closest_food_distance > eyesight_range:
                # Food is no longer visible but was remembered
                if self.debug:
                    print(
                        f"[{self.name}] Closest food {self.closest_food} is out of eyesight range. Reducing memory {self.memory}."
                    )
                self.memory = max(
                    0, self.memory - 1
                )  # Decrease memory but ensure it doesn't drop below 0
                if self.memory == 0:
                    if self.debug:
                        print(f"[{self.name}] Memory depleted. Forgetting food.")
                    self.closest_food_distance = 100000
                    self.closest_food = [100000, 100000, 100000]

    def MovementVector(self, vector):
        vector_mag = VectorMagnitude(vector)
        if vector_mag <= self.speed:
            return vector
        result = NormalizedVector(vector)
        return [result[0] * self.speed, result[1] * self.speed, result[2] * self.speed]

    def SelectAction(self, env):
        """Determine the agent's action for the current timestep."""
        # print("Action selection")
        if self.IsThreatened(env):
            self.RunAway(env)
        elif self.IsBreeding(env):
            self.BreedingBehavior(env)
        else:
            self.AgentBehavior(env)
        self.CheckVitals(env)

    def IsThreatened(self, env):
        # TODO: If the agent has a carnivorous or omnivorous agent nearby from a different species - return true
        # Use the existing agent dictionaries created in the CalculateDistances function
        """Determine if the agent is threatened by nearby predators.
        
        Returns:
            bool: True if a threat is detected, False otherwise
        """
        # Only herbivores can be threatened
        if self.type != "herbivore":
            return False
            
        # Check each visible agent
        for agent_name in self.sorted_agents:
            if agent_name not in env.agents:
                continue
                
            threat = env.agents[agent_name]
            # Agent is threatened if there's a carnivore or omnivore of a different species nearby
            if (threat.type in ["carnivore", "omnivore"] and 
                threat.stats["species"] != self.stats["species"] and
                self.visible_agents[agent_name]["distance"] < self.stats["eyesight_range"] * self.life_factor):
                return True
                
        return False

    def RunAway(self, env):
        # TODO: The agent should select a vector that brings it exactly away from the threat move there like in the movement functions.
        # The vector must avoid clipping through walls or going outside of the map.
        # The agent should move the maximum distance it can during this time.
        # Use the existing agent dictionaries created in the CalculateDistances function
        """Execute evasive maneuvers when threatened.
        
        The agent moves directly away from the nearest threat while staying within
        map boundaries and avoiding obstacles.
        """
        # Find the closest threat
        closest_threat = None
        closest_distance = float('inf')
        
        for agent_name in self.sorted_agents:
            if agent_name not in env.agents:
                continue
                
            threat = env.agents[agent_name]
            if threat.type in ["carnivore", "omnivore"] and threat.stats["species"] != self.stats["species"]:
                distance = self.visible_agents[agent_name]["distance"]
                if distance < closest_distance:
                    closest_threat = threat
                    closest_distance = distance
        
        if not closest_threat:
            return
            
        # Calculate vector away from threat
        away_vector = [
            self.x - closest_threat.x,
            self.y - closest_threat.y,
            self.depth - closest_threat.depth
        ]
        
        # Normalize the vector
        magnitude = (away_vector[0]**2 + away_vector[1]**2 + away_vector[2]**2)**0.5
        if magnitude > 0:
            away_vector = [
                away_vector[0] / magnitude * self.speed,
                away_vector[1] / magnitude * self.speed,
                away_vector[2] / magnitude * self.speed
            ]
        
        # Calculate new position
        new_x = self.x + away_vector[0]
        new_y = self.y + away_vector[1]
        new_depth = self.depth + away_vector[2]
        
        # Ensure new position is within map boundaries
        new_x = max(0, min(new_x, env.params["mapsize"] - 1))
        new_y = max(0, min(new_y, env.params["mapsize"] - 1))
        
        # Ensure new depth is within water column and agent's depth tolerance
        seabed_depth = env.map[int(new_x), int(new_y)]
        min_depth = max(env.params["min_depth"], self.depth_min)
        max_depth = min(seabed_depth - 1, self.depth_max)
        new_depth = max(min_depth, min(new_depth, max_depth))
        
        # Calculate final movement vector
        final_vector = [
            new_x - self.x,
            new_y - self.y,
            new_depth - self.depth
        ]
        
        # Move the agent
        self.Move(final_vector, env) 

    def HerbivoreBehavior(self, env):
        # print(f"[{self.name}] Choosing Herbivore Behavior: Searching for food.")
        """Define the behavior of an herbivore agent."""
        if self.closest_food_distance < self.stats["feed_range"] * self.life_factor:
            if self.debug:
                print(
                    f"[{self.name}] Eating food: Food is within feeding range ({self.closest_food_distance})."
                )
            self.Eat(env)
        elif (
            self.closest_food_distance < self.stats["eyesight_range"] * self.life_factor
        ):
            if self.debug:
                print(
                    f"[{self.name}] Moving towards food: Food is within eyesight range ({self.closest_food_distance})."
                )
            self.Move(self.MovementVector(GetFoodVector(self, self.closest_food)), env)
        elif self.memory > 0 and self.closest_food_distance < 90000:
            if self.debug:
                print(
                    f"[{self.name}] Recalling food location from memory: Moving to remembered food at {self.rememberedFood}."
                )
            self.memory -= 1
            up_and_towards = GetFoodVector(self, self.rememberedFood)
            up_and_towards[2] = self.speed * 2
            self.Move(self.MovementVector(up_and_towards), env)
        else:
            self.PatrolLogic(env)

    def AgentBehavior(self, env):
        """Define the behavior of an agent based on its type."""
        if self.type == "herbivore":
            self.HerbivoreBehavior(env)
        elif self.type == "carnivore":
            self.CarnivoreBehavior(env)
        elif self.type == "omnivore":
            self.OmnivoreBehavior(env)

    def CarnivoreBehavior(self, env):
        """Define the behavior of a carnivore agent."""
        # Closest agent distance logic (carnivorous feeding logic)
        if self.closest_agent_distance < self.stats["bite_range"] * self.life_factor:
            if self.debug:
                print(
                    f"[{self.name}] Attacking agent: Agent is within attack range ({self.closest_agent_distance})."
                )
            self.Attack(env, self.closest_agent)
        elif (
            self.closest_agent_distance
            < self.stats["eyesight_range"] * self.life_factor
        ):
            if self.debug:
                print(
                    f"[{self.name}] Moving towards agent: Agent is within eyesight range ({self.closest_agent_distance})."
                )
            self.Move(
                self.MovementVector(GetAgentVector(self, self.closest_agent)), env
            )
        elif self.memory > 0 and self.closest_agent_distance < 90000:
            if self.debug:
                print(
                    f"[{self.name}] Recalling agent location from memory: Moving to remembered agent at {self.rememberedAgent}."
                )
            self.memory -= 1
            toward_agent = GetAgentVector(self, self.rememberedAgent)
            toward_agent[2] = self.speed * 2
            self.Move(self.MovementVector(toward_agent), env)
        else:
            self.PatrolLogic(env)

    def OmnivoreBehavior(self, env):
        """Define the behavior of an omnivore agent."""
        # Prioritize carnivorous feeding first
        if self.closest_agent_distance < self.stats["attack_range"] * self.life_factor:
            if self.debug:
                print(
                    f"[{self.name}] Attacking agent: Agent is within attack range ({self.closest_agent_distance})."
                )
            self.Attack(env, self.closest_agent)
        elif self.closest_food_distance < self.stats["bite_range"] * self.life_factor:
            if self.debug:
                print(
                    f"[{self.name}] Eating food: Food is within feeding range ({self.closest_food_distance})."
                )
            self.Eat(env)
        elif (
            self.closest_agent_distance
            < self.stats["eyesight_range"] * self.life_factor
        ):
            if self.debug:
                print(
                    f"[{self.name}] Moving towards agent: Agent is within eyesight range ({self.closest_agent_distance})."
                )
            self.Move(
                self.MovementVector(GetAgentVector(self, self.closest_agent)), env
            )
        elif (
            self.closest_food_distance < self.stats["eyesight_range"] * self.life_factor
        ):
            if self.debug:
                print(
                    f"[{self.name}] Moving towards food: Food is within eyesight range ({self.closest_food_distance})."
                )
            self.Move(self.MovementVector(GetFoodVector(self, self.closest_food)), env)
        elif self.memory > 0:
            if self.debug:
                print(f"[{self.name}] Recalling food or agent location from memory.")
            if self.rememberedAgent:
                self.memory -= 1
                toward_agent = GetAgentVector(self, self.rememberedAgent)
                toward_agent[2] = self.speed * 2
                self.Move(self.MovementVector(toward_agent), env)
            elif self.rememberedFood:
                self.memory -= 1
                toward_food = GetFoodVector(self, self.rememberedFood)
                toward_food[2] = self.speed * 2
                self.Move(self.MovementVector(toward_food), env)
        else:
            self.PatrolLogic(env)

    def PatrolLogic(self, env):
        """Handle patrol behavior for all types of agents."""
        if self.patrolPoint == [-1, -1, -1]:
            if self.debug:
                print(f"[{self.name}] No patrol point found: Assigning a new one.")
            seabed_depth = env.map[int(self.x), int(self.y)]
            target_depth = max(self.depth_min, min(self.depth_max, seabed_depth - 3))
            self.patrolPoint = [env.gridRInt("x"), env.gridRInt("y"), target_depth]
        distToPatrol = GetDistance(
            self.x,
            self.y,
            self.depth,
            self.patrolPoint[0],
            self.patrolPoint[1],
            self.patrolPoint[2],
        )
        if distToPatrol <= 1:
            if self.debug:
                print(f"[{self.name}] Reached patrol point: Assigning a new one.")
            self.patrolPoint = [
                env.gridRInt("x"),
                env.gridRInt("y"),
                float(env.params["min_depth"] - 3),
            ]
        else:
            if self.debug:
                print(
                    f"[{self.name}] Patrolling to current patrol point at {self.patrolPoint}."
                )
            self.Move(self.MovementVector(GetFoodVector(self, self.patrolPoint)), env)

    def BreedingBehavior(self, env):
        """Define the behavior for breeding."""
        if self.debug:
            print(f"[{self.name}] Choosing Breeding Behavior: Agent is ready to breed.")
        if self.possible_mate is not None:
            if self.debug:
                print(
                    f"[{self.name}] Moving towards a visible mate: {self.possible_mate}."
                )
            mate = env.agents[self.possible_mate]
            self.Move(self.MovementVector(GetAgentVector(self, mate)), env)
            if VectorMagnitude(GetAgentVector(self, mate)) < 3:
                if self.debug:
                    print(f"[{self.name}] Close to mate: Initiating mating process.")
                self.Mate(env, mate)
        else:
            if self.debug:
                print(f"[{self.name}] No mate visible: Switching to stadnard behavior.")
            self.HerbivoreBehavior(env)

    def Mate(self, env, partner):
        if not self.IsMate(env, partner):
            return  # Ensure both agents are valid mates
        print("Mating")

        # Generate a new genome for the offspring
        mutationFactor = env.params["mutation_factor"]
        newGenome = self.NewGenome(env, partner, mutationFactor)

        # Create the offspring agent
        offspring_name = f"{self.name}-{partner.name}-child_{env.timestep}"
        env.createNewAgent(
            (self.x + partner.x) / 2,
            (self.y + partner.y) / 2,
            (self.depth + partner.depth) / 2,
            offspring_name,
            True,
            newGenome,
        )

        # Reduce food to simulate resource cost of breeding
        self.food -= self.stats["stomach_size"] / 3
        partner.food -= partner.stats["stomach_size"] / 3

        # Reset breeding timers
        self.egg_permitted = 0
        partner.egg_permitted = 0

    def Move(self, vector, env):
        self.food += self.movement_cost * env.params["movement_cost_factor"]
        self.x += vector[0]
        self.y += vector[1]
        self.depth += vector[2]

    def Eat(self, env):
        self.food += env.params["food_value"]
        if self.food > self.stomach_size:
            self.food = self.stomach_size
        env.foods.remove(self.closest_food)
        env.generateNewFood()

    def Attack(self, env, target):
        self.food += target.GetDamaged(env, self.stats["bite_damage"])
        if self.food > self.stomach_size:
            self.food = self.stomach_size

    def GetDamaged(self, env, amount):
        damage = amount - self.stats["armor"]
        if damage > 0:
            self.health -= damage
        else:
            return 0
        self.CheckVitals(env)
        return damage

    def ApplyDepthDamage(self, env):
        """Apply damage if the agent is outside its depth tolerance range."""
        if self.depth < self.depth_min:
            damage = (self.depth_min - self.depth) / self.stats[
                "depth_tolerance_range"
            ]  # Damage scales with distance below min depth
            self.GetDamaged(env, damage)
        elif self.depth > self.depth_max:
            damage = (self.depth - self.depth_max) / self.stats[
                "depth_tolerance_range"
            ]  # Damage scales with distance above max depth
            self.GetDamaged(env, damage)

    def CheckVitals(self, env):
        if self.health <= 0:
            self.Die(env)
        if self.food <= 0:
            self.Die(env)
        if self.life > self.lifespan:
            self.Die(env)

    def Die(self, env):
        if self.debug:
            print("Agent death")
        print("Agent death")
        env.KillAgent(self.name)

    def IsBreeding(self, env):
        if (
            self.egg_permitted >= self.lifespan * 0.3
            and self.food >= self.stomach_size / 2
        ):
            return True
        return False

    def IsMate(self, env, agent):
        if self.stats["type"] == agent.stats["type"] and agent.IsBreeding(env):
            return True
        return False

    def NewGenome(self, env, partner, mutationFactor):
        newGenome = {}
        for gene, x1 in self.stats.items():
            # Skip if the gene is a string
            if isinstance(x1, str):
                if np.random.random() < 0.5:
                    newGenome[gene] = x1
                else:
                    newGenome[gene] = partner.stats[gene]
                continue

            # Convert integer to float for processing
            if isinstance(x1, int):
                x1 = float(x1)
                x2 = float(partner.stats[gene])

                # Process only numeric types
                inheritance_weight = np.random.random()

                # Calculate base value between parents using weight
                base_value = x1 * inheritance_weight + x2 * (1 - inheritance_weight)

                # Calculate standard deviation for normal distribution
                std_dev = abs(x2 - x1) * mutationFactor

                # Generate mutation using normal distribution
                mutation = np.random.normal(0, std_dev)

                # Calculate new value with mutation
                new_value = base_value + mutation

                # Ensure the value doesn't fall below 10% of the parent's values
                min_allowed = min(x1, x2) * 0.1

                # Round to integer for integer inputs
                newGenome[gene] = int(max(new_value, min_allowed))

            # For float values, do the full floating-point calculation
            elif isinstance(x1, float):
                x2 = partner.stats[gene]

                inheritance_weight = np.random.random()

                # Calculate base value between parents using weight
                base_value = x1 * inheritance_weight + x2 * (1 - inheritance_weight)

                # Calculate standard deviation for normal distribution
                std_dev = abs(x2 - x1) * mutationFactor

                # Generate mutation using normal distribution
                mutation = np.random.normal(0, std_dev)

                # Calculate new value with mutation
                new_value = base_value + mutation

                # Ensure the value doesn't fall below 10% of the parent's values
                min_allowed = min(x1, x2) * 0.1
                newGenome[gene] = max(new_value, min_allowed)

        return newGenome
