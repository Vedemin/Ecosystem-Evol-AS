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
    "depth_point": 20.0,
    "depth_tolerance_range": 15.0,
}


class Agent:
    def __init__(
        self, name, position, genome=default_genome, avg_lifespan=5000, life_start_point=0, debug=False
    ):
        """
        Initializes an Agent object with specific attributes such as position, genome, and starting parameters. 
        Sets up the agent's life, speed, health, depth range, and initial food levels.

        Parameters:
        - name: Unique name or identifier for the agent.
        - position: A list or tuple specifying the agent's starting x, y, and depth coordinates.
        - genome: A dictionary containing genetic traits of the agent, such as speed, health, and feeding range.
        - avg_lifespan: Average lifespan of the species, used to calculate specific lifespan for the agent.
        - life_start_point: Proportion of life already completed at initialization (0 to 1).
        - debug: Boolean flag to enable or disable debug messages for the agent.
        """
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
        self.life_factor = self.LifeFactor(None)
        self.speed = self.stats["speed"] * self.life_factor
        self.stomach_size = self.stats["stomach_size"] * self.life_factor
        self.food = self.stomach_size / 3
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
        self.breeding_lifespan_threshold = self.stats["egg_lifespan_required"] * avg_lifespan
        self.patrolPoint = [-1, -1, -1]
        self.invisibleVictim = None
        self.invisibleVictimMemory = 0

    def LifeFactor(self, env):
        """
        Calculates a life factor that scales certain attributes (e.g., speed, stomach size) based on the agent's current age. 
        The factor varies over the agent's lifespan, with reduced efficiency at the start and end of life.

        Parameters:
        - env: The simulation environment, required for interaction with the agent.

        Returns:
        - Float: The life factor, ranging from 0.001 to 1.0, depending on the agent's life stage.
        """
        if self.life >= self.lifespan:
            self.CheckVitals(env)

        t1 = 0.2 * self.lifespan
        t2 = 0.6 * self.lifespan

        if 0 <= self.life < t1:
            return 0.5 + (0.5 * (self.life / t1))
        elif t1 <= self.life < t2:
            return 1.0
        elif t2 <= self.life < self.lifespan:
            return 1.0 - (0.9 * ((self.life - t2) / (self.lifespan - t2)))
        return 0.001

    def Activate(self, env):
        """
        Performs the agent's actions during a simulation step, including calculating distances, selecting actions, 
        applying depth damage, and updating the agent's state.

        Parameters:
        - env: The simulation environment containing other agents, plants, and environmental settings.
        """
        if self.name not in env.agentNames:
            env.agentNames.append(self.name)
        self.life_factor = self.LifeFactor(env)
        self.speed = self.stats["speed"] * self.life_factor
        self.stomach_size = self.stats["stomach_size"] * self.life_factor
        if self.debug:
            print(f"Activate {self.name}")
        self.CalculateDistances(env)
        self.SelectAction(env)
        self.ApplyDepthDamage(env)
        self.egg_permitted += 1
        self.life += 1

        if self.invisibleVictimMemory > 0:
            self.invisibleVictimMemory -= 1
            if self.invisibleVictimMemory <= 0:
                self.invisibleVictim = None

    def CalculateDistances(self, env):
        """
        Calculates distances between the agent and other agents or food in its vicinity. Determines visible entities based 
        on the agent's eyesight range and updates the closest food, potential mates, and threats.

        Parameters:
        - env: The simulation environment containing other agents and plants.
        """
        eyesight_range = self.stats["eyesight_range"] * self.life_factor

        for name, agent in env.agents.items():
            if name == self.name or name == self.invisibleVictim:
                continue
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

            self.distances[agent.name] = {"distance": distance, "visible": visible}

        self.visible_agents = {
            name: data for name, data in self.distances.items() if data["visible"] and name in env.agents and env.agents[name].stats["species"] != self.stats["species"]
        }

        self.same_species = {
            name: data for name, data in self.distances.items() if data["visible"] and name in env.agents and env.agents[name].stats["species"] == self.stats["species"]
        }

        self.sorted_agents = sorted(
            self.visible_agents.keys(), key=lambda name: self.visible_agents[name]["distance"]
        )

        self.sorted_friendly = sorted(
            self.same_species.keys(), key=lambda name: self.same_species[name]["distance"]
        )

        if (len(self.sorted_agents) > 0):
            self.closest_agent = env.agents[self.sorted_agents[0]]
            self.closest_agent_distance = self.visible_agents[self.sorted_agents[0]]["distance"]
        else:
            self.closest_agent_distance = 100000
            self.closest_agent = None

        if self.IsBreeding(env):
            self.possible_mate = next(
                (
                    agent
                    for agent in self.sorted_friendly
                    if agent in env.agentNames and self.IsMate(env, env.agents[agent])
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
                    if self.debug:
                        print(f"[{self.name}] Memory depleted. Forgetting agent.")
                    self.closest_agent_distance = 100000
                    self.closest_agent = None
                    self.remembered_agent = None
            return
        
        if -10 < self.memory <= 0 or self.closest_food not in env.foods:
            if self.debug:
                print(
                    f"[{self.name}] Forgetting closest food: Memory {self.memory} or food no longer available."
                )
            self.memory = -10
            self.closest_food_distance = 100000
            self.closest_food = [
                self.closest_food_distance,
                self.closest_food_distance,
                self.closest_food_distance,
            ]

        if self.stats["type"] == "herbivore":
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
                    )

            if self.closest_food_distance > eyesight_range:
                if self.debug:
                    print(
                        f"[{self.name}] Closest food {self.closest_food} is out of eyesight range. Reducing memory {self.memory}."
                    )
                self.memory = max(
                    0, self.memory - 1
                )
                if self.memory == 0:
                    if self.debug:
                        print(f"[{self.name}] Memory depleted. Forgetting food.")
                    self.closest_food_distance = 100000
                    self.closest_food = [100000, 100000, 100000]

    def MovementVector(self, vector):
        """
        Scales a movement vector to ensure that its magnitude does not exceed the agent's speed.

        Parameters:
        - vector: A list or tuple representing the x, y, and depth components of the movement.

        Returns:
        - List: A scaled movement vector within the agent's speed limit.
        """
        vector_mag = VectorMagnitude(vector)
        if vector_mag <= self.speed:
            return vector
        result = NormalizedVector(vector)
        return [result[0] * self.speed, result[1] * self.speed, result[2] * self.speed]

    def SelectAction(self, env):
        """
        Chooses the appropriate action for the agent based on its current state, such as running away, breeding, or foraging.

        Parameters:
        - env: The simulation environment containing agents, plants, and environmental settings.
        """
        if self.IsThreatened(env):
            self.RunAway(env)
        elif self.IsBreeding(env):
            self.BreedingBehavior(env)
        else:
            self.AgentBehavior(env)
        self.CheckVitals(env)

    def IsThreatened(self, env):
        """
        Determines whether the agent is under threat from nearby carnivores or omnivores.

        Parameters:
        - env: The simulation environment containing agents and their attributes.

        Returns:
        - Boolean: True if the agent is threatened, otherwise False.
        """
        if self.type != "herbivore":
            return False
            
        for agent_name in self.sorted_agents:
            if agent_name not in env.agents:
                continue
                
            threat = env.agents[agent_name]
            if (threat.type in ["carnivore", "omnivore"] and 
                threat.stats["species"] != self.stats["species"] and
                self.visible_agents[agent_name]["distance"] < self.stats["eyesight_range"] * self.life_factor):
                return True
                
        return False

    def RunAway(self, env):
        """
        Executes a fleeing behavior to move the agent away from the closest threat. Calculates a vector away from the threat 
        and moves the agent accordingly, staying within valid depth ranges.

        Parameters:
        - env: The simulation environment containing agents and environmental depth data.
        """
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
            
        away_vector = [
            self.x - closest_threat.x,
            self.y - closest_threat.y,
            self.depth - closest_threat.depth
        ]
        magnitude = (away_vector[0]**2 + away_vector[1]**2 + away_vector[2]**2)**0.5
        if magnitude > 0:
            away_vector = [
                away_vector[0] / magnitude * self.speed,
                away_vector[1] / magnitude * self.speed,
                away_vector[2] / magnitude * self.speed
            ]
        new_x = self.x + away_vector[0]
        new_y = self.y + away_vector[1]
        new_depth = self.depth + away_vector[2]
        new_x = max(0, min(new_x, env.params["mapsize"] - 1))
        new_y = max(0, min(new_y, env.params["mapsize"] - 1))
        seabed_depth = env.map[int(new_x), int(new_y)]
        min_depth = max(env.params["min_depth"], self.depth_min)
        max_depth = min(seabed_depth - 1, self.depth_max)
        new_depth = max(min_depth, min(new_depth, max_depth))
        
        final_vector = [
            new_x - self.x,
            new_y - self.y,
            new_depth - self.depth
        ]
        self.Move(final_vector, env) 

    def HerbivoreBehavior(self, env):
        """
        Defines the behavior of herbivores, including eating plants, moving toward food, and recalling food locations from memory. 
        If no food is found, the agent defaults to patrolling.

        Parameters:
        - env: The simulation environment containing plants and environmental settings.
        """
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
        """
        Defines the general behavior for an agent based on its type (herbivore, carnivore, or omnivore). 
        Calls the corresponding behavior function for the agent type.

        Parameters:
        - env: The simulation environment containing agents and environmental settings.
        """
        if self.type == "herbivore":
            self.HerbivoreBehavior(env)
        elif self.type == "carnivore":
            self.CarnivoreBehavior(env)
        elif self.type == "omnivore":
            self.OmnivoreBehavior(env)

    def CarnivoreBehavior(self, env):
        """
        Defines the behavior of carnivores, including attacking prey, moving toward visible agents, and recalling prey locations from memory. 
        If no prey is found, the agent defaults to patrolling.

        Parameters:
        - env: The simulation environment containing agents and environmental settings.
        """
        if self.closest_agent_distance < self.stats["bite_range"] * self.life_factor and self.stomach_size - self.food < self.stats["bite_damage"]:
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
        """
        Defines the behavior of omnivores, which includes eating plants, attacking prey, moving toward food or agents, and recalling 
        locations from memory. Omnivores alternate between herbivore and carnivore behaviors based on food availability.

        Parameters:
        - env: The simulation environment containing agents, plants, and environmental settings.
        """
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
        """
        Handles patrol behavior for agents when no specific action (e.g., eating, attacking, breeding) is required. 
        Assigns new patrol points and moves the agent toward them.

        Parameters:
        - env: The simulation environment containing agents and environmental settings.
        """
        if self.patrolPoint == [-1, -1, -1]:
            if self.debug:
                print(f"[{self.name}] No patrol point found: Assigning a new one.")
            new_pt = self.findValidPatrolPoint(env)
            if new_pt == [-1, -1, -1]:
                if self.debug:
                    print(f"[{self.name}] Could not find valid patrol point. Skipping patrol.")
                return
            else:
                self.patrolPoint = new_pt

        distToPatrol = GetDistance(
            self.x, self.y, self.depth,
            self.patrolPoint[0], self.patrolPoint[1], self.patrolPoint[2]
        )

        if distToPatrol <= 1:
            if self.debug:
                print(f"[{self.name}] Reached patrol point: Assigning a new one.")
            new_pt = self.findValidPatrolPoint(env)
            if new_pt == [-1, -1, -1]:
                if self.debug:
                    print(f"[{self.name}] No valid new patrol point found. Staying put.")
                return
            self.patrolPoint = new_pt

        else:
            if self.debug:
                print(f"[{self.name}] Patrolling to current patrol point {self.patrolPoint}")
            self.Move(self.MovementVector(GetFoodVector(self, self.patrolPoint)), env)

    def BreedingBehavior(self, env):
        """
        Defines the behavior for agents ready to breed. Moves the agent toward a visible mate or switches to a default behavior 
        if no mate is found.

        Parameters:
        - env: The simulation environment containing agents and environmental settings.
        """
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
        """
        Handles the mating process between two agents, generating offspring with a blended genome and mutation. 
        Consumes food resources for both agents and resets their breeding timers.

        Parameters:
        - env: The simulation environment containing agents and environmental settings.
        - partner: The agent's breeding partner.
        """
        if not self.IsMate(env, partner):
            return

        mutationFactor = env.params["mutation_factor"]
        newGenome = self.NewGenome(env, partner, mutationFactor)

        amount = env.species[self.stats["species"]]["spawn_count"]
        if amount > 1:
            amount = np.random.randint(1, env.species[self.stats["species"]]["spawn_count"])
        for i in range(amount):
            offspring_name = f"{self.stats['species']}-{env.species_counter[self.stats['species']]}-{i}"
            env.createNewAgent(
                (self.x + partner.x) / 2,
                (self.y + partner.y) / 2,
                (self.depth + partner.depth) / 2,
                offspring_name,
                True,
                newGenome,
            )

        self.food -= self.stats["stomach_size"] / 3
        partner.food -= partner.stats["stomach_size"] / 3

        self.egg_permitted = 0
        partner.egg_permitted = 0

    def Move(self, vector, env):
        """
        Moves the agent by applying the given movement vector and deducts the movement cost from the agent's food supply.

        Parameters:
        - vector: A list or tuple representing the movement in x, y, and depth directions.
        - env: The simulation environment containing agents and environmental settings.
        """
        self.food += self.movement_cost * env.params["movement_cost_factor"]
        self.x += vector[0]
        self.y += vector[1]
        self.depth += vector[2]

    def Eat(self, env):
        """
        Consumes a plant located at the agent's current position. Increases the agent's food and health based on the plant's 
        growth percentage and removes the plant from the environment.

        Parameters:
        - env: The simulation environment containing plants and their attributes.
        """
        growth_percentage = self.closest_food[3]
        food_energy = env.params["food_value"] * growth_percentage

        self.food += food_energy
        if self.food > self.stomach_size:
            self.food = self.stomach_size

        health_gain = (food_energy / self.stomach_size) * self.stats["health"]
        self.health += health_gain
        if self.health > self.stats["health"]:
            self.health = self.stats["health"]

        if self.closest_food in env.foods:
            env.foods.remove(self.closest_food)

    def Attack(self, env, target):
        """
        Executes an attack on a target agent. Transfers food from the target to the attacking agent and marks the target as an 
        "invisible victim" to prevent immediate re-attack.

        Parameters:
        - env: The simulation environment containing agents.
        - target: The agent being attacked.
        """
        self.food += target.GetDamaged(env, self.stats["bite_damage"])
        if self.food > self.stomach_size:
            self.food = self.stomach_size

        if self.invisibleVictim != target.name:
            self.invisibleVictim = target.name
            self.invisibleVictimMemory = 20

    def GetDamaged(self, env, amount):
        """
        Applies damage to the agent based on the specified amount, reduced by the agent's armor. 
        If the agent's health drops to zero or below, it triggers the death process.

        Parameters:
        - env: The simulation environment containing agents.
        - amount: The amount of damage to apply.

        Returns:
        - Float: The actual amount of damage taken by the agent.
        """
        damage = amount - self.stats["armor"]
        if damage > 0:
            self.health -= damage
        else:
            return 0
        self.CheckVitals(env)
        return damage

    def ApplyDepthDamage(self, env):
        """
        Applies incremental damage to the agent if it moves outside its depth tolerance range. 
        The damage is proportional to the distance from the depth limit.

        Parameters:
        - env: The simulation environment containing depth data.
        """
        if self.depth < self.depth_min:
            damage = (self.depth_min - self.depth) / self.stats[
                "depth_tolerance_range"
            ]
            self.GetDamaged(env, damage)
        elif self.depth > self.depth_max:
            damage = (self.depth - self.depth_max) / self.stats[
                "depth_tolerance_range"
            ]
            self.GetDamaged(env, damage)

    def CheckVitals(self, env):
        """
        Checks the agent's vital statistics, including health, food levels, and lifespan. 
        Triggers the death process if any of these drop below critical thresholds.

        Parameters:
        - env: The simulation environment containing agents and environmental settings.
        """
        if self.health <= 0:
            if self.debug:
                print(f"[{self.name}] Died due to health=0")
            self.Die(env, "health=0")
            
        if self.food <= 0:
            if self.debug:
                print(f"[{self.name}] Died due to starvation")
            self.Die(env, "starved")
            
        if self.life > self.lifespan:
            if self.debug:
                print(f"[{self.name}] Died of old age")
            self.Die(env, "old age")

    def Die(self, env, reason="unknown"):
        """
        Handles the death of the agent, removing it from the environment and logging the reason for its death.

        Parameters:
        - env: The simulation environment containing agents.
        - reason: The cause of death (e.g., starvation, old age, or attack).
        """
        env.KillAgent(self.name, reason)


    def IsBreeding(self, env):
        """
        Determines if the agent is ready to breed based on its food levels and breeding timer.

        Parameters:
        - env: The simulation environment containing agents.

        Returns:
        - Boolean: True if the agent is ready to breed, otherwise False.
        """
        if (
            self.egg_permitted >= self.breeding_lifespan_threshold * 0.3
            and self.food >= self.stomach_size / 2
        ):
            return True
        return False

    def IsMate(self, env, agent):
        """
        Determines if another agent is a suitable mate for breeding based on species and readiness.

        Parameters:
        - env: The simulation environment containing agents.
        - agent: The potential mate to evaluate.

        Returns:
        - Boolean: True if the agent is a suitable mate, otherwise False.
        """
        if self.stats["type"] == agent.stats["type"] and agent.IsBreeding(env):
            return True
        return False

    def NewGenome(self, env, partner, mutationFactor):
        """
        Generates a new genome for offspring by blending the genomes of the parent agents and applying mutations.

        Parameters:
        - env: The simulation environment containing agents.
        - partner: The other parent agent.
        - mutationFactor: The strength of random variation applied to the offspring's genome.

        Returns:
        - Dictionary: The new genome for the offspring.
        """
        newGenome = {}
        for gene, x1 in self.stats.items():
            if isinstance(x1, str):
                if np.random.random() < 0.5:
                    newGenome[gene] = x1
                else:
                    newGenome[gene] = partner.stats[gene]
                continue

            if isinstance(x1, int):
                x1 = float(x1)
                x2 = float(partner.stats[gene])
                inheritance_weight = np.random.random()
                base_value = x1 * inheritance_weight + x2 * (1 - inheritance_weight)
                std_dev = abs(x2 - x1) * mutationFactor
                mutation = np.random.normal(0, std_dev)
                new_value = base_value + mutation
                min_allowed = min(x1, x2) * 0.1
                newGenome[gene] = int(max(new_value, min_allowed))

            elif isinstance(x1, float):
                x2 = partner.stats[gene]
                inheritance_weight = np.random.random()
                base_value = x1 * inheritance_weight + x2 * (1 - inheritance_weight)
                std_dev = abs(x2 - x1) * mutationFactor
                mutation = np.random.normal(0, std_dev)
                new_value = base_value + mutation
                min_allowed = min(x1, x2) * 0.1
                newGenome[gene] = max(new_value, min_allowed)

        return newGenome

    def findValidPatrolPoint(self, env, max_tries=100):
        """
        Attempts to find a valid patrol point within the environment that satisfies the agent's depth constraints.

        Parameters:
        - env: The simulation environment containing depth data.
        - max_tries: The maximum number of attempts to find a valid patrol point.

        Returns:
        - List: A list containing the x, y, and depth of the patrol point, or [-1, -1, -1] if no valid point is found.
        """
        for _ in range(max_tries):
            rx = env.gridRInt("x")
            ry = env.gridRInt("y")
            seabed_depth = env.map[int(rx), int(ry)]

            local_min_depth = max(self.depth_min, env.params["min_depth"])
            local_max_depth = min(self.depth_max, seabed_depth - 1)

            if local_max_depth > local_min_depth:
                chosen_depth = np.random.uniform(local_min_depth, local_max_depth)
                return [rx, ry, chosen_depth]

        return [-1, -1, -1]

    def to_dict(self):
        """
        Converts the agent's attributes into a dictionary for logging or serialization purposes.

        Returns:
        - Dictionary: A dictionary representation of the agent's attributes and current state.
        """
        return {
            "name": self.name,
            "stats": self.stats,
            "health": self.health,
            "lifespan": self.lifespan,
            "life_factor": self.life_factor,
            "speed": self.speed,
            "stomach_size": self.stomach_size,
            "food": self.food,
            "type": self.type,
            "depth_min": self.depth_min,
            "depth_max": self.depth_max,
            "movement_cost": self.movement_cost,
            "x": self.x,
            "y": self.y,
            "depth": self.depth,
            "egg_permitted": self.egg_permitted,
        }
