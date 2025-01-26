# **Part 1: Model Description**

## **1. Introduction**

Our simulation aims to model an **agent-based aquatic ecosystem** in which multiple fish species (herbivores, carnivores, omnivores) attempt to coexist. Agent-based modeling lets us capture unique behaviors, such as movement, feeding patterns, reproduction, and depth preferences, all driven by individual stats. The overarching goal is to see how predator–prey interactions emerge, especially when an invasive is introduced into the ecosystem.

Unlike purely mathematical models (e.g., simple Lotka-Volterra equations), this agent-based approach lets each fish act on local decisions, memory, or other constraints (like armor or speed).

---

## **2. Agent Types & Roles**

1. **Herbivores**
   - Survive by eating plants that spawn/spread around the map.
   - Typically moderate in speed/health, lower bite damage, and rely heavily on the availability of plants.
2. **Carnivores**
   - Attack and eat other fish for food instead of plants.
   - Tend to have **higher speed**, **stronger bite**, and a moderate to high statistics potential. They cost more to move around (movement cost).
3. **Omnivores**
   - Can survive on both plants and weaker fish.
   - They present an interesting dynamic: they can act like herbivores if carnivore prey is scarce, or like carnivores if plants get scarce. This can keep them quite adaptable but also potentially disruptive.
4. **Invasive Species**
   - A specialized fish type (often also carnivorous or omnivorous) introduced into a previously balanced ecosystem.
   - The question is **how** their arrival impacts the balance among other species: do they wipe out existing species, do they settle into equilibrium, or do they die off?

---

## **3. Agent Stats (Genome)**

Each agent has a **genome** that defines its numeric attributes. These attributes are typically chosen from min/max ranges at creation:

- **speed** - Movement per timestep. Faster fish can chase prey or escape predators more easily but pay a higher movement cost.
- **health**

  Maximum health. A fish with high health resists starvation or damage longer. Health is typically in the hundreds, so we keep its movement cost factor quite small.

- **stomach_size** - How much food the fish can store internally. A bigger stomach reduces starvation chances but mildly increases movement costs.
- **armor** - Reduces incoming damage from an attack. High armor helps carnivores remain dominant or herbivores survive attacks. It’s also moderate in cost.
- **bite_damage** - How hard the fish can bite. High bite damage leads to more lethal attacks, but that also increases movement cost.
- **eyesight_range** - How far the fish can detect potential prey, predators, or plants. Larger range is helpful but has a small cost.
- **feed_range** - The distance within which a herbivore can “eat” plants. Typically small, requiring the fish to be close.
- **bite_range** - The distance at which a carnivore or omnivore can successfully land a bite on another fish.
- **memory** - Number of steps an agent can remember a target (food or predator) after losing direct line-of-sight. A bigger memory can help chase prey or recall threats but adds minimal movement cost.
- **depth_point** and **depth_tolerance_range** - Where a fish _prefers_ to live (e.g., 20 m below surface) and how flexible it is about going deeper or shallower. Going outside that range leads to depth damage.
- **lifespan** and **egg_lifespan_required** - The fish’s overall maximum life expectancy (randomly drawn around some mean) and how long (or what fraction of lifespan) it must live before breeding is allowed. Agents die of old age once `life > lifespan`.

---

## **4. The Environment & Core Parameters**

1. **Map**
   - Defined by `mapsize`, e.g. 128×128 or up to 1200×1200 for a large aquatic area.
   - A depth map loaded from an image (grayscale). The environment normalizes pixel intensities to represent deeper or shallower regions.
   - `min_depth` and `max_depth` define the overall depth range of the entire environment.
2. **Food (Plants)**
   - A certain **`starting_plant_population`** seeds the map. Plants **grow** at a rate `plant_growth_speed` and can **spread** new seeds at intervals of `plant_spread_interval` if they’ve reached `plant_minimum_growth_percentage`.
   - Each spread event adds up to `plant_spread_amount` new plants around the parent in a radius `plant_spread_radius`.
   - A **global** limit `max_plants_global` caps how many plants can exist, preventing explosive growth.
3. **Agents**
   - We can define one or more species with custom parameter ranges. E.g., `hb1` for herbivores, `cv1` for carnivores, an invasive species, etc.
   - `starting_population` indicates how many fish spawn initially (or we define population in each species dictionary).
   - `movement_cost_factor` multiplies an agent’s negative movement cost each time it moves, so faster or more heavily “equipped” fish pay more.
   - `egg_incubation_time` (if used) sets how long new spawns or eggs take before they become active.
   - `mutation_factor` controls random variation when two fish breed, so offspring can differ slightly.
4. **Time & Execution**
   - The simulation runs up to `max_timesteps` or until all fish die. Each step:
     1. Agents sense surroundings (RayCast, etc.),
     2. Agents move/eat/attack/breed,
     3. Plants grow or spread,
     4. Depth damage is applied if fish stray too far from `depth_point`.
   - We can optionally **render** the environment in a PyGame window (human mode), or run it headless for speed.

---

## **5. Simulation Mechanics**

1. **Movement & Movement Cost**
   - Each step, a fish decides on an **action** (chase prey, find plants, run from predators, or patrol).
   - That movement adds a negative cost to its internal food store. Speed is key but drains resources faster if the agent invests heavily in it.
2. **Feeding**
   - **Herbivores** check if plants are within `feed_range`. If yes, they gain energy proportional to plant growth percentage.
   - **Carnivores/Omnivores** search for agent targets within `bite_range`. If they succeed, they deal damage and gain food.
   - Omnivores can also switch to plant-eating if no prey is in sight.
3. **Depth**
   - Agents live within `[depth_min, depth_max]` from their genome. If they exceed it (like going below `depth_min` or above the seabed), they take incremental damage each step.
4. **Breeding**
   - For breeding, an agent typically needs:
     - Enough time lived (`egg_permitted >= egg_lifespan_required * lifespan`),
     - Enough stored food (≥ 50% of stomach),
     - A mate of the same `type` (herbivore or carnivore or omnivore).
   - The child inherits a **blended** set of stats with random mutation.
5. **Death & Logging**
   - Agents die from:
     1. Starvation (food ≤ 0),
     2. Health ≤ 0 (injuries, repeated attacks),
     3. Old Age (life exceeds random `lifespan`).
   - The environment logs the cause of death and the timestep, letting us analyze later who died of hunger vs. old age vs. predator attacks.
6. **Invasive Species**
   - Introduced as a separate species with possibly higher speed or bigger `bite_damage`, or even omnivorous feeding.
   - We watch how they spread or dominate existing species if they adapt better to available resources or outcompete local fish.

---

## **6. Quick Reference: Key Parameters**

| **Parameter**            | **Description**                                                              | **Typical Range**  |
| ------------------------ | ---------------------------------------------------------------------------- | ------------------ |
| **mapsize**              | Dimension of the square map in cells (width = height)                        | 128–1200           |
| **min_depth, max_depth** | Global min/max for water depth in this environment                           | 10–70              |
| **starting_population**  | Base # of agents (or defined per species)                                    | ~10–60 each        |
| **food_value**           | Energy a fully grown plant provides                                          | ~200               |
| **movement_cost_factor** | Multiplies agent’s negative movement cost per step                           | 0.01–0.1           |
| **max_plants_global**    | Overall plant cap; no new plants spawn above this count                      | 200–1000+          |
| **plant_spread_amount**  | How many new plants spawn per spread event                                   | 2–10               |
| **plant_spread_radius**  | Max radius around the plant’s position for new spawns                        | 400–800+           |
| **egg_incubation_time**  | Steps an egg needs before the agent is active (if used)                      | 100–500 (optional) |
| **mutation_factor**      | Strength of random variation in offspring stats                              | 0.5–2.0            |
| **species**              | Dictionary of species definitions (herbivore, carnivore, omnivore, invasive) | Custom logic       |

---

## **7. Conclusion**

With these definitions, our **agent-based** ecosystem can exhibit a range of outcomes: stable predator–prey cycles, chaotic booms and crashes, or near extinction if an **invasive** fish outperforms local species. By adjusting **parameters** (e.g., `speed`, `armor`, `plant_spread_amount`), we can explore scenarios like:

- **Two species coexisting** vs. one out-competing the other,
- **Lotka-Volterra–style** population oscillations,
- **Invasive species** introduction leading to disruption or total takeover.

The next sections will delve deeper into how we built, tested, and analyzed these interactions.
