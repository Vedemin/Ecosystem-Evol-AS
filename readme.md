**Project Report: Evolutionary Algorithms in an Underwater Ecosystem Simulation**

### Introduction

The aim of this project is to develop an agent-based simulation of an underwater ecosystem, with evolutionary processes. The simulation models interactions between multiple species, including herbivores, carnivores, and omnivores, each exhibiting distinct behaviors. A primary objective is to simulate evolution and mutation, enabling the creation of new traits and behaviors through breeding. Additionally, the system is designed to support the introduction of new species and behaviors dynamically, ensuring scalability and adaptability.

This report outlines the goals, implementation, functionality, and conclusions drawn from the project.

### Objectives

1. **Agent-Based Simulation** : Create a dynamic underwater ecosystem with autonomous agents representing different species.
2. **Evolution and Mutation** : Implement genetic inheritance mechanisms to allow traits to pass from parents to offspring, introducing mutations for diversity.
3. **Behavioral Diversity** : Simulate a variety of behaviors, including feeding, breeding, patrolling, and survival mechanisms, specific to species types.
4. **Scalability** : Allow the system to accommodate new species and behaviors with minimal changes.
5. **Ecological Insights** : Observe emergent patterns and interactions.

### Implementation Details

#### **Agent Behaviors**

Agents are classified into three primary types based on their feeding behavior:

- **Herbivores** : Consume plant matter and patrol the environment searching for food.
- **Carnivores** : Attack other agents for sustenance, using patrol and memory-based tracking of prey.
- **Omnivores** : Balance feeding between plant matter and prey, prioritizing based on proximity and opportunity.

Each agent’s behavior is dictated by:

- **Environment Interaction** : Agents perceive their surroundings using parameters like eyesight range, bite range, and memory capabilities.
- **Patrolling** : Agents without immediate targets randomly patrol the environment, simulating exploration.
- **Breeding** : Mating occurs when agents meet specific food and health thresholds, initiating offspring creation.
- **Survival** : Agents are subject to depth-based damage, health constraints, and starvation risks, ensuring a natural life cycle.

#### **Evolution and Mutation**

The genetic inheritance model is the core of the evolutionary process:

- Traits (e.g., eyesight range, bite damage, depth tolerance) are passed from parents to offspring using weighted averages and random mutations.
- Mutations introduce variability, calculated using a normal distribution centered on parental traits, allowing gradual evolutionary shifts.
- Safeguards prevent trait degradation by ensuring minimum viable values for inherited characteristics.

#### **System Components**

1. **Environment** : The environment simulates an underwater 3D world with varying depths, food sources, and environmental constraints.
2. **Agents** : Each agent is initialized with:

- Species-specific traits.
- Dynamic attributes such as health, food levels, and memory.
- Life events (e.g., birth, movement, feeding, breeding, and death).

1. **Simulation Logic** : Centralized logic governs agent interactions, environmental updates, and statistics tracking.
2. **Scalability Features** : The modular design allows new species and behaviors to be introduced by defining new traits and extending behavior classes.

#### **Functionalities**

**Feeding Mechanisms** :

- Herbivores consume plants.
- Carnivores attack other agents within range.
- Omnivores switch between feeding modes based on availability.

**Breeding** :

- Breeding triggers the creation of offspring with inherited and mutated traits.
- Each offspring’s genome is a combination of parent traits and environmental influence - mutation factor.

**Memory and Navigation** :

- Agents track recent encounters with food or prey, influencing future movements.
- Patrolling behavior is implemented to simulate exploration.

**Survival Dynamics** :

- Depth-based damage encourages agents to stay within optimal depth ranges.
- Health and food levels dictate life expectancy.
- Agents run away from danger.

### Observations and Conclusions

#### **Emergent Behaviors**

1. **Population Dynamics** :

- Overpopulation of a particular species led to resource depletion, naturally regulating population sizes.
- Predation created a balance between herbivores and carnivores, reflecting real-world predator-prey relationships.

1. **Rising evolution metric** :

- Over time the movement cost directly tied to agent statistics was rising - survival of the fittest.

#### **Evolutionary Insights**

1. **Trait Optimization** :

- Traits like eyesight range and bite damage evolved over generations, improving agent survival and reproductive success.
- Mutations occasionally resulted in extreme values, but stabilizing selection ensured population viability.

1. **Speciation Potential** :

- Introducing high mutation rates led to significant deviations in offspring traits, hinting at speciation opportunities.
- New species can be modeled by setting unique initial traits and observing their integration into the ecosystem.

### Conclusion

This project demonstrates the potential of evolutionary algorithms in simulating complex ecological systems. The integration of adaptive behaviors, genetic inheritance, and mutation showcases how evolution drives diversity and survival. The system's scalability and modularity position it as a good start for studying evolutionary processes and developing artificial ecosystems.
