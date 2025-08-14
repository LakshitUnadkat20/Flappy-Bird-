# Flappy Bird AI – Genetic Algorithm & NEAT Approach

## Overview
This project implements an **AI agent** that learns to play *Flappy Bird* autonomously using a combination of:
- **Genetic Algorithms (GA)** for optimizing neural network weights
- **NEAT (NeuroEvolution of Augmenting Topologies)** for evolving network architecture over generations

The AI starts with simple decision-making abilities and gradually improves through **evolutionary learning**. Over multiple generations, it develops strategies to survive longer and achieve higher scores in the game.

---

## Features
- **Fully functional Flappy Bird game** built in Python using `pygame`
- **AI-controlled gameplay** via feedforward neural networks
- **Genetic Algorithm** with:
  - Tournament selection
  - Two-point crossover
  - Mutation with adjustable rates
- **NEAT integration** for evolving both weights and topology
- **Real-time performance statistics** displayed in-game:
  - Current score
  - Generation number
  - Best score
  - Number of alive agents
- **Post-training analysis** with:
  - Score vs. generation plot
  - Performance statistics table

---

## How It Works
1. **Game Environment**  
   - Physics-based bird movement (gravity & flapping mechanics)  
   - Randomly generated pipes with a fixed vertical gap  
   - Collision detection with pipes and boundaries  

2. **Neural Network Inputs**  
   - Bird’s vertical position (relative to window height)  
   - Vertical distance to the next pipe’s gap  
   - Horizontal distance to the next pipe  

3. **Decision Making**  
   - Single output neuron (sigmoid activation)  
     - Output > 0.5 → Bird jumps  
     - Output ≤ 0.5 → Bird does nothing  

4. **Learning Process**  
   - Each bird is controlled by a neural network  
   - GA optimizes weights based on a **fitness function**:  
     ```
     Fitness = (Survival Time × 0.1) + Pipes Passed − Penalty for Collision
     ```
   - NEAT evolves network topology over generations  

---

## Installation
### Prerequisites
- Python 3.7+
- `pygame`
- `neat-python`
- `numpy`
- `matplotlib`
- `pandas`

Install dependencies:
```bash
pip install pygame neat-python numpy matplotlib pandas
