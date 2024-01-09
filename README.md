# Quadruped Robot using Reinforcement Learning

Welcome to the Quadruped Robot using Reinforcement Learning project! This repository contains the code and resources for training and evaluating a quadruped robot using reinforcement learning techniques, specifically the Augmented Random Search (ARS) algorithm.

## Overview

This project focuses on training a quadruped robot to navigate and adapt in its environment through the application of reinforcement learning. The physical robot is represented through a 3D-printed model, and essential components are acquired to enable both simulation and potential physical testing.

### Key Objectives

- Train a reinforcement learning agent using the ARS algorithm to control the quadruped robot.
- Investigate the impact of various training configurations on the agent's performance.
- Evaluate the adaptability and navigation capabilities of the trained agent.

## Implementation Details

### Physical Quadruped Robot

Before delving into the reinforcement learning aspects, a physical quadruped robot is created by 3D printing the model and acquiring necessary components, establishing a tangible representation for both simulation and physical testing.

#### 3D Printing
The quadruped robot model is replicated using 3D printing technology, providing a tangible representation for reinforcement learning experiments.

#### Components Acquisition
Various components, including motors and sensors, are acquired to build the physical quadruped robot, forming the hardware foundation for its movements.

### Evaluation Scripts
  
- **Evaluation Script (`spot_ars_eval.py`):** Allows for the evaluation of a pre-trained agent, providing options for visualization and data saving. Command-line arguments enable flexibility in evaluation configurations.

## Project Overview

The project is structured to encompass the physical robot creation, reinforcement learning training, and subsequent evaluation of the trained agent's performance.

### Reinforcement Learning

#### Training Steps:

1. **Initialize Environment:** Set up the simulation environment for the quadruped robot.
2. **3D Print Model & Acquire Components:** Physically replicate the quadruped robot using 3D printing and acquire necessary components for the hardware.
3. **Initialize Agent:** Create a reinforcement learning agent, initializing its policy and normalizer.
4. **Training Loop:** Utilize the ARS algorithm for efficient training, periodically saving the trained model.

#### Evaluation Steps:

1. **Initialize Environment:** Set up the simulation environment for the quadruped robot.
2. **3D Print Model & Acquire Components:** Physically replicate the quadruped robot using 3D printing and acquire necessary components for the hardware.
3. **Load Pre-Trained Model:** Load the pre-trained reinforcement learning model for evaluation.
4. **Evaluation Loop:** Execute the policy to generate actions, observe states and rewards, and record actions and states during each episode.

## Getting Started

1. Clone this repository: `git clone https://github.com/begad-tamim/Quadruped-Robot-using-Reinforcement.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Follow the instructions in the implementation details to run training and evaluation scripts.
