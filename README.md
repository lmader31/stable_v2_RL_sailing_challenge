# REINFORCEMENT LEARNING SAILING CHALLENGE (v2)

![Sailing Environment](illustration_challenge.png)

## Getting Started

Clone this repository to your local machine:

```bash
git clone https://github.com/lmader31/RL_project_sailing_v2.git
cd RL_project_sailing_v2
```

## Challenge Overview

Your mission is to develop an intelligent agent capable of navigating a sailboat from a starting point to a destination under varying wind conditions.

**IMPORTANT**: You must submit a **pre-trained agent** (a fixed policy mapping observations to actions), NOT a learning algorithm. Your agent should make decisions based on the current observation without further learning during evaluation.

### The Environment

| Parameter | Value |
|-----------|-------|
| Grid size | 128 x 128 |
| Start position | Bottom center (64, 0) |
| Goal position | Top center (64, 127) |
| Obstacle | Island in the center (rectangle + triangular prow pointing south) |
| Max episode length | 500 steps |
| Discount factor | 0.995 |
| Reward | 100 on reaching the goal, 0 otherwise |
| Actions | 9 (N, NE, E, SE, S, SW, W, NW, Stay) |

The environment features:
- A 128x128 grid with an **island obstacle** in the center (a composite shape with a triangular prow pointing toward the start)
- Realistic wind fields that vary spatially and temporally
- Physics-based boat movement influenced by wind direction and sailing efficiency
- **Crashing into the island** freezes the boat in place for the rest of the episode (reward = 0)

### Train / Test Split

You are provided with **3 public training wind scenarios** (`training_1`, `training_2`, `training_3`) with different spatial wind patterns. All three share the same temporal dynamics.

A **hidden test wind scenario** is used for final evaluation on Codabench. You cannot evaluate on it locally — your only option is to submit your agent to Codabench.

**This is the core RL challenge**: train on known environments, generalize to unseen conditions.

## Installation

We recommend using a virtual environment:

```bash
python -m venv sailing-env
source sailing-env/bin/activate   # macOS/Linux
# sailing-env\Scripts\activate    # Windows

pip install -r requirements.txt
```

#### Using Conda

```bash
conda create -n sailing-env python=3.10
conda activate sailing-env
pip install -r requirements.txt
```

## Notebooks

Explore the notebooks in order:

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `challenge_walkthrough.ipynb` | Environment mechanics, observation space, wind, island |
| 2 | `design_agent.ipynb` | Agent design: rule-based and Q-learning examples |
| 3 | `validate_agent.ipynb` | Check your agent meets the required interface |
| 4 | `evaluate_agent.ipynb` | Evaluate performance across training wind scenarios |
| 5 | `visualize_agent.ipynb` | Visualize trajectories, multi-agent races, GIF export |

## Submission Instructions

Submissions are made through **Codabench**. The link to the challenge will be shared separately.

### How to Submit

1. Create a `.py` file with a class named `MyAgent` inheriting from `BaseAgent`
2. **Important** — use this import in your submission file:
   ```python
   from evaluator.base_agent import BaseAgent
   ```
   (This is different from the local import `from agents.base_agent import BaseAgent`)
3. Implement the required methods: `act(observation)`, `reset()`, `seed(seed)`
4. **Create a ZIP archive** with your `.py` file **at the root** (not inside a subfolder) and upload to Codabench

```python
from evaluator.base_agent import BaseAgent

class MyAgent(BaseAgent):
    def act(self, observation):
        # Return an integer in [0, 8]
        ...
    
    def reset(self):
        ...
    
    def seed(self, seed=None):
        ...
```

### Creating the ZIP file

**Common mistake**: zipping a folder instead of the file itself. The `.py` file must be at the **root** of the archive, not inside a subfolder.

```bash
# Correct — zip the file directly:
zip my_submission.zip my_agent.py

# WRONG — this creates a subfolder inside the zip:
zip -r my_submission.zip my_agent_folder/
```

You can verify your zip is correct with:
```bash
unzip -l my_submission.zip
# Should show:  my_agent.py   (NOT  some_folder/my_agent.py)
```

### Local Validation and Evaluation

Validate your agent:

```bash
cd src
python3 test_agent_validity.py agents/your_agent.py
```

Evaluate on a training scenario:

```bash
cd src
python3 evaluate_submission.py agents/your_agent.py --wind_scenario training_1 --seeds 1 --num-seeds 10 --verbose
```

Evaluate on all training scenarios:

```bash
cd src
python3 evaluate_submission.py agents/your_agent.py --seeds 1 --num-seeds 10
```

## Challenge Timeline

See the Codabench challenge page for dates and deadlines.

## Communication

Email: t.rahier at criteo.com | Codabench forum
