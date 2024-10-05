# Quadrupedal Locomotion RL Pybullet Environment

A custom OpenAI Gym environment designed for training reinforcement learning agents to perform quadrupedal locomotion tasks.
![Alt text](videos/rl-video-model29-final2_20241001-230411-episode-0-ezgif.com-video-to-gif-converter.gif)

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
   - [Prerequisites](#prerequisites)
   - [Installation Steps](#installation-steps)
4. [Usage](#usage)
   - [Training](#training)
   - [Evaluation](#evaluation)
   - [Visualisation](#visualisation)
5. [Contributing](#contributing)
6. [License](#license)
7. [Credits](#credits)

---

## Overview

This project provides a custom OpenAI Gym environment for simulating quadrupedal locomotion using reinforcement learning in pybullet. It supports various RL algorithms (PPO, SAC, etc.) using Stable Baselines3.

---

## Features

- **Custom OpenAI Gym Environment**: Tailored for quadrupedal locomotion tasks.
- **Multiple RL Algorithms**: Supports PPO, SAC, and other algorithms via Stable Baselines3.

---

## Installation

### Prerequisites

- Python 3.7+
- PyTorch (with appropriate CUDA version if running on GPU)
- Gym (OpenAI Gym environment)

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/nbaron17/quadruped_RL_pybullet
   cd quadruped_RL_pybullet
2. **Set up virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
4. **Install the Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Training

To train the agent:
```bash
python scripts/train_agent.py --config config/config.yaml
```

### Evaluation

To evaluate a pre-trained agent:
```bash
python scripts/evaluate_agent.py --models/latest_model.zip
```

### Visualisation

You can visualize the training process using TensorBoard. Run the following command to launch TensorBoard:
```bash
tensorboard --logdir logs/
```

---

## Contributing

Contributions are welcome! If you'd like to contribute to the project, please follow these steps:
1. Fork the repository.
2. Create a new branch (e.g., feature/my-new-feature).
3. Make your changes.
4. Submit a pull request.


---


## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for more details.


---

## Credits

This project makes use of several open-source libraries:

- Stable Baselines3
- OpenAI Gymnasium
- The urdf file for the quadruped and some other code sourced from [this repo](https://github.com/miguelasd688/4-legged-robot-model/tree/PureSimulation_V1.0).
