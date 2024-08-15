# AI driven lunar descent

## Table of Contents

1. [Project Overview](#project-overview)
2. [Environment Setup](#environment-setup)
3. [Dependencies](#dependencies)
4. [Running the Code](#running-the-code)
5. [Repository Structure](#repository-structure)
6. [Troubleshooting](#troubleshooting)
7. [Credits](#credits)

## Project Overview

In this project, a DQN agent is trained to play the Lunar Lander game using reinforcement learning. The agent is built using PyTorch and utilizes experience replay and a target network to stabilize learning.

### Key Features:

- Q-Network: A neural network model to approximate the Q-value function.
- Experience Replay: A mechanism to store past experiences and sample them during training.
- Target Network: A separate network to provide stable target values for training.
- Epsilon-Greedy Policy: A policy that balances exploration and exploitation during training.

## Environment Setup

To recreate the environment and run the code, follow these steps:

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/lunar-lander-dqn.git
cd lunar-lander-dqn
```

### 2. Create a Virtual Environment (Optional but Recommended)

It's recommended to create a virtual environment to manage dependencies:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies

All the necessary packages are listed in the `requirements.txt` file. To install them, use:

```bash
pip install -r requirements.txt
```

### 4. Install Additional Dependencies (If Needed)

In some environments, especially if using a GPU, you might need to manually install the appropriate version of PyTorch:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Dependencies

The primary dependencies for this project include:

- Python 3.7+
- PyTorch
- Gym
- Matplotlib
- Numpy

All dependencies are listed in the `requirements.txt` file.

## Running the Code

### 1. Training the Agent

To train the agent, run the `main.py` script:

```bash
python main.py
```

This script will initialize the environment, train the agent, and save the trained model periodically. It will also generate plots showing the training progress, including total rewards and losses per episode.

### 2. Testing the Agent

To test a pre-trained agent, ensure you have a saved model (`trainedmodel.pth`), then run the script:

```bash
python main.py --test
```

This will load the trained model and run it on the environment for a few episodes, displaying the agent's performance.

## Repository Structure

```bash
lunar-lander-dqn/
│
├── main.py                    # Main script to train and test the DQN agent
├── qnetwork.py                # Defines the Q-network architecture
├── dqn_agent.py               # Defines the DQN agent class and its methods
├── requirements.txt           # Lists all the dependencies required for the project
├── README.md                  # Detailed instructions on how to set up and run the project
└── trainedmodel.pth           # Pre-trained model file (generated after training)
```

## Troubleshooting

- PyTorch Installation Issues: If you face issues with PyTorch installation, make sure you're installing the correct version compatible with your system's CUDA version.
- Gym Rendering Issues: On some systems, especially headless servers, rendering might not work. You can use alternative rendering modes (like 'rgb_array') or skip rendering during training.

## Credits

- Project Authors: [Your Name] and [Collaborator's Name]
- Tools Used: OpenAI Gym, PyTorch, Matplotlib, Numpy

If you encounter any issues or have questions, please feel free to open an issue in the repository or contact us at [your email address].

---

Feel free to modify the sections as necessary to match your specific project details. This README file should give clear instructions on setting up the environment, installing dependencies, and running the code, which is crucial for anyone who wants to use or build upon your work.
