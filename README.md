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

In our project, a DQN agent is trained to optimize the Lunar Lander pod using reinforcement learning.

### Key Features:

- Q-Network: A neural network model to approximate the Q-value function.
- Experience Replay: A mechanism to store past experiences and sample them during training.
- Target Network: A separate network to provide stable target values for training.
- Epsilon-Greedy Policy: A policy that balances exploration and exploitation during training.

## Environment Setup

To recreate the environment and run the code, please follow these steps:

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/modishre/lunarlandingproj.git
cd lunarlandingproj
```

### 2. Create a Virtual Environment (This is optional but recommended)

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

### Visualization

When training the agent, the episode and reward are printed to the console.
When testing, a GUI will appear to watch the agent live.

## Running the Code

### 1. Testing the Agent

To test a pre-trained agent, ensure you have a saved model (`trainedModel.pth`), then run the `lunarLander.py` script:

If you want to test our pre-trained model, make sure that the test_agent function is called in the script and the train_agent is commented out!!!!
**The train_agent function is commented out to prevent retraining the model.**

This will load the trained model and run it on the environment for 100 episodes, displaying the agent's performance.

### 2. Training the Agent

To train the agent, **uncomment the train_agent() function on line 359** and run the `lunarLander.py` script:

**#### THIS WILL OVERRIDE OUR CURRENT SAVED MODEL!**

```bash
python3 lunarLander.py
```

Our script will initialize the environment, train the agent, and save the trained model periodically. It will also generate plots showing the training progress, including total rewards and losses per episode.

## Repository Structure

```bash
lunar-lander-dqn/
│
├── lunarLander.py             # Main script to train and test the DQN agent
├── requirements.txt           # Lists all the dependencies required for the project
├── README.md                  # Detailed instructions on how to set up and run the project
└── trainedmodel.pth           # Our pre-trained model file (the one in the repo is ours generated after training)
```

## Troubleshooting

- PyTorch Installation Issues: If you face issues with PyTorch installation, make sure you're installing the correct version compatible with your system's CUDA version.
- Gym Rendering Issues: On some systems, especially headless servers, rendering might not work. You can use alternative rendering modes (like 'rgb_array') or skip rendering during training.

## Credits

- Project Authors: Shreya Modi and Alex Smith

---
