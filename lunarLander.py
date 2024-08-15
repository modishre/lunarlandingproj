
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import warnings
import matplotlib.pyplot as plt

# Define the Q-Network architecture
class QNetwork(nn.Module):
    """
    Neural network model for approximating the Q-value function in DQN.
    
    Args:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        hidden_dim (int): Number of neurons in the hidden layers.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Second fully connected layer
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # Output layer

    def forward(self, x):
        """
        Defines the forward pass of the Q-network.
        
        Args:
            x (torch.Tensor): Input state tensor.
        
        Returns:
            torch.Tensor: Output Q-values for each action.
        """
        x = torch.relu(self.fc1(x))  # Apply ReLU activation to the first layer
        x = torch.relu(self.fc2(x))  # Apply ReLU activation to the second layer
        return self.fc3(x)  # Output layer (Q-values)

# Global variables to store loss and Q-values during training
losses = []
q_values_history = []  # To store Q-network Q-values
target_q_values_history = []  # To store target network Q-values

# Define the DQN Agent
class DQNAgent:
    """
    Deep Q-Network agent that interacts with the environment and learns to take actions.
    
    Args:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        device (str): Device to run the computations on ("cpu" or "cuda").
        lr (float): Learning rate for the optimizer.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Initial exploration rate.
        epsilon_decay (float): Decay rate for epsilon.
        epsilon_min (float): Minimum value for epsilon.
        target_update_freq (int): Frequency of target network updates.
    """
    def __init__(self, state_dim, action_dim, device="cpu", lr=0.0005, gamma=0.999, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, target_update_freq=5):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_freq = target_update_freq
        
        # Initialize Q-network and target network
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        
        # Optimizer for the Q-network
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay buffer
        self.memory = deque(maxlen=10000)
        
        # Initialize target network with the same weights as Q-network
        self.update_target_network()

    def act(self, state):
        """
        Selects an action using an epsilon-greedy policy.
        
        Args:
            state (np.array): The current state of the environment.
        
        Returns:
            int: The action selected by the agent.
        """
        if np.random.rand() < self.epsilon:
            # Exploration: choose a random action
            return np.random.choice(self.action_dim)
        else:
            # Exploitation: choose the action with the highest Q-value
            state = torch.tensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state)
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        """
        Stores an experience in the replay buffer.
        
        Args:
            state (np.array): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (np.array): The next state after taking the action.
            done (bool): Whether the episode is finished.
        """
        self.memory.append((np.array(state), action, reward, np.array(next_state), done))

    def replay(self, batch_size=64):
        """
        Trains the Q-network using a mini-batch of experiences from the replay buffer.
        
        Args:
            batch_size (int): The number of experiences to sample from the replay buffer.
        """
        if len(self.memory) < batch_size:
            return
        
        # Sample a mini-batch from the replay buffer
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
    
        # Compute Q-values of the current states
        q_values = self.q_network(states).gather(1, actions)
        
        # Compute Q-values of the next states from the target network
        next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
        
        # Compute the target Q-values
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Store Q-values for analysis
        q_values_history.append(q_values.mean().item())
        target_q_values_history.append(target_q_values.mean().item())

        # Compute the loss between predicted Q-values and target Q-values
        loss = nn.MSELoss()(q_values, target_q_values)
        losses.append(loss.item())

        # Perform backpropagation and update the Q-network's weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        """
        Decays the exploration rate epsilon after each episode.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        """
        Copies the weights from the Q-network to the target network.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, filename):
        """
        Saves the Q-network model to a file.
        
        Args:
            filename (str): The file path to save the model.
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, filename)

    def load_agent(self, filename):
        """
        Loads a saved Q-network model from a file.
        
        Args:
            filename (str): The file path to load the model from.
        """
        checkpoint = torch.load(filename, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.q_network.eval()
        print(f"Agent loaded from {filename}")

def train_agent(agent, env, num_episodes):
    """
    Trains the DQN agent in the specified environment.
    
    Args:
        agent (DQNAgent): The agent to train.
        env (gym.Env): The environment to train the agent in.
        num_episodes (int): The number of episodes to train the agent.
    """
    rewards_per_episode = []  # List to store the total reward for each episode

    for episode in range(num_episodes):
        state, _ = env.reset()  # Reset the environment and get the initial state
        total_reward = 0  # Initialize the total reward for this episode
        done = False  # Flag to indicate whether the episode is finished

        while not done:
            action = agent.act(state)  # Agent selects an action using its policy
            result = env.step(action)  # Execute the action in the environment

            # Handle the result of the action, which may vary depending on the environment's API
            if len(result) == 4:
                next_state, reward, done, info = result
            elif len(result) == 5:
                next_state, reward, done, truncated, info = result
                done = done or truncated  # Handle truncated episodes

            # Store the experience in the agent's replay buffer
            agent.remember(state, action, reward, next_state, done)
            # Train the Q-network with a mini-batch from the replay buffer
            agent.replay()

            state = next_state  # Update the current state
            total_reward += reward  # Accumulate the reward

        # Decay epsilon to reduce exploration over time
        agent.update_epsilon()
        # Record the total reward for this episode
        rewards_per_episode.append(total_reward)

        # Update the target network periodically
        if episode % agent.target_update_freq == 0:
            agent.update_target_network()

        # Save the model at 700 episodes (Model seems to peak around this area)
        if episode % 700 == 0:
            agent.save_model(f'trainedModel.pth')
        print(f'Episode {episode} reward: {total_reward}, epsilon: {agent.epsilon}')

    # Plot the total reward per episode after training
    plt.figure()
    plt.plot(rewards_per_episode, label="Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode")
    plt.grid(True)

    def smooth_rewards(rewards, window_size=10):
        """
        Smooths the reward data using a moving average.
        
        Args:
            rewards (list): List of rewards per episode.
            window_size (int): The size of the moving average window.
        
        Returns:
            np.array: Smoothed rewards.
        """
        smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        return smoothed_rewards
    
    smoothed_rewards = smooth_rewards(rewards_per_episode, window_size=10)

    # Plotting the smoothed rewards
    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_rewards, label=f"Smoothed Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Smoothed Total Reward per Episode")
    plt.legend()
    plt.grid(True)

    # Plot the loss per episode
    plt.figure()
    plt.plot(losses, label="Average loss per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Total loss per Episode")
    plt.grid(True)
    plt.show()

test_rewards_per_episode = []
def test_agent(agent, env, episodes=5):
    """
    Tests the trained DQN agent in the environment.
    
    Args:
        agent (DQNAgent): The trained agent to test.
        env (gym.Env): The environment to test the agent in.
        episodes (int): The number of episodes to test the agent.
    """
    agent.epsilon = 0  # Disable exploration (only exploit learned policy)
    test_rewards_per_episode = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        with torch.no_grad():  # Disable gradient calculations for testing
            while not done:
                env.render()  # Render the environment for visualization
                action = agent.act(state)
                result = env.step(action)

                if len(result) == 4:
                    next_state, reward, done, info = result
                elif len(result) == 5:
                    next_state, reward, done, truncated, info = result
                    done = done or truncated

                total_reward += reward
                state = next_state

        test_rewards_per_episode.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    # Plot the total test reward per episode
    plt.figure()
    plt.plot(test_rewards_per_episode, label="Total Test Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode (Test)")
    plt.grid(True)
    plt.show()



# initialize number of training episodes
num_episodes = 800

# Initialize the environment for testing
env = gym.make("LunarLander-v2")

# Set the device (CPU or GPU) for computations
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} device")

# Set the dimensions of the state and action spaces
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Initialize the DQN agent with the given state and action dimensions to the correct device
agent = DQNAgent(state_dim, action_dim, device)


##########################################

# COMMENT OUT TO NOT TRAIN AND ONLY RUN TEST
train_agent(agent, env, num_episodes)

##########################################

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize the environment for testing (Render Mode is set to human for GUI to display)
env = gym.make("LunarLander-v2", render_mode="human") 

# Set the device (CPU or GPU) for computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the DQN agent with the given state and action dimensions to the correct device
agent = DQNAgent(state_dim, action_dim, device)

# Load a pre-trained agent model from a file
agent.load_agent("trainedModel.pth")

# Test the loaded agent in the environment
test_agent(agent, env, episodes=100)


