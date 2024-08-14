import time
from turtle import st
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import gym
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt




# Define the Q-Network architecture
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64): # try making it smaller  IF THE LOSS STOPS DECREASING preventing it from converging DECREASING UPDATE target network
        super(QNetwork, self).__init__()
        print(state_dim)
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        # Define the forward pass
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

losses = []
q_values_history = []  # To store Q-network Q-values
target_q_values_history = []  
# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, device="cpu", lr=0.0005, gamma=0.999, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, target_update_freq=5):
        self.device = device
        self.state_dim = state_dim  # Dimension of the state space
        self.action_dim = action_dim  # Dimension of the action space
        self.gamma = gamma  # Discount factor for future rewards
        self.epsilon = epsilon  # Initial exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for exploration
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.target_update_freq = target_update_freq  # Frequency of target network updates
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)  # Initialize the Q-Network
        self.target_network = QNetwork(state_dim, action_dim).to(self.device) # Initialize the Target Network
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)  # Optimizer for the Q-Network
        self.memory = deque(maxlen=10000)  # Experience replay buffer
        self.update_target_network()  # Initialize target network weights to match Q-network weights
    
    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            # print(f' joe!!!!!!!!!{type(state)} and the action space {self.action_dim}')
            return np.random.choice(self.action_dim)  # Exploration: random action
        state = torch.tensor(state).unsqueeze(0).to(self.device)
        # print(f' here!!!!!!!!!{type(state)} ')
        q_values = self.q_network(state)
        return q_values.argmax().item()  # Exploitation: action with highest Q-value
    
    def remember(self, state, action, reward, next_state, done):
        # print(f'saving to memory state {state} as type {type(state)}')
        state = np.array(state)
        next_state = np.array(next_state)

        # print(f"Remembering state shape: {state.shape}, next_state shape: {next_state.shape}")
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=64):
        # Update Q-Network using mini-batch from replay buffer
        if len(self.memory) < batch_size:
            return
        
        # Sample a mini-batch from the replay buffer
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device).float()
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device).float()
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device).float()
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device).float()
    
        # Compute Q-values of the current states
        q_values = self.q_network(states).gather(1, actions)
        
        # Compute Q-values of the next states from the target network
        next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
        
        # Compute the target Q-values
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        q_values_history.append(q_values.mean().item())
        target_q_values_history.append(target_q_values.mean().item())

        # Compute the loss between predicted Q-values and target Q-values
        loss = nn.MSELoss()(q_values, target_q_values)
        losses.append(loss.item())

        # print(loss)
        # Perform backpropagation and optimize the Q-network
        self.optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Backpropagate the loss
        self.optimizer.step()  # Update the Q-network's weights

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        # Copy the weights from the Q-Network to the Target Network
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, filename="dqn_agent.pth"):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': 0,
        }, filename)

    def load_agent(self, filename="dqn_agent.pth"):
        checkpoint = torch.load(filename, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.q_network.eval()
        print(f"Agent loaded from {filename}")


rewards_per_episode = []
def train_agent(agent, env, episodes):
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            result = env.step(action)

            if len(result) == 4:
                next_state, reward, done, info = result
            elif len(result) == 5:
                next_state, reward, done, truncated, info = result
                done = done or truncated

            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            
            state = next_state
            total_reward += reward
            
        agent.update_epsilon()
        rewards_per_episode.append(total_reward)

        if episode % agent.target_update_freq == 0:
            agent.update_target_network()

        if episode % 700 == 0:
            agent.save_model(f'/Users/alexsmithlizandra/Documents/Artificial Intelligence/Project/trainedModel.pth')
        
        print(f'this episode {episode} reward was {total_reward} epsilon {agent.epsilon}')

    # agent.save_model(f'/Users/alexsmithlizandra/Documents/Artificial Intelligence/Project/trainedModel.pth')
    env.close()
    # After training, plot the average reward per episode
    plt.figure()
    plt.plot(rewards_per_episode, label="Total Reward per Episode")

    # Calculate the average reward per episode
    average_rewards_per_episode = [sum(rewards_per_episode[:i+1]) / (i+1) for i in range(len(rewards_per_episode))]

    # # Plot the average reward per episode
    # plt.figure()
    # plt.plot(average_rewards_per_episode, label="Average Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode")
    plt.grid(True)

# ADD GRAPH OF THE TESTED

    def smooth_rewards(rewards, window_size=10):
        """Smooth the reward data using a moving average."""
        smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        return smoothed_rewards

    # Assuming `rewards_per_episode` is your list of rewards
     # This should be your list of rewards per episode
    window_size = 10  # Adjust this to change the smoothing level

    smoothed_rewards = smooth_rewards(rewards_per_episode, window_size=window_size)

    # Plotting the smoothed rewards
    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_rewards, label=f"Smoothed Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Smoothed Total Reward per Episode")
    plt.legend()
    plt.grid(True)
    plt.show()








    # plt.figure()
    # plt.plot(average_rewards_per_episode, label="Average Reward per Episode")
    # plt.xlabel("Episode")
    # plt.ylabel("Total Reward")
    # plt.title("Total Reward per Episode")
    # plt.grid(True)

    # Ensure you are plotting the correct x-values
    # plt.figure()
    # # Assuming each value in q_values_history corresponds to a step
    # plt.scatter(list(range(len(q_values_history))), q_values_history, label='Q-Network Q-Values', marker='o')
    # plt.scatter(list(range(len(target_q_values_history))), target_q_values_history, label='Target Network Q-Values', marker='x')
    # plt.title('Q-Network vs. Target Network Convergence')
    # plt.xlabel('Training Steps')
    # plt.ylabel('Mean Q-Value')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    plt.figure()
    plt.plot(losses, label="Average loss per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Total loss per Episode")
    plt.grid(True)
    plt.show()





warnings.filterwarnings("ignore", category=DeprecationWarning)

num_episodes = 800

env = gym.make("LunarLander-v2")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} device")


device = torch.device("cpu")
print(f"Using {device} device")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim, device)


# train_agent(agent, env, episodes=100)




test_rewards_per_episode = []

def test_agent(agent, env, episodes=5):
    agent.epsilon = 0  # Disable exploration (epsilon-greedy strategy)
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        with torch.no_grad():
            while not done:
                env.render()
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

    # env.close()

    plt.figure()
    plt.plot(test_rewards_per_episode, label="Total Test Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode")
    plt.grid(True)
    plt.show()


warnings.filterwarnings("ignore", category=DeprecationWarning)

env = gym.make("LunarLander-v2", render_mode="human") 

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


agent = DQNAgent(state_dim, action_dim, device)
print("here")

agent.load_agent("/Users/alexsmithlizandra/Documents/Artificial Intelligence/Project/trainedModel.pth")
print('joe')

test_agent(agent, env, episodes=5)

