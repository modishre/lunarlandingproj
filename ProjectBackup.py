import time
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





# Define the Q-Network architecture
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        # Define the neural network layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        # Define the forward pass
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, target_update_freq=10):
        self.state_dim = state_dim  # Dimension of the state space
        self.action_dim = action_dim  # Dimension of the action space
        self.gamma = gamma  # Discount factor for future rewards
        self.epsilon = epsilon  # Initial exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for exploration
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.target_update_freq = target_update_freq  # Frequency of target network updates
        self.q_network = QNetwork(state_dim, action_dim)  # Initialize the Q-Network
        self.target_network = QNetwork(state_dim, action_dim)  # Initialize the Target Network
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)  # Optimizer for the Q-Network
        self.memory = deque(maxlen=10000)  # Experience replay buffer
        self.update_target_network()  # Initialize target network weights to match Q-network weights
    
    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            # print(f' joe!!!!!!!!!{type(state)} and the action space {self.action_dim}')
            print('random chosen')
            return np.random.choice(self.action_dim)  # Exploration: random action
        
        state = torch.tensor(state).unsqueeze(0)
        # print(f' here!!!!!!!!!{type(state)} ')
        q_values = self.q_network(state)
        print("not random")
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

        # print(f' here {type(states)} of {states}')
        # Debug: Check the shapes of the sampled states
        # print(f"Sampled states shape: {[state.shape for state in states]}")
        # print(f"Sampled next_states shape: {[next_state.shape for next_state in next_states]}")
        # for i, state in enumerate(states):
        #     print(f"State {i} shape: {np.array(state).shape}")
        # for i, next_state in enumerate(next_states):
        #     print(f"Next state {i} shape: {np.array(next_state).shape}")
        # Convert lists to tensors
        # Convert lists to numpy arrays
        # states = np.array(states)
        # actions = np.array(actions)
        # rewards = np.array(rewards)
        # next_states = np.array(next_states)
        # dones = np.array(dones)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)


        # Debug: Print shapes of the tensors
        print(f"states shape: {states.shape}")
        print(f"actions shape: {actions.shape}")
        print(f"rewards shape: {rewards.shape}")
        print(f"next_states shape: {next_states.shape}")
        print(f"dones shape: {dones.shape}")
        

        # Debug: Print shapes of the tensors
        # print(f"states shape: {states.shape}")
        # print(f"actions shape: {actions.shape}")
        # print(f"rewards shape: {rewards.shape}")
        # print(f"next_states shape: {next_states.shape}")
        # print(f"dones shape: {dones.shape}")
        
        # Compute Q-values of the current states
        q_values = self.q_network(states).gather(1, actions)
        
        # Compute Q-values of the next states from the target network
        next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
        
        # Compute the target Q-values
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute the loss between predicted Q-values and target Q-values
        loss = nn.MSELoss()(q_values, target_q_values)

        # Perform backpropagation and optimize the Q-network
        self.optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Backpropagate the loss
        self.optimizer.step()  # Update the Q-network's weights
        
        # Decay epsilon to reduce exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def update_target_network(self):
        # Copy the weights from the Q-Network to the Target Network
        self.target_network.load_state_dict(self.q_network.state_dict())


warnings.filterwarnings("ignore", category=DeprecationWarning)
env = gym.make("LunarLander-v2", render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim)

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 1001
else:
    num_episodes = 101



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

    if episode % agent.target_update_freq == 0:
        agent.update_target_network()

    # if episode % 100 == 0:
    if episode % 20 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")
        time.sleep(3)
