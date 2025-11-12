import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import networkx as nx
from typing import Dict, Any, List, Tuple, Optional
from collections import deque
import random


def create_small_world_topology(input_dim: int, hidden_size: int, output_dim: int, 
                                k: int, p: float, seed: int) -> Tuple[nx.DiGraph, List[int], List[int]]:
"""
Generate the small-world network topology as a single connected graph.

Args:
        input_dim: Number of input nodes
        hidden_size: Number of hidden nodes
        output_dim: Number of output nodes
        k: Number of neighbors in ring lattice (must be < hidden_size)
        p: Rewiring probability
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (topology graph, input_nodes list, output_nodes list)
    """
    # Bounds checking
    if k >= hidden_size:
        raise ValueError(f"k ({k}) must be less than hidden_size ({hidden_size})")
    if k < 2:
        raise ValueError(f"k ({k}) must be at least 2 for ring lattice")
    
    rng = np.random.RandomState(seed)
total_nodes = input_dim + hidden_size + output_dim

G = nx.DiGraph()
G.add_nodes_from(range(total_nodes))

hidden_start = input_dim
hidden_end = hidden_start + hidden_size

# Create initial ring lattice structure for hidden nodes (directed, acyclic)
for i in range(hidden_start, hidden_end):
    # Only add edges to higher-indexed hidden nodes to maintain acyclicity
    for j in range(1, k // 2 + 1):
        target = hidden_start + ((i - hidden_start + j) % hidden_size)
        if target > i and target < hidden_end:  # Only add forward edges within hidden layer
            G.add_edge(i, target)

# Rewire edges with probability p (maintaining acyclicity)
for edge in list(G.edges()):
    if rng.random() < p:
        # Remove the edge
        G.remove_edge(*edge)
        # Add a new random edge (only to higher-indexed hidden nodes)
            # Error handling: ensure we can find a valid target
            if edge[0] + 1 < hidden_end:
                max_attempts = 100
                attempts = 0
        new_node = rng.randint(edge[0] + 1, hidden_end)
                while G.has_edge(edge[0], new_node) and attempts < max_attempts:
            new_node = rng.randint(edge[0] + 1, hidden_end)
                    attempts += 1
                if attempts < max_attempts:
        G.add_edge(edge[0], new_node)
                # If we can't find a valid edge, just skip rewiring for this edge

# Add connections from input nodes to hidden nodes
if input_dim is not None:
    for input_node in range(input_dim):
        for hidden_node in range(hidden_start, hidden_start + min(k, hidden_size)):
            G.add_edge(input_node, hidden_node)

# Add connections from hidden nodes to output nodes
if output_dim is not None:
    for output_node in range(hidden_end, total_nodes):
        for hidden_node in range(hidden_end - min(k, hidden_size), hidden_end):
            G.add_edge(hidden_node, output_node)

    # Ensure topology is a DAG
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("FFN requires a Directed Acyclic Graph (DAG) topology")

input_nodes = list(range(input_dim))
output_nodes = list(range(input_dim + hidden_size, input_dim + hidden_size + output_dim))

    return G, input_nodes, output_nodes


class TopologyNetwork(nn.Module):
    """
    PyTorch module for a feedforward network with custom topology.
    Maintains interpretability through manual topological forward pass while
    using PyTorch's autograd for efficient training.
    """
    
    def __init__(self, topology: nx.DiGraph, input_nodes: List[int], 
                 output_nodes: List[int], seed: int = 47, activation: str = 'leaky_relu'):
        """
        Initialize the topology network.
        
        Args:
            topology: NetworkX DiGraph representing the network topology
            input_nodes: List of input node indices
            output_nodes: List of output node indices
            seed: Random seed for weight initialization
            activation: Activation function ('leaky_relu' or 'tanh')
        """
        super(TopologyNetwork, self).__init__()
        
        self.topology = topology
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.node_order = list(nx.topological_sort(topology))

# Set seed for reproducible weight initialization
np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Convert node_states to nn.Parameters for automatic gradient computation
        self.biases = nn.ParameterDict()
        self.weights = nn.ParameterDict()

for node in list(topology.nodes()):
            # Initialize bias as a parameter
            bias_value = np.random.normal(0, 0.1)
            self.biases[str(node)] = nn.Parameter(torch.tensor(bias_value, dtype=torch.float32))
            
            # Initialize weights for incoming edges (predecessors)
            for neighbor in topology.predecessors(node):
                weight_value = np.random.normal(0, 0.1)
                param_name = f"{neighbor}_to_{node}"
                self.weights[param_name] = nn.Parameter(torch.tensor(weight_value, dtype=torch.float32))
        
        # Set activation function
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the topology network.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim] (Q-values)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize activations dictionary
        activations = {}
        
        # Set input node activations
        for i, input_node in enumerate(self.input_nodes):
            activations[input_node] = x[:, i]
        
        # Initialize all other nodes to zero
        for node in self.topology.nodes():
            if node not in activations:
                activations[node] = torch.zeros(batch_size, device=device)
        
        # Process through network in topological order
        for node in self.node_order:
            if node not in self.input_nodes:
                # Get bias
                bias = self.biases[str(node)]
                
                # Sum weighted inputs from predecessors
                weighted_sum = torch.full((batch_size,), bias.item(), dtype=torch.float32, device=device)
                
                for neighbor in self.topology.predecessors(node):
                    weight_param = self.weights[f"{neighbor}_to_{node}"]
                    weighted_sum = weighted_sum + activations[neighbor] * weight_param
                
                # Apply activation function
                activations[node] = self.activation(weighted_sum)
        
        # Collect output node activations
        output_values = []
        for output_node in sorted(self.output_nodes):
            output_values.append(activations[output_node])
        
        # Stack into [batch_size, output_dim] tensor
        return torch.stack(output_values, dim=1)


class ExperienceReplayBuffer:
    """
    Simple experience replay buffer for DQN training.
    Stores transitions and provides random batch sampling.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def store(self, state: np.ndarray, action: int, reward: float, 
              next_state: np.ndarray, done: bool):
        """
        Store a transition in the replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                               torch.Tensor, torch.Tensor]:
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN agent that uses a custom topology network for Q-value estimation.
    """
    
    def __init__(self, network: TopologyNetwork, learning_rate: float = 0.001, 
                 gamma: float = 0.99, device: str = 'cpu'):
        """
        Initialize the DQN agent.
        
        Args:
            network: TopologyNetwork instance for the main network
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            device: Device to run on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.gamma = gamma
        
        # Main network
        self.main_network = network.to(self.device)
        
        # Target network (copy of main network)
        self.target_network = TopologyNetwork(
            network.topology, network.input_nodes, network.output_nodes,
            seed=47, activation='leaky_relu'
        ).to(self.device)
        self.update_target_network()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=learning_rate)
    
    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            epsilon: Exploration probability
            
        Returns:
            Selected action
        """
        num_actions = len(self.main_network.output_nodes)
        if random.random() < epsilon:
            # Random action
            return random.randint(0, num_actions - 1)
        else:
            # Greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.main_network(state_tensor)
                return q_values.argmax().item()
    
    def train_step(self, replay_buffer: ExperienceReplayBuffer, batch_size: int) -> float:
        """
        Perform one training step.
        
        Args:
            replay_buffer: Experience replay buffer
            batch_size: Batch size for training
            
        Returns:
            Loss value
        """
        if len(replay_buffer) < batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute current Q-values: Q(s, a)
        current_q_values = self.main_network(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values: r + γ * max(Q_target(s', a')) * (1 - done)
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + self.gamma * max_next_q_values * (~dones).float()
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Copy main network parameters to target network."""
        self.target_network.load_state_dict(self.main_network.state_dict())


def train_dqn(env_name: str = "CartPole-v1", hidden_size: int = 64, k: int = 8, 
              p: float = 0.1, seed: int = 47, num_episodes: int = 500,
              max_steps_per_episode: int = 500, batch_size: int = 32,
              learning_rate: float = 0.001, gamma: float = 0.99,
              epsilon_start: float = 1.0, epsilon_end: float = 0.01,
              epsilon_decay: float = 0.995, target_update_freq: int = 10,
              replay_buffer_size: int = 10000, min_buffer_size: int = 1000,
              device: str = 'cpu'):
    """
    Train a DQN agent with custom topology network.
    
    Args:
        env_name: Gymnasium environment name
        hidden_size: Number of hidden nodes
        k: Number of neighbors in ring lattice
        p: Rewiring probability
        seed: Random seed
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        batch_size: Batch size for training
        learning_rate: Learning rate
        gamma: Discount factor
        epsilon_start: Initial exploration probability
        epsilon_end: Final exploration probability
        epsilon_decay: Epsilon decay factor per episode
        target_update_freq: Frequency of target network updates (in episodes)
        replay_buffer_size: Size of replay buffer
        min_buffer_size: Minimum buffer size before training starts
        device: Device to run on
        
    Returns:
        Dictionary with training history
    """
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Create environment
    env = gym.make(env_name)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    
    # Get dimensions
    from stable_baselines3.common.preprocessing import get_flattened_obs_dim
    input_dim = int(get_flattened_obs_dim(env.observation_space))
    action_space = env.action_space
    if hasattr(action_space, 'n'):
        output_dim = int(action_space.n)
    else:
        raise ValueError("Action space not properly configured")
    
    # Create topology
    topology, input_nodes, output_nodes = create_small_world_topology(
        input_dim, hidden_size, output_dim, k, p, seed
    )
    
    # Create network
    network = TopologyNetwork(topology, input_nodes, output_nodes, seed=seed)
    
    # Create agent
    agent = DQNAgent(network, learning_rate=learning_rate, gamma=gamma, device=device)
    
    # Create replay buffer
    replay_buffer = ExperienceReplayBuffer(capacity=replay_buffer_size)
    
    # Training history
    episode_rewards = []
    episode_losses = []
    epsilon = epsilon_start
    
    print(f"Starting DQN training on {env_name}")
    print(f"Network topology: {len(topology.nodes())} nodes, {len(topology.edges())} edges")
    print(f"Input dim: {input_dim}, Output dim: {output_dim}, Hidden size: {hidden_size}")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = 0
        num_train_steps = 0
        
        for step in range(max_steps_per_episode):
            # Select action
            action = agent.select_action(state, epsilon)
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            replay_buffer.store(state, action, reward, next_state, done)
            
            # Train if buffer is large enough
            if len(replay_buffer) >= min_buffer_size:
                loss = agent.train_step(replay_buffer, batch_size)
                episode_loss += loss
                num_train_steps += 1
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Update target network periodically
        if episode % target_update_freq == 0:
            agent.update_target_network()
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Record history
        episode_rewards.append(episode_reward)
        if num_train_steps > 0:
            episode_losses.append(episode_loss / num_train_steps)
        else:
            episode_losses.append(0.0)
        
        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_loss = np.mean(episode_losses[-50:]) if episode_losses else 0.0
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Loss: {avg_loss:.4f} | "
                  f"Epsilon: {epsilon:.3f}")
    
    env.close()
    
    return {
        'episode_rewards': episode_rewards,
        'episode_losses': episode_losses,
        'network': agent.main_network
    }


if __name__ == "__main__":
    # DQN hyperparameters
    dqn_learning_rate = 0.001
    dqn_gamma = 0.99
    dqn_batch_size = 32
    num_episodes = 500
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    target_update_freq = 10
    replay_buffer_size = 10000
    min_buffer_size = 1000
    
    # Topology parameters
    network_type = "small_world"
    seed = 47
    hidden_size = 64
    k = 8
    p = 0.1
    
    # Training
    print("=" * 60)
    print("DQN Training with Custom Small-World Topology")
    print("=" * 60)
    
    history = train_dqn(
        env_name="CartPole-v1",
        hidden_size=hidden_size,
        k=k,
        p=p,
        seed=seed,
        num_episodes=num_episodes,
        batch_size=dqn_batch_size,
        learning_rate=dqn_learning_rate,
        gamma=dqn_gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        target_update_freq=target_update_freq,
        replay_buffer_size=replay_buffer_size,
        min_buffer_size=min_buffer_size,
        device='cpu'
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final average reward (last 50 episodes): {np.mean(history['episode_rewards'][-50:]):.2f}")
    print(f"Best episode reward: {max(history['episode_rewards']):.2f}")
