---
sidebar_position: 12
title: "Chapter 12: Learning and Adaptation"
---

# Chapter 12: Learning and Adaptation

## Learning Outcomes
By the end of this chapter, students will be able to:
- Implement reinforcement learning algorithms for robotic control
- Apply imitation learning techniques for skill acquisition
- Design adaptive control systems for changing environments
- Evaluate learning algorithms for humanoid robotics applications
- Implement transfer learning between simulation and reality
- Develop self-improving robotic systems

## Overview

Learning and adaptation are crucial capabilities for humanoid robots operating in dynamic, unstructured environments. Unlike traditional robots that rely on pre-programmed behaviors, learning-enabled robots can acquire new skills, adapt to changing conditions, and improve their performance over time. This chapter explores various learning paradigms and their applications in humanoid robotics, from low-level motor control to high-level decision making.

The ability to learn from experience enables humanoid robots to handle the variability and uncertainty inherent in human environments. Through learning, robots can acquire complex motor skills, adapt their behavior to individual users, and continuously improve their performance without explicit programming for every possible scenario.

## Reinforcement Learning for Robotics

### Markov Decision Processes (MDPs)

Reinforcement Learning (RL) is a powerful paradigm for learning control policies in robotics. In an MDP, an agent learns to make decisions by interacting with an environment to maximize cumulative rewards.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class RobotMDP:
    def __init__(self, state_dim, action_dim, max_steps=1000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.current_step = 0
        self.state = None

    def reset(self):
        """Reset environment to initial state"""
        self.state = self._get_initial_state()
        self.current_step = 0
        return self.state

    def step(self, action):
        """Execute action and return (next_state, reward, done, info)"""
        # Apply action to robot
        next_state = self._transition_function(self.state, action)
        reward = self._reward_function(self.state, action, next_state)
        self.current_step += 1
        done = self.current_step >= self.max_steps

        self.state = next_state
        return next_state, reward, done, {}

    def _get_initial_state(self):
        """Get initial robot state"""
        # Example: joint positions, velocities, IMU readings
        return np.random.randn(self.state_dim)

    def _transition_function(self, state, action):
        """Define state transition dynamics"""
        # Simplified robot dynamics
        next_state = state + 0.1 * action + 0.01 * np.random.randn(self.state_dim)
        return np.clip(next_state, -10, 10)  # Limit state values

    def _reward_function(self, state, action, next_state):
        """Define reward function"""
        # Example: penalize large actions, reward stability
        action_penalty = -0.01 * np.sum(np.square(action))
        stability_reward = -0.1 * np.sum(np.square(next_state[:3]))  # Reward stable position

        return action_penalty + stability_reward
```

### Deep Q-Network (DQN) for Robot Control

```python
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.network(state)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Neural networks
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay buffer
        self.memory = deque(maxlen=100000)
        self.batch_size = 32

        # Update target network
        self.update_target_network()

    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
```

### Deep Deterministic Policy Gradient (DDPG) for Continuous Control

```python
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.network(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        return self.network(torch.cat([state, action], dim=1))

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, lr_actor=1e-4, lr_critic=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        # Actor and Critic networks
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Replay buffer
        self.memory = deque(maxlen=100000)
        self.batch_size = 100

        # Noise for exploration
        self.noise_std = 0.2
        self.noise_max = 0.5

        # Update target networks
        self.update_target_networks(tau=1.0)

    def update_target_networks(self, tau=0.005):
        """Soft update target networks"""
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def add_noise(self, action):
        """Add noise to action for exploration"""
        noise = torch.randn_like(action) * self.noise_std
        noise = torch.clamp(noise, -self.noise_max, self.noise_max)
        return torch.clamp(action + noise, -self.max_action, self.max_action)

    def select_action(self, state, add_noise=True):
        """Select action based on current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_tensor)

        if add_noise:
            action = self.add_noise(action)

        return action.cpu().data.numpy().flatten()

    def train(self):
        """Train the agent on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.FloatTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch]).unsqueeze(1)
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch]).unsqueeze(1)

        # Critic update
        next_actions = self.actor_target(next_states)
        next_q_values = self.critic_target(next_states, next_actions)
        target_q_values = rewards + (0.99 * next_q_values * ~dones)

        current_q_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q_values, target_q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.update_target_networks()
```

## Imitation Learning

### Behavioral Cloning

Behavioral cloning learns to mimic expert demonstrations by treating the problem as supervised learning.

```python
class BehavioralCloning(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(BehavioralCloning, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.network(state)

class ImitationLearner:
    def __init__(self, state_dim, action_dim, lr=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.model = BehavioralCloning(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.demonstrations = []

    def add_demonstration(self, states, actions):
        """Add expert demonstration to training data"""
        for state, action in zip(states, actions):
            self.demonstrations.append((state, action))

    def train(self, epochs=100, batch_size=32):
        """Train the imitation learning model"""
        for epoch in range(epochs):
            total_loss = 0

            # Create batches
            indices = list(range(len(self.demonstrations)))
            random.shuffle(indices)

            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]

                batch_states = torch.FloatTensor([self.demonstrations[idx][0] for idx in batch_indices])
                batch_actions = torch.FloatTensor([self.demonstrations[idx][1] for idx in batch_indices])

                # Forward pass
                predicted_actions = self.model(batch_states)
                loss = self.criterion(predicted_actions, batch_actions)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if epoch % 10 == 0:
                avg_loss = total_loss / (len(self.demonstrations) // batch_size + 1)
                print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')

    def predict(self, state):
        """Predict action for given state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.model(state_tensor)
        return action.numpy().flatten()
```

### Generative Adversarial Imitation Learning (GAIL)

GAIL uses adversarial training to learn policies that are indistinguishable from expert demonstrations.

```python
class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Discriminator, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, state, action):
        return self.network(torch.cat([state, action], dim=1))

class GAILAgent:
    def __init__(self, state_dim, action_dim, max_action, lr_actor=1e-4, lr_disc=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        # Actor network (policy)
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Discriminator network
        self.discriminator = Discriminator(state_dim, action_dim)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr_disc)

        # Expert demonstrations
        self.expert_data = []
        self.policy_data = []

    def add_expert_demonstration(self, states, actions):
        """Add expert demonstration data"""
        for state, action in zip(states, actions):
            self.expert_data.append((state, action))

    def train_discriminator(self, num_steps=10):
        """Train discriminator to distinguish expert vs. policy data"""
        for _ in range(num_steps):
            if len(self.expert_data) == 0 or len(self.policy_data) == 0:
                continue

            # Sample from expert and policy data
            expert_batch = random.sample(self.expert_data, min(32, len(self.expert_data)))
            policy_batch = random.sample(self.policy_data, min(32, len(self.policy_data)))

            expert_states = torch.FloatTensor([d[0] for d in expert_batch])
            expert_actions = torch.FloatTensor([d[1] for d in expert_batch])

            policy_states = torch.FloatTensor([d[0] for d in policy_batch])
            policy_actions = torch.FloatTensor([d[1] for d in policy_batch])

            # Discriminator loss
            expert_labels = torch.ones(expert_states.size(0), 1)
            policy_labels = torch.zeros(policy_states.size(0), 1)

            expert_predictions = self.discriminator(expert_states, expert_actions)
            policy_predictions = self.discriminator(policy_states, policy_actions)

            expert_loss = nn.BCELoss()(expert_predictions, expert_labels)
            policy_loss = nn.BCELoss()(policy_predictions, policy_labels)

            discriminator_loss = expert_loss + policy_loss

            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

    def get_reward_from_discriminator(self, state, action):
        """Get reward from discriminator (1 - D(s,a))"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = torch.FloatTensor(action).unsqueeze(0)

        with torch.no_grad():
            prob_real = self.discriminator(state_tensor, action_tensor)
            reward = torch.log(prob_real + 1e-8) - torch.log(1 - prob_real + 1e-8)

        return reward.item()
```

## Learning from Human Demonstration

### Learning from Observation (LfO)

Learning from observation allows robots to learn by watching human demonstrations without direct physical interaction.

```python
class LearningFromObservation:
    def __init__(self, visual_encoder_dim=512, action_dim=6):
        # Visual encoder (could be a CNN or Vision Transformer)
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, visual_encoder_dim),  # Adjust based on input size
            nn.ReLU()
        )

        # Imitation network
        self.imitation_network = nn.Sequential(
            nn.Linear(visual_encoder_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

        self.optimizer = optim.Adam(
            list(self.visual_encoder.parameters()) + list(self.imitation_network.parameters()),
            lr=1e-4
        )
        self.criterion = nn.MSELoss()

    def encode_visual_state(self, image):
        """Encode visual input to feature representation"""
        image_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)  # HWC to CHW to BCHW
        features = self.visual_encoder(image_tensor)
        return features

    def predict_action(self, image):
        """Predict action from visual input"""
        features = self.encode_visual_state(image)
        action = self.imitation_network(features)
        return action.detach().numpy()

    def train_from_video(self, video_frames, expert_actions):
        """Train from video demonstration"""
        for frame, expert_action in zip(video_frames, expert_actions):
            # Encode visual state
            features = self.encode_visual_state(frame)

            # Forward pass
            predicted_action = self.imitation_network(features)
            expert_action_tensor = torch.FloatTensor(expert_action).unsqueeze(0)

            # Compute loss
            loss = self.criterion(predicted_action, expert_action_tensor)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
```

### Kinesthetic Teaching

Kinesthetic teaching involves physically guiding the robot through motions to teach new skills.

```python
class KinestheticTeaching:
    def __init__(self, robot_interface, joint_dim):
        self.robot_interface = robot_interface
        self.joint_dim = joint_dim
        self.demonstrations = []
        self.current_demonstration = []

        # Compliance control parameters
        self.compliance_stiffness = 50.0  # N/m
        self.compliance_damping = 10.0   # N*s/m

    def start_demonstration(self):
        """Start recording a new demonstration"""
        self.current_demonstration = []
        print("Starting kinesthetic teaching demonstration...")

        # Switch robot to compliant mode
        self.robot_interface.set_compliance_mode(
            stiffness=self.compliance_stiffness,
            damping=self.compliance_damping
        )

    def record_step(self, joint_positions, joint_velocities, timestamp):
        """Record one step of the demonstration"""
        step_data = {
            'joint_positions': joint_positions.copy(),
            'joint_velocities': joint_velocities.copy(),
            'timestamp': timestamp
        }
        self.current_demonstration.append(step_data)

    def end_demonstration(self):
        """End current demonstration and save it"""
        if self.current_demonstration:
            self.demonstrations.append(self.current_demonstration)
            print(f"Recorded demonstration with {len(self.current_demonstration)} steps")

        # Return robot to normal control mode
        self.robot_interface.set_position_mode()

    def play_demonstration(self, demo_idx=0, speed=1.0):
        """Play back a recorded demonstration"""
        if demo_idx >= len(self.demonstrations):
            print(f"Invalid demonstration index: {demo_idx}")
            return

        demonstration = self.demonstrations[demo_idx]

        for step in demonstration:
            # Set robot to demonstration position
            self.robot_interface.set_joint_positions(step['joint_positions'])

            # Wait for next step based on original timing
            if speed > 0:
                time.sleep(step['dt'] / speed)  # dt would need to be calculated from timestamps

    def generalize_demonstration(self, new_start_pos, new_end_pos):
        """Generalize demonstration to new start/end positions"""
        if not self.demonstrations:
            return []

        # Use the most recent demonstration as base
        base_demo = self.demonstrations[-1]

        # Calculate transformation from original start/end to new start/end
        orig_start = base_demo[0]['joint_positions']
        orig_end = base_demo[-1]['joint_positions']

        # Simple linear scaling (in practice, more sophisticated methods would be used)
        generalized_demo = []
        for step in base_demo:
            # Interpolate between new start and end based on original progression
            orig_progress = np.linalg.norm(step['joint_positions'] - orig_start) / \
                           np.linalg.norm(orig_end - orig_start)

            new_position = (1 - orig_progress) * new_start_pos + orig_progress * new_end_pos
            generalized_demo.append(new_position)

        return generalized_demo
```

## Adaptive Control Systems

### Model Reference Adaptive Control (MRAC)

MRAC adapts controller parameters to match a desired reference model.

```python
class ModelReferenceAdaptiveController:
    def __init__(self, plant_order=2, reference_model_params=None):
        self.plant_order = plant_order

        # Reference model parameters (second-order system)
        if reference_model_params is None:
            self.ref_zeta = 0.7  # Damping ratio
            self.ref_omega_n = 2.0  # Natural frequency
        else:
            self.ref_zeta = reference_model_params['zeta']
            self.ref_omega_n = reference_model_params['omega_n']

        # Adaptive parameters
        self.theta_a = np.zeros(plant_order)  # Adaptive gain for input
        self.theta_b = np.zeros(plant_order)  # Adaptive gain for output

        # Learning rates
        self.gamma_a = 0.1
        self.gamma_b = 0.1

        # State variables
        self.plant_states = np.zeros(plant_order)
        self.ref_states = np.zeros(plant_order)
        self.error_states = np.zeros(plant_order)

        # For parameter estimation
        self.phi = np.zeros(2 * plant_order)  # Regressor vector

    def reference_model(self, r, dt=0.01):
        """Second-order reference model: wn^2 / (s^2 + 2*zeta*wn*s + wn^2)"""
        # State-space representation
        A = np.array([[0, 1],
                      [-self.ref_omega_n**2, -2*self.ref_zeta*self.ref_omega_n]])
        B = np.array([0, self.ref_omega_n**2])

        # Update reference states: dx/dt = Ax + Br
        self.ref_states = self.ref_states + dt * (A @ self.ref_states + B * r)

        return self.ref_states[0]  # Return position

    def control(self, y, r, dt=0.01):
        """Generate control signal using MRAC"""
        # Calculate tracking error
        e = r - y
        self.error_states[0] = e

        # Regressor vector (contains plant states and reference signal)
        self.phi[:self.plant_order] = self.plant_states
        self.phi[self.plant_order:] = [r] + [0]*(self.plant_order-1)

        # Adaptive control law
        u_adaptive = self.theta_a @ self.plant_states + self.theta_b @ np.array([r] + [0]*(self.plant_order-1))

        # Calculate control input
        u = u_adaptive

        # Update plant model states (simplified first-order approximation)
        self.plant_states = self.plant_states + dt * np.array([self.plant_states[1], -4*self.plant_states[0] - 0.5*self.plant_states[1] + u])

        # Parameter adaptation law
        # dθ/dt = -γ * φ * e
        self.theta_a = self.theta_a + self.gamma_a * self.plant_states * e
        self.theta_b = self.theta_b + self.gamma_b * np.array([r] + [0]*(self.plant_order-1)) * e

        return u

    def is_converged(self, tolerance=0.01):
        """Check if adaptation has converged"""
        # Check if tracking error is small
        return abs(self.error_states[0]) < tolerance
```

### Self-Organizing Maps for Skill Learning

Self-Organizing Maps can be used to learn spatial relationships and organize motor skills.

```python
class SelfOrganizingMap:
    def __init__(self, input_dim, map_width, map_height, learning_rate=0.1, neighborhood_radius=2.0):
        self.input_dim = input_dim
        self.map_width = map_width
        self.map_height = map_height
        self.learning_rate = learning_rate
        self.initial_radius = neighborhood_radius
        self.current_radius = neighborhood_radius

        # Initialize weight vectors randomly
        self.weights = np.random.random((map_width, map_height, input_dim))

        # Create coordinate grid
        self.coords = np.zeros((map_width, map_height, 2))
        for i in range(map_width):
            for j in range(map_height):
                self.coords[i, j] = [i, j]

    def find_bmu(self, input_vector):
        """Find Best Matching Unit (BMU)"""
        distances = np.linalg.norm(self.weights - input_vector, axis=2)
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx

    def update_weights(self, input_vector, bmu_idx, iteration, total_iterations):
        """Update weights based on BMU"""
        # Calculate current learning parameters
        current_lr = self.learning_rate * (1 - iteration / total_iterations)
        self.current_radius = self.initial_radius * (1 - iteration / total_iterations)

        # Calculate distance from each unit to BMU
        bmu_coords = self.coords[bmu_idx[0], bmu_idx[1]]
        distances = np.linalg.norm(self.coords - bmu_coords, axis=2)

        # Calculate neighborhood function
        neighborhood = np.exp(-(distances**2) / (2 * self.current_radius**2))

        # Update weights
        for i in range(self.map_width):
            for j in range(self.map_height):
                influence = neighborhood[i, j]
                self.weights[i, j] += influence * current_lr * (input_vector - self.weights[i, j])

    def train_batch(self, input_data, epochs=100):
        """Train SOM on batch of input data"""
        total_iterations = len(input_data) * epochs

        for epoch in range(epochs):
            for iteration, input_vector in enumerate(input_data):
                global_iter = epoch * len(input_data) + iteration
                bmu_idx = self.find_bmu(input_vector)
                self.update_weights(input_vector, bmu_idx, global_iter, total_iterations)

    def get_activation_map(self, input_vector):
        """Get activation strength for each unit"""
        distances = np.linalg.norm(self.weights - input_vector, axis=2)
        # Convert distances to activations (closer = higher activation)
        activations = np.max(distances) - distances
        return activations

class SOMSkillLearner:
    def __init__(self, state_dim, action_dim):
        # SOM for state space organization
        self.state_som = SelfOrganizingMap(state_dim, 10, 10)

        # SOM for action space organization
        self.action_som = SelfOrganizingMap(action_dim, 8, 8)

        # Mapping between state and action SOMs
        self.state_action_mapping = np.zeros((10, 10, 8, 8))

    def learn_from_demonstration(self, states, actions):
        """Learn skill from state-action demonstration"""
        # Train state SOM
        self.state_som.train_batch(states, epochs=50)

        # Train action SOM
        self.action_som.train_batch(actions, epochs=50)

        # Learn mapping between state and action representations
        for state, action in zip(states, actions):
            state_bmu = self.state_som.find_bmu(state)
            action_bmu = self.action_som.find_bmu(action)

            # Strengthen connection between state and action representations
            self.state_action_mapping[state_bmu[0], state_bmu[1], action_bmu[0], action_bmu[1]] += 1

    def generate_action(self, current_state):
        """Generate action for current state using learned mapping"""
        state_bmu = self.state_som.find_bmu(current_state)

        # Get action mapping for this state region
        action_mapping = self.state_action_mapping[state_bmu[0], state_bmu[1]]

        # Find most associated action (could be probabilistic)
        max_idx = np.unravel_index(np.argmax(action_mapping), action_mapping.shape)

        # Return action corresponding to this SOM unit
        return self.action_som.weights[max_idx[0], max_idx[1]]
```

## Transfer Learning and Domain Adaptation

### Sim-to-Real Transfer

```python
class DomainRandomization:
    def __init__(self, env):
        self.env = env
        self.param_ranges = {
            'mass': (0.8, 1.2),  # 80% to 120% of nominal
            'friction': (0.5, 1.5),
            'gravity': (9.0, 10.0),
            'motor_torque': (0.8, 1.2)
        }

    def randomize_environment(self):
        """Randomize environment parameters"""
        for param, (min_val, max_val) in self.param_ranges.items():
            if hasattr(self.env, param):
                random_val = np.random.uniform(min_val, max_val)
                setattr(self.env, param, random_val)

    def train_with_randomization(self, agent, episodes=1000):
        """Train agent with domain randomization"""
        for episode in range(episodes):
            # Randomize environment at start of episode
            self.randomize_environment()

            # Train agent in randomized environment
            state = self.env.reset()
            done = False

            while not done:
                action = agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)

                # Store experience and train
                agent.remember(state, action, reward, next_state, done)
                agent.train()

                state = next_state

class DomainAdaptationNetwork(nn.Module):
    def __init__(self, input_dim, feature_dim=256):
        super(DomainAdaptationNetwork, self).__init__()

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        # Label predictor (task-specific)
        self.label_predictor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Example: single output
        )

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 2 domains: sim and real
        )

    def forward(self, x, alpha=0.0):
        features = self.feature_extractor(x)

        # Reverse gradient for domain adaptation
        reversed_features = self.gradient_reverse_layer(features, alpha)

        labels = self.label_predictor(features)
        domains = self.domain_classifier(reversed_features)

        return labels, domains

    def gradient_reverse_layer(self, x, alpha):
        """Gradient reversal layer for domain adaptation"""
        return GradientReverseFunction.apply(x, alpha)

class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.alpha * grad_output.neg(), None
```

## Online Learning and Adaptation

### Online Parameter Estimation

```python
class RecursiveLeastSquares:
    def __init__(self, n_params, forgetting_factor=0.98):
        self.n_params = n_params
        self.lam = forgetting_factor  # Forgetting factor (0 < λ ≤ 1)

        # Initialize parameter estimate
        self.theta = np.zeros(n_params)

        # Initialize covariance matrix (large initial values = uninformative prior)
        self.P = np.eye(n_params) / 0.001

    def update(self, phi, y):
        """Update parameter estimate with new measurement"""
        # phi: regressor vector (input features)
        # y: measurement (output)

        # Calculate gain
        denom = self.lam + phi.T @ self.P @ phi
        K = self.P @ phi / denom

        # Update parameter estimate
        self.theta = self.theta + K * (y - phi.T @ self.theta)

        # Update covariance
        self.P = (self.P - np.outer(K, phi) @ self.P) / self.lam

        return self.theta.copy()

    def predict(self, phi):
        """Predict output for given regressor"""
        return phi.T @ self.theta

class OnlineLearningController:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Online parameter estimation for robot dynamics
        self.dynamics_estimator = RecursiveLeastSquares(state_dim + action_dim)

        # Adaptive controller parameters
        self.kp = np.ones(action_dim) * 10.0  # Proportional gains
        self.ki = np.ones(action_dim) * 1.0   # Integral gains
        self.kd = np.ones(action_dim) * 1.0   # Derivative gains

        # Integral and derivative terms
        self.integral_error = np.zeros(action_dim)
        self.previous_error = np.zeros(action_dim)

    def update_dynamics_model(self, state, action, next_state):
        """Update dynamics model using RLS"""
        # Create regressor vector: [state, action]
        phi = np.concatenate([state, action])

        # Update for each state dimension
        for i in range(self.state_dim):
            self.dynamics_estimator.update(phi, next_state[i])

    def adaptive_control(self, current_state, desired_state, dt=0.01):
        """Generate adaptive control with online learning"""
        # Calculate error
        error = desired_state - current_state

        # Update integral and derivative terms
        self.integral_error += error * dt
        derivative_error = (error - self.previous_error) / dt
        self.previous_error = error.copy()

        # PID control with adaptive gains
        proportional = self.kp * error
        integral = self.ki * self.integral_error
        derivative = self.kd * derivative_error

        # Base control signal
        u_base = proportional + integral + derivative

        # Adapt gains based on performance
        self.adapt_gains(error, dt)

        return u_base

    def adapt_gains(self, error, dt):
        """Adapt control gains based on tracking performance"""
        error_magnitude = np.linalg.norm(error)

        # Increase gains if error is large
        if error_magnitude > 0.5:
            self.kp *= 1.01
            self.ki *= 1.01
        # Decrease gains if error is small but oscillating
        elif error_magnitude < 0.05 and np.linalg.norm(self.previous_error - error) > 0.1:
            self.kp *= 0.99
            self.ki *= 0.99

        # Keep gains within reasonable bounds
        self.kp = np.clip(self.kp, 1.0, 100.0)
        self.ki = np.clip(self.ki, 0.1, 20.0)
        self.kd = np.clip(self.kd, 0.1, 20.0)
```

## Learning System Integration

### Multi-Task Learning Framework

```python
class MultiTaskLearningFramework:
    def __init__(self, tasks, shared_layers=2):
        self.tasks = tasks  # List of task names
        self.task_networks = {}
        self.shared_encoder = None

        # Create shared feature extractor
        self.shared_encoder = nn.Sequential(
            nn.Linear(128, 256),  # Adjust input size as needed
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # Create task-specific heads
        for task in tasks:
            self.task_networks[task] = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.get_task_output_size(task))
            )

        # Optimizers
        self.shared_optimizer = optim.Adam(self.shared_encoder.parameters(), lr=1e-4)
        self.task_optimizers = {
            task: optim.Adam(list(self.task_networks[task].parameters()), lr=1e-4)
            for task in tasks
        }

        # Task weights for balanced learning
        self.task_weights = {task: 1.0 for task in tasks}

    def get_task_output_size(self, task):
        """Get output size for specific task"""
        task_sizes = {
            'walking': 12,  # Joint commands for walking
            'balancing': 12,  # Joint commands for balance
            'manipulation': 6,  # End-effector commands
            'perception': 1000  # Detection outputs
        }
        return task_sizes.get(task, 64)

    def forward(self, x, task):
        """Forward pass for specific task"""
        shared_features = self.shared_encoder(x)
        task_output = self.task_networks[task](shared_features)
        return task_output

    def train_step(self, batch_data, task):
        """Single training step for specific task"""
        inputs, targets = batch_data

        # Forward pass
        shared_features = self.shared_encoder(inputs)
        outputs = self.task_networks[task](shared_features)

        # Compute loss
        criterion = nn.MSELoss()
        loss = criterion(outputs, targets) * self.task_weights[task]

        # Backward pass
        self.shared_optimizer.zero_grad()
        self.task_optimizers[task].zero_grad()

        loss.backward()

        self.shared_optimizer.step()
        self.task_optimizers[task].step()

        return loss.item()

    def train_multitask(self, task_data, epochs=100):
        """Train on multiple tasks simultaneously"""
        for epoch in range(epochs):
            epoch_losses = {}

            for task in self.tasks:
                if task in task_data and len(task_data[task]) > 0:
                    # Sample batch for this task
                    batch = random.choice(task_data[task])
                    loss = self.train_step(batch, task)
                    epoch_losses[task] = loss

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Losses: {epoch_losses}")
```

### Lifelong Learning System

```python
class LifelongLearningSystem:
    def __init__(self, initial_tasks):
        self.tasks = initial_tasks
        self.task_models = {}
        self.task_buffers = {}  # Experience replay for each task
        self.shared_representation = None

        # Initialize for each task
        for task in self.tasks:
            self.task_models[task] = self.create_task_model(task)
            self.task_buffers[task] = deque(maxlen=10000)

        # Shared representation network
        self.shared_representation = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # Catastrophic forgetting prevention
        self.ewc_lambda = 1.0  # Elastic Weight Consolidation strength

    def create_task_model(self, task):
        """Create model for specific task"""
        if task == 'walking':
            return nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 12)  # 12 joint commands
            )
        elif task == 'object_manipulation':
            return nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 6)  # 6 DOF end-effector
            )
        else:
            return nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 8)  # Default output
            )

    def add_task(self, new_task):
        """Add new task to the system"""
        if new_task not in self.tasks:
            self.tasks.append(new_task)
            self.task_models[new_task] = self.create_task_model(new_task)
            self.task_buffers[new_task] = deque(maxlen=10000)

    def consolidate_knowledge(self, old_task, new_task):
        """Prevent catastrophic forgetting when learning new tasks"""
        # Store important weights from old task
        old_params = {name: param.clone() for name, param in self.task_models[old_task].named_parameters()}

        # Compute importance weights using Fisher Information Matrix approximation
        # (Simplified version - in practice, this would be more complex)
        old_importance = {name: torch.ones_like(param) for name, param in old_params.items()}

        # When training on new task, add regularization to preserve old knowledge
        def ewc_loss(new_params, old_params, importance):
            loss = 0
            for name, param in new_params.items():
                if name in old_params:
                    loss += (importance[name] * (param - old_params[name]) ** 2).sum()
            return loss

        return old_params, old_importance, ewc_loss

    def learn_from_experience(self, experiences, task):
        """Learn from experience replay"""
        if task not in self.task_models:
            print(f"Task {task} not recognized")
            return

        model = self.task_models[task]
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # Process experiences
        for exp in experiences:
            state, action, reward, next_state, done = exp

            # Encode state through shared representation
            encoded_state = self.shared_representation(torch.FloatTensor(state))

            # Get model prediction
            prediction = model(encoded_state)

            # Compute loss and update
            target = torch.FloatTensor(action)  # or computed target
            loss = nn.MSELoss()(prediction, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def select_task(self, current_state):
        """Select appropriate task based on current context"""
        # This would use context recognition, state evaluation, etc.
        # For simplicity, return a default task
        return 'walking'  # This would be more sophisticated in practice

    def execute_task(self, state, task):
        """Execute specific task"""
        if task in self.task_models:
            encoded_state = self.shared_representation(torch.FloatTensor(state))
            action = self.task_models[task](encoded_state)
            return action.detach().numpy()
        else:
            return np.zeros(6)  # Default action
```

## Weekly Breakdown for Chapter 12
- **Week 12.1**: Reinforcement learning fundamentals and algorithms
- **Week 12.2**: Imitation learning and human demonstration
- **Week 12.3**: Adaptive control and online learning
- **Week 12.4**: Multi-task learning and lifelong learning

## Assessment
- **Quiz 12.1**: Learning paradigms and algorithms (Multiple choice and short answer)
- **Assignment 12.2**: Implement a reinforcement learning controller for robot balance
- **Lab Exercise 12.1**: Train a robot to learn from human demonstration

## Diagram Placeholders
- ![Reinforcement Learning Architecture](./images/rl_architecture.png)
- ![Imitation Learning Framework](./images/imitation_learning_framework.png)
- ![Adaptive Control System](./images/adaptive_control_system.png)

## Code Snippet: Complete Learning System
```python
#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time

class CompleteLearningSystem:
    """
    Complete learning system for humanoid robot with multiple learning modalities
    """
    def __init__(self, state_dim=24, action_dim=12, device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Initialize all learning components
        self.rl_agent = DDPGAgent(state_dim, action_dim, max_action=1.0)
        self.imitation_learner = ImitationLearner(state_dim, action_dim)
        self.adaptive_controller = OnlineLearningController(state_dim, action_dim)
        self.lifelong_learner = LifelongLearningSystem(['balance', 'walk', 'manipulate'])

        # Experience replay for RL
        self.replay_buffer = deque(maxlen=100000)

        # Performance tracking
        self.episode_rewards = []
        self.success_rates = []

        # Learning flags
        self.learning_modes = {
            'reinforcement': True,
            'imitation': True,
            'adaptive': True,
            'lifelong': True
        }

        print("Complete Learning System initialized")

    def collect_experience(self, state, action, reward, next_state, done):
        """Collect experience for learning"""
        experience = (state, action, reward, next_state, done)
        self.replay_buffer.append(experience)

        # Add to lifelong learning system
        if self.learning_modes['lifelong']:
            task = self.select_current_task(state)
            self.lifelong_learner.task_buffers[task].append(experience)

    def select_current_task(self, state):
        """Select current task based on state"""
        # Simple heuristic - in practice, this would be more sophisticated
        com_height = state[2]  # Assuming CoM height is at index 2
        if com_height < 0.5:
            return 'balance'
        elif abs(state[0]) > 0.1:  # Forward velocity
            return 'walk'
        else:
            return 'balance'

    def learn(self):
        """Execute learning from collected experiences"""
        if len(self.replay_buffer) < 32:
            return  # Need minimum experiences to start learning

        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, 32)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.FloatTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        # Update RL agent if enabled
        if self.learning_modes['reinforcement']:
            # Add experiences to RL agent's memory
            for exp in batch:
                self.rl_agent.memory.append(exp)
            self.rl_agent.train()

        # Update imitation learning if we have demonstrations
        if (self.learning_modes['imitation'] and
            len(self.imitation_learner.demonstrations) > 100):
            # Train imitation learner on recent demonstrations
            self.imitation_learner.train(epochs=1, batch_size=16)

        # Update adaptive controller
        if self.learning_modes['adaptive']:
            for state, action, reward, next_state, done in batch:
                self.adaptive_controller.update_dynamics_model(state, action, next_state)

    def select_action(self, state, exploration=True):
        """Select action using ensemble of learning methods"""
        actions = {}

        # Get action from RL agent
        if self.learning_modes['reinforcement']:
            rl_action = self.rl_agent.select_action(state, add_noise=exploration)
            actions['rl'] = rl_action

        # Get action from imitation learning (if trained)
        if (self.learning_modes['imitation'] and
            len(self.imitation_learner.demonstrations) > 0):
            try:
                il_action = self.imitation_learner.predict(state)
                actions['imitation'] = il_action
            except:
                pass  # Skip if not trained yet

        # Get action from adaptive controller
        if self.learning_modes['adaptive']:
            # For adaptive controller, we need a reference trajectory
            # Using a simple reference for demonstration
            reference_state = np.zeros_like(state)
            adaptive_action = self.adaptive_controller.adaptive_control(
                state, reference_state
            )
            actions['adaptive'] = adaptive_action

        # Get action from lifelong learning
        if self.learning_modes['lifelong']:
            current_task = self.select_current_task(state)
            lifelong_action = self.lifelong_learner.execute_task(state, current_task)
            actions['lifelong'] = lifelong_action

        # Combine actions using weighted average or voting
        if len(actions) > 0:
            # Simple averaging of available actions
            combined_action = np.zeros(self.action_dim)
            weight = 1.0 / len(actions)

            for action in actions.values():
                combined_action += weight * np.clip(action, -1, 1)

            return combined_action
        else:
            # Return random action if no learning is active
            return np.random.uniform(-0.1, 0.1, self.action_dim)

    def add_demonstration(self, states, actions):
        """Add demonstration for imitation learning"""
        self.imitation_learner.add_demonstration(states, actions)

        # Also add to lifelong learning system
        task = 'demonstration_task'  # Could be more specific
        for s, a in zip(states, actions):
            self.lifelong_learner.task_buffers[task].append((s, a, 0, s, False))

    def evaluate_performance(self, num_episodes=10):
        """Evaluate learning system performance"""
        total_reward = 0
        success_count = 0

        for episode in range(num_episodes):
            episode_reward = 0
            state = self.reset_environment()
            done = False
            steps = 0

            while not done and steps < 1000:  # Max 1000 steps per episode
                action = self.select_action(state, exploration=False)
                next_state, reward, done, info = self.step_environment(action)

                # Collect experience
                self.collect_experience(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward
                steps += 1

            total_reward += episode_reward

            # Count as success if robot maintained balance
            if steps >= 1000:  # Survived full episode
                success_count += 1

        avg_reward = total_reward / num_episodes
        success_rate = success_count / num_episodes

        self.episode_rewards.append(avg_reward)
        self.success_rates.append(success_rate)

        return avg_reward, success_rate

    def reset_environment(self):
        """Reset environment for evaluation (placeholder)"""
        # In practice, this would interface with simulation or real robot
        return np.random.randn(self.state_dim) * 0.1

    def step_environment(self, action):
        """Step environment with action (placeholder)"""
        # In practice, this would interface with simulation or real robot
        next_state = np.random.randn(self.state_dim) * 0.1
        reward = np.random.randn()  # Random reward for demonstration
        done = random.random() < 0.01  # 1% chance to end
        info = {}
        return next_state, reward, done, info

    def save_model(self, filepath):
        """Save learning system model"""
        model_data = {
            'rl_actor_state': self.rl_agent.actor.state_dict(),
            'rl_critic_state': self.rl_agent.critic.state_dict(),
            'imitation_model_state': self.imitation_learner.model.state_dict(),
            'episode_rewards': self.episode_rewards,
            'success_rates': self.success_rates
        }
        torch.save(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load learning system model"""
        model_data = torch.load(filepath, map_location=self.device)

        self.rl_agent.actor.load_state_dict(model_data['rl_actor_state'])
        self.rl_agent.critic.load_state_dict(model_data['rl_critic_state'])
        self.imitation_learner.model.load_state_dict(model_data['imitation_model_state'])

        self.episode_rewards = model_data.get('episode_rewards', [])
        self.success_rates = model_data.get('success_rates', [])

        print(f"Model loaded from {filepath}")

    def train(self, num_episodes=1000, eval_frequency=100):
        """Main training loop"""
        print("Starting training...")

        for episode in range(num_episodes):
            episode_reward = 0
            state = self.reset_environment()
            done = False
            steps = 0

            while not done and steps < 1000:
                # Select action with exploration
                action = self.select_action(state, exploration=True)

                # Execute action
                next_state, reward, done, info = self.step_environment(action)

                # Store experience
                self.collect_experience(state, action, reward, next_state, done)

                # Learn from experience
                if steps % 10 == 0:  # Learn every 10 steps
                    self.learn()

                state = next_state
                episode_reward += reward
                steps += 1

            # Print progress
            if episode % 50 == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}, Steps: {steps}")

            # Evaluate periodically
            if episode > 0 and episode % eval_frequency == 0:
                avg_reward, success_rate = self.evaluate_performance(num_episodes=5)
                print(f"Evaluation - Episode {episode}: Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.2f}")

        print("Training completed!")

def main():
    # Initialize learning system
    learning_system = CompleteLearningSystem(state_dim=24, action_dim=12)

    # Example: Add a demonstration
    demo_states = [np.random.randn(24) for _ in range(50)]
    demo_actions = [np.random.randn(12) for _ in range(50)]
    learning_system.add_demonstration(demo_states, demo_actions)

    # Train the system
    try:
        learning_system.train(num_episodes=1000)
    except KeyboardInterrupt:
        print("Training interrupted by user")

    # Save the trained model
    learning_system.save_model("trained_learning_system.pth")

if __name__ == "__main__":
    main()
```

## Additional Resources
- Reinforcement Learning: An Introduction by Sutton and Barto
- Deep Learning by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- Robot Learning Literature and Conferences (CoRL, RSS, ICRA)
- OpenAI Spinning Up for RL Education
- PyTorch and TensorFlow for Deep Learning