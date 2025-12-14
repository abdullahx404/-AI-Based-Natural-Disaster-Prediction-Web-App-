"""
Reinforcement Learning Module - Week 12 Requirement
Implements Q-Learning for flood evacuation decisions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class FloodEvacuationEnvironment:
    """
    Environment for flood evacuation simulation.
    
    States: (flood_level, population_at_risk, resources_available, time_remaining)
    Actions: 
        0 - Wait and monitor
        1 - Issue warning
        2 - Begin voluntary evacuation
        3 - Begin mandatory evacuation
        4 - Deploy emergency resources
    
    Rewards:
        - Saving lives: +100 per person evacuated safely
        - Resource costs: -10 per resource deployed
        - False alarm penalty: -50 if evacuation called but no flood
        - Delay penalty: -20 per time step in dangerous conditions
        - Casualty: -500 per person harmed
    """
    
    # Flood levels
    FLOOD_NONE = 0
    FLOOD_LOW = 1
    FLOOD_MODERATE = 2
    FLOOD_HIGH = 3
    FLOOD_SEVERE = 4
    
    # Actions
    ACTION_WAIT = 0
    ACTION_WARN = 1
    ACTION_VOLUNTARY_EVAC = 2
    ACTION_MANDATORY_EVAC = 3
    ACTION_DEPLOY_RESOURCES = 4
    
    def __init__(self, max_time: int = 24, max_population: int = 1000):
        """
        Initialize environment.
        
        Args:
            max_time: Maximum time steps (hours)
            max_population: Maximum population at risk
        """
        self.max_time = max_time
        self.max_population = max_population
        
        self.action_names = [
            "Wait & Monitor",
            "Issue Warning", 
            "Voluntary Evacuation",
            "Mandatory Evacuation",
            "Deploy Resources"
        ]
        
        self.reset()
    
    def reset(self, seed: int = None) -> Tuple:
        """
        Reset environment to initial state.
        
        Returns:
            state: Initial state tuple
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Initial conditions
        self.flood_level = np.random.choice([0, 1], p=[0.7, 0.3])
        self.population_at_risk = np.random.randint(100, self.max_population)
        self.evacuated = 0
        self.resources_deployed = 0
        self.time_remaining = self.max_time
        self.warning_issued = False
        self.evacuation_started = False
        self.done = False
        
        # Hidden: True flood trajectory (agent doesn't see this)
        self.will_flood = np.random.random() < 0.4  # 40% chance of major flood
        self.flood_peak_time = np.random.randint(8, 20) if self.will_flood else -1
        
        return self._get_state()
    
    def _get_state(self) -> Tuple:
        """Get current state tuple"""
        return (
            self.flood_level,
            min(self.population_at_risk // 100, 10),  # Discretize to 0-10
            min(self.resources_deployed // 2, 5),     # Discretize to 0-5
            min(self.time_remaining // 4, 6)          # Discretize to 0-6
        )
    
    def _update_flood_level(self):
        """Simulate flood progression"""
        time_elapsed = self.max_time - self.time_remaining
        
        if self.will_flood:
            # Flood builds up before peak
            if time_elapsed < self.flood_peak_time:
                progress = time_elapsed / self.flood_peak_time
                target_level = int(progress * 4)  # Gradually increase to SEVERE
                if np.random.random() < 0.3:  # Some randomness
                    self.flood_level = min(self.flood_level + 1, target_level)
            else:
                # At or past peak
                self.flood_level = min(self.flood_level + 1, self.FLOOD_SEVERE)
        else:
            # No major flood, might have minor fluctuations
            if np.random.random() < 0.1:
                self.flood_level = min(self.flood_level + 1, self.FLOOD_LOW)
            elif np.random.random() < 0.2 and self.flood_level > 0:
                self.flood_level -= 1
    
    def step(self, action: int) -> Tuple[Tuple, float, bool, Dict]:
        """
        Take action in environment.
        
        Args:
            action: Action to take (0-4)
            
        Returns:
            next_state: New state
            reward: Reward received
            done: Whether episode is finished
            info: Additional information
        """
        if self.done:
            return self._get_state(), 0, True, {}
        
        reward = 0
        info = {'action': self.action_names[action]}
        
        # Process action
        if action == self.ACTION_WAIT:
            # Waiting in dangerous conditions is risky
            if self.flood_level >= self.FLOOD_HIGH:
                reward -= 20  # Delay penalty
                
        elif action == self.ACTION_WARN:
            if not self.warning_issued:
                self.warning_issued = True
                reward += 5  # Small bonus for warning
                
        elif action == self.ACTION_VOLUNTARY_EVAC:
            if not self.evacuation_started:
                self.evacuation_started = True
                # Some people evacuate voluntarily
                evac_rate = 0.2 + 0.1 * self.flood_level  # Higher flood = more compliance
                evacuated_now = int(self.population_at_risk * evac_rate)
                self.evacuated += evacuated_now
                self.population_at_risk -= evacuated_now
                reward += evacuated_now * 2  # Reward for each person evacuated
                
        elif action == self.ACTION_MANDATORY_EVAC:
            if not self.evacuation_started:
                self.evacuation_started = True
            # Mandatory evacuation is more effective but costly
            evac_rate = 0.5 + 0.1 * self.flood_level
            evacuated_now = int(self.population_at_risk * evac_rate)
            self.evacuated += evacuated_now
            self.population_at_risk -= evacuated_now
            reward += evacuated_now * 3  # Higher reward for mandatory evac
            reward -= 30  # Cost of mandatory evacuation
            
        elif action == self.ACTION_DEPLOY_RESOURCES:
            self.resources_deployed += 2
            reward -= 10  # Resource cost
            # Resources help save lives if flood is active
            if self.flood_level >= self.FLOOD_MODERATE:
                saved = min(self.population_at_risk, 50)
                self.evacuated += saved
                self.population_at_risk -= saved
                reward += saved * 5
        
        # Update flood level
        self._update_flood_level()
        self.time_remaining -= 1
        
        # Calculate casualties if flood is severe and people remain
        if self.flood_level >= self.FLOOD_HIGH and self.population_at_risk > 0:
            # Casualties depend on flood level and resources
            base_casualty_rate = 0.02 * (self.flood_level - 2)
            protection = min(0.5, self.resources_deployed * 0.05)
            casualty_rate = max(0, base_casualty_rate - protection)
            
            casualties = int(self.population_at_risk * casualty_rate)
            self.population_at_risk -= casualties
            reward -= casualties * 500  # Severe penalty for casualties
            info['casualties'] = casualties
        
        # Check terminal conditions
        if self.time_remaining <= 0 or self.population_at_risk <= 0:
            self.done = True
            # Final bonus for people saved
            reward += self.evacuated * 10
            
            # Penalty for false alarm if no flood and evacuation was called
            if not self.will_flood and self.evacuation_started:
                reward -= 50
        
        info['flood_level'] = self.flood_level
        info['evacuated'] = self.evacuated
        info['remaining'] = self.population_at_risk
        
        return self._get_state(), reward, self.done, info


class QLearningAgent:
    """
    Q-Learning agent for flood evacuation decisions.
    
    Learns optimal policy through trial and error:
    - Explores environment to discover state-action-reward patterns
    - Updates Q-values using Bellman equation
    - Balances exploration vs exploitation with epsilon-greedy
    """
    
    def __init__(self, n_actions: int = 5, learning_rate: float = 0.1,
                 discount_factor: float = 0.95, epsilon: float = 1.0,
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01):
        """
        Initialize Q-Learning agent.
        
        Args:
            n_actions: Number of possible actions
            learning_rate: Alpha - how much to update Q-values
            discount_factor: Gamma - importance of future rewards
            epsilon: Initial exploration rate
            epsilon_decay: How fast to reduce exploration
            epsilon_min: Minimum exploration rate
        """
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: state -> action values
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        
        # Training history
        self.episode_rewards = []
        self.episode_lengths = []
        
    def choose_action(self, state: Tuple, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (enables exploration)
            
        Returns:
            action: Chosen action
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: best known action
            return np.argmax(self.q_table[state])
    
    def update(self, state: Tuple, action: int, reward: float, 
               next_state: Tuple, done: bool):
        """
        Update Q-value using Bellman equation.
        
        Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode ended
        """
        current_q = self.q_table[state][action]
        
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        # Q-learning update
        self.q_table[state][action] = current_q + self.learning_rate * (target - current_q)
    
    def decay_epsilon(self):
        """Reduce exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self, env: FloodEvacuationEnvironment, episodes: int = 1000,
              verbose: bool = True) -> Dict:
        """
        Train agent on environment.
        
        Args:
            env: Environment to train on
            episodes: Number of training episodes
            verbose: Print progress
            
        Returns:
            history: Training history
        """
        print("\n" + "=" * 60)
        print("Q-LEARNING TRAINING")
        print("=" * 60)
        
        for episode in range(episodes):
            state = env.reset(seed=episode)
            total_reward = 0
            steps = 0
            
            while not env.done:
                # Choose action
                action = self.choose_action(state, training=True)
                
                # Take action
                next_state, reward, done, info = env.step(action)
                
                # Update Q-values
                self.update(state, action, reward, next_state, done)
                
                total_reward += reward
                state = next_state
                steps += 1
            
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            self.decay_epsilon()
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode + 1}/{episodes} - Avg Reward: {avg_reward:.2f} - Epsilon: {self.epsilon:.3f}")
        
        return {
            'rewards': self.episode_rewards,
            'lengths': self.episode_lengths,
            'final_epsilon': self.epsilon
        }
    
    def get_policy(self) -> Dict[Tuple, int]:
        """
        Get learned policy (best action for each state).
        
        Returns:
            policy: Mapping from state to best action
        """
        policy = {}
        for state, q_values in self.q_table.items():
            policy[state] = int(np.argmax(q_values))
        return policy
    
    def get_q_values(self, state: Tuple) -> np.ndarray:
        """Get Q-values for a state"""
        return self.q_table[state].copy()


class FloodEvacuationRL:
    """
    High-level interface for Flood Evacuation RL system.
    Combines environment and agent with easy-to-use methods.
    """
    
    def __init__(self):
        self.env = FloodEvacuationEnvironment()
        self.agent = QLearningAgent(n_actions=5)
        self.trained = False
        
    def train(self, episodes: int = 1000) -> Dict:
        """Train the RL agent"""
        history = self.agent.train(self.env, episodes)
        self.trained = True
        return history
    
    def evaluate(self, n_episodes: int = 100) -> Dict:
        """
        Evaluate trained policy.
        
        Args:
            n_episodes: Number of evaluation episodes
            
        Returns:
            results: Evaluation metrics
        """
        if not self.trained:
            raise ValueError("Agent not trained. Call train() first.")
        
        total_rewards = []
        total_evacuated = []
        total_casualties = []
        
        for _ in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_casualties = 0
            
            while not self.env.done:
                action = self.agent.choose_action(state, training=False)
                state, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_casualties += info.get('casualties', 0)
            
            total_rewards.append(episode_reward)
            total_evacuated.append(self.env.evacuated)
            total_casualties.append(episode_casualties)
        
        return {
            'avg_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'avg_evacuated': np.mean(total_evacuated),
            'avg_casualties': np.mean(total_casualties),
            'success_rate': np.mean([1 if c == 0 else 0 for c in total_casualties])
        }
    
    def demonstrate_policy(self, seed: int = 42) -> List[Dict]:
        """
        Demonstrate learned policy on single episode.
        
        Args:
            seed: Random seed for reproducible demo
            
        Returns:
            trajectory: List of step information
        """
        if not self.trained:
            raise ValueError("Agent not trained. Call train() first.")
        
        trajectory = []
        state = self.env.reset(seed=seed)
        
        print("\n" + "=" * 60)
        print("POLICY DEMONSTRATION")
        print("=" * 60)
        print(f"Initial state: Flood={state[0]}, Pop risk={state[1]*100}, Resources={state[2]*2}, Time={state[3]*4}h")
        
        step = 0
        while not self.env.done:
            q_values = self.agent.get_q_values(state)
            action = self.agent.choose_action(state, training=False)
            
            next_state, reward, done, info = self.env.step(action)
            
            step_info = {
                'step': step,
                'state': state,
                'action': action,
                'action_name': self.env.action_names[action],
                'reward': reward,
                'q_values': q_values.tolist(),
                'info': info
            }
            trajectory.append(step_info)
            
            print(f"\nStep {step}: {info['action']}")
            print(f"  Flood Level: {info['flood_level']}, Evacuated: {info['evacuated']}, Remaining: {info['remaining']}")
            print(f"  Reward: {reward:.1f}")
            
            state = next_state
            step += 1
        
        print(f"\n--- EPISODE COMPLETE ---")
        print(f"Total evacuated: {self.env.evacuated}")
        print(f"Final population at risk: {self.env.population_at_risk}")
        
        return trajectory
    
    def get_recommendation(self, flood_level: int, population: int, 
                          resources: int, time_remaining: int) -> Dict:
        """
        Get action recommendation for current situation.
        
        Args:
            flood_level: Current flood level (0-4)
            population: Population at risk
            resources: Resources already deployed
            time_remaining: Time remaining (hours)
            
        Returns:
            recommendation: Action recommendation with explanation
        """
        if not self.trained:
            raise ValueError("Agent not trained. Call train() first.")
        
        # Discretize to match state space
        state = (
            min(flood_level, 4),
            min(population // 100, 10),
            min(resources // 2, 5),
            min(time_remaining // 4, 6)
        )
        
        q_values = self.agent.get_q_values(state)
        best_action = int(np.argmax(q_values))
        
        # Generate explanation
        explanations = {
            0: "Conditions are being monitored. No immediate action required.",
            1: "Issue a flood warning to alert the population.",
            2: "Begin voluntary evacuation for at-risk areas.",
            3: "Initiate mandatory evacuation immediately.",
            4: "Deploy emergency rescue and relief resources."
        }
        
        return {
            'recommended_action': best_action,
            'action_name': self.env.action_names[best_action],
            'explanation': explanations[best_action],
            'confidence': float(np.max(q_values) - np.mean(q_values)),
            'all_q_values': dict(zip(self.env.action_names, q_values.tolist()))
        }


def demo_reinforcement_learning():
    """Demonstrate Q-Learning for flood evacuation"""
    print("=" * 60)
    print("Q-LEARNING FOR FLOOD EVACUATION DECISIONS - Demo")
    print("=" * 60)
    
    # Create RL system
    rl_system = FloodEvacuationRL()
    
    # Train
    history = rl_system.train(episodes=500)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    results = rl_system.evaluate(n_episodes=100)
    print(f"\nAverage Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Average Evacuated: {results['avg_evacuated']:.1f}")
    print(f"Average Casualties: {results['avg_casualties']:.2f}")
    print(f"Success Rate (0 casualties): {results['success_rate']*100:.1f}%")
    
    # Demonstrate policy
    trajectory = rl_system.demonstrate_policy(seed=123)
    
    # Get a recommendation
    print("\n" + "=" * 60)
    print("SAMPLE RECOMMENDATION")
    print("=" * 60)
    
    recommendation = rl_system.get_recommendation(
        flood_level=3,  # HIGH
        population=500,
        resources=2,
        time_remaining=12
    )
    
    print(f"\nSituation: Flood Level HIGH, 500 at risk, 2 resources, 12h remaining")
    print(f"Recommended Action: {recommendation['action_name']}")
    print(f"Explanation: {recommendation['explanation']}")
    print(f"Confidence: {recommendation['confidence']:.2f}")
    
    return rl_system, history, results


if __name__ == "__main__":
    demo_reinforcement_learning()
