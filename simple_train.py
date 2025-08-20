#!/usr/bin/env python3
"""
Super Simple RL Training with Real-time Reward Display
Shows reward at every epoch/episode
"""

import numpy as np
import matplotlib.pyplot as plt
import random

print("🚀 Starting Simple RL Training!")
print("📊 Showing reward vs epoch in real-time")
print("=" * 50)

class SimpleAgent:
    """Very simple RL agent"""
    def __init__(self):
        self.q_table = {}  # Simple Q-table
        self.learning_rate = 0.1
        self.epsilon = 0.8
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.1
    
    def get_action(self, state):
        """Choose action based on epsilon-greedy"""
        if random.random() < self.epsilon:
            return random.randint(0, 2)  # Random action (0=local, 1=fog1, 2=fog2)
        
        # Get Q-values for this state
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]
        
        return np.argmax(self.q_table[state])
    
    def update_q(self, state, action, reward, next_state):
        """Update Q-table"""
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0, 0.0, 0.0]
        
        # Q-learning update
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + 0.9 * max_next_q - current_q)
        self.q_table[state][action] = new_q
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def simulate_environment():
    """Simulate IoT-Fog environment"""
    # Generate random task
    task_size = random.randint(1, 5)
    fog_load = random.randint(0, 3)
    
    state = (task_size, fog_load)
    return state

def calculate_reward(state, action):
    """Calculate reward based on state and action"""
    task_size, fog_load = state
    
    if action == 0:  # Local processing
        delay = task_size * 1.5  # Local is slower
        reward = -delay
    elif action == 1:  # Fog node 1
        delay = task_size * 0.8 + fog_load * 0.5
        reward = -delay
    else:  # Fog node 2
        delay = task_size * 0.7 + fog_load * 0.3
        reward = -delay
    
    # Add some randomness
    reward += random.uniform(-0.5, 0.5)
    
    return reward

def train_agent(n_episodes=100):
    """Train the agent and show progress"""
    agent = SimpleAgent()
    episode_rewards = []
    episode_actions = []
    
    print("Epoch | Reward | Action | Epsilon | Progress")
    print("-" * 45)
    
    for episode in range(n_episodes):
        # Simulate episode
        total_reward = 0
        actions_taken = []
        
        # Run multiple steps per episode
        for step in range(10):
            # Get current state
            state = simulate_environment()
            
            # Choose action
            action = agent.get_action(state)
            actions_taken.append(action)
            
            # Get reward
            reward = calculate_reward(state, action)
            total_reward += reward
            
            # Get next state and update Q-table
            next_state = simulate_environment()
            agent.update_q(state, action, reward, next_state)
        
        # Record episode data
        episode_rewards.append(total_reward)
        episode_actions.append(actions_taken)
        
        # Show progress every episode
        avg_action = np.mean(actions_taken)
        progress = "█" * (episode // 5) + "░" * ((n_episodes - episode) // 5)
        
        print(f" {episode+1:3d}  | {total_reward:6.1f} | {avg_action:6.2f} | {agent.epsilon:7.3f} | {progress[:20]}")
        
        # Show running average every 10 episodes
        if (episode + 1) % 10 == 0:
            recent_avg = np.mean(episode_rewards[-10:])
            print(f"      └─ Last 10 episodes average: {recent_avg:.2f}")
    
    return episode_rewards, episode_actions

def plot_results(episode_rewards):
    """Create and save plots"""
    episodes = range(1, len(episode_rewards) + 1)
    
    plt.figure(figsize=(12, 8))
    
    # Main reward plot
    plt.subplot(2, 2, 1)
    plt.plot(episodes, episode_rewards, 'b-', alpha=0.6, linewidth=1, label='Episode Reward')
    
    # Add moving average
    window = 10
    if len(episode_rewards) >= window:
        moving_avg = []
        for i in range(window-1, len(episode_rewards)):
            moving_avg.append(np.mean(episode_rewards[i-window+1:i+1]))
        plt.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, label=f'{window}-Episode Moving Average')
    
    plt.title('Reward vs Epoch (Episode)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch (Episode)')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Cumulative reward
    plt.subplot(2, 2, 2)
    cumulative = np.cumsum(episode_rewards)
    plt.plot(episodes, cumulative, 'g-', linewidth=2)
    plt.title('Cumulative Reward', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('Cumulative Reward')
    plt.grid(True, alpha=0.3)
    
    # Reward distribution
    plt.subplot(2, 2, 3)
    plt.hist(episode_rewards, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Reward Distribution', fontsize=12)
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Learning progress (improvement over time)
    plt.subplot(2, 2, 4)
    if len(episode_rewards) > 20:
        early_avg = np.mean(episode_rewards[:20])
        late_avg = np.mean(episode_rewards[-20:])
        improvement = late_avg - early_avg
        
        plt.bar(['Early Episodes\n(1-20)', 'Recent Episodes\n(-20 to end)'], 
               [early_avg, late_avg], 
               color=['lightcoral', 'lightgreen'],
               alpha=0.8)
        plt.title(f'Learning Progress\nImprovement: {improvement:.2f}', fontsize=12)
        plt.ylabel('Average Reward')
        
        # Add improvement text
        plt.text(0.5, max(early_avg, late_avg) * 0.5, 
                f'Δ = {improvement:+.2f}', 
                ha='center', va='center', 
                fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('reward_vs_epoch.png', dpi=200, bbox_inches='tight')
    print(f"\n📊 Plot saved as 'reward_vs_epoch.png'")
    
    return episodes, episode_rewards

def main():
    """Main training function"""
    print("🎯 Training agent with real-time reward tracking...")
    
    # Train the agent
    episode_rewards, episode_actions = train_agent(n_episodes=100)
    
    # Show final statistics
    print("\n" + "="*50)
    print("🎉 Training completed!")
    print(f"📈 Final Statistics:")
    print(f"   • Total episodes: {len(episode_rewards)}")
    print(f"   • Final episode reward: {episode_rewards[-1]:.2f}")
    print(f"   • Best episode reward: {max(episode_rewards):.2f}")
    print(f"   • Worst episode reward: {min(episode_rewards):.2f}")
    print(f"   • Average reward: {np.mean(episode_rewards):.2f}")
    print(f"   • Standard deviation: {np.std(episode_rewards):.2f}")
    
    # Show improvement
    if len(episode_rewards) >= 20:
        early_avg = np.mean(episode_rewards[:10])
        late_avg = np.mean(episode_rewards[-10:])
        improvement = late_avg - early_avg
        print(f"   • Learning improvement: {improvement:+.2f}")
        
        if improvement > 0:
            print("   ✅ Agent is learning and improving!")
        else:
            print("   📈 Agent needs more training time")
    
    # Create plots
    episodes, rewards = plot_results(episode_rewards)
    
    print(f"\n✨ All done! Check 'reward_vs_epoch.png' for detailed analysis.")
    
    return episode_rewards

if __name__ == "__main__":
    main()
