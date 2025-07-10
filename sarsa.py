import numpy as np
import random
import time

class GridWorld:
    def __init__(self, p1=1.0, p2=0):
        self.rows = 10
        self.cols = 10
        self.actions = ["up", "down", "right", "left"]
        self.goal_state = (9, 9)  # G state at (10,10) in 1-indexed coordinates
        # Walls between rooms (vertical and horizontal lines)
        self.walls = {((i, 2), (i, 3)) for i in range(10)} | {((3, i), (4, i)) for i in range(10)}
        # Doors in the walls
        self.doors = {((3, 2), (3, 3)), ((8, 3), (8, 4))}
        self.p1 = p1  # Probability of moving in the chosen direction
        self.p2 = p2  # Probability of staying in the same state
        self.p3 = 1 - p1 - p2  # Probability of moving to adjacent states

    def get_next_state(self, state, action):
        """Get the next state given the current state and action with stochastic transitions"""
        x, y = state
        
        # Calculate the intended next state
        if action == "up":
            next_x, next_y = x, y + 1
        elif action == "down":
            next_x, next_y = x, y - 1
        elif action == "right":
            next_x, next_y = x + 1, y
        elif action == "left":
            next_x, next_y = x - 1, y
        
        # Check if the next state is valid (within grid and not through a wall)
        is_valid = (0 <= next_x < self.cols and 0 <= next_y < self.rows and 
                   not (((x, y), (next_x, next_y)) not in self.doors and 
                        (((x, y), (next_x, next_y)) in self.walls or 
                         ((next_x, next_y), (x, y)) in self.walls)))
        
        # Get possible adjacent states
        adjacent_states = []
        for adj_action in self.actions:
            if adj_action == action:
                continue
            
            if adj_action == "up":
                adj_x, adj_y = x, y + 1
            elif adj_action == "down":
                adj_x, adj_y = x, y - 1
            elif adj_action == "right":
                adj_x, adj_y = x + 1, y
            elif adj_action == "left":
                adj_x, adj_y = x - 1, y
            
            # Check if the adjacent state is valid
            is_adj_valid = (0 <= adj_x < self.cols and 0 <= adj_y < self.rows and 
                          not (((x, y), (adj_x, adj_y)) not in self.doors and 
                               (((x, y), (adj_x, adj_y)) in self.walls or 
                                ((adj_x, adj_y), (x, y)) in self.walls)))
            
            if is_adj_valid:
                adjacent_states.append((adj_x, adj_y))
        
        # Stochastic transition
        rand_val = random.random()
        if not is_valid:  # If intended state is invalid, stay in the same state
            return state, self.get_reward(state)
        
        if rand_val < self.p1:  # Move in the intended direction
            next_state = (next_x, next_y)
        elif rand_val < self.p1 + self.p2:  # Stay in the same state
            next_state = state
        else:  # Move to one of the adjacent states
            if adjacent_states:
                next_state = random.choice(adjacent_states)
            else:
                next_state = state
        
        return next_state, self.get_reward(next_state)
    
    def get_reward(self, state):
        """Return the reward for the given state"""
        if state == self.goal_state:
            return 1.0
        else:
            return -1.0
    
    def is_terminal(self, state):
        """Check if the state is terminal"""
        return state == self.goal_state
    
    def random_state(self):
        """Generate a random state in the grid"""
        return (random.randint(0, self.cols - 1), random.randint(0, self.rows - 1))


class SarsaAgent:
    def __init__(self, env, gamma=0.9, epsilon=0.1, alpha=0.1):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.q = {}  # Q-values
        
        # Initialize Q-values
        for x in range(env.cols):
            for y in range(env.rows):
                for action in env.actions:
                    self.q[((x, y), action)] = 0.0
    
    def choose_action(self, state, epsilon=None):
        """Choose an action using epsilon-greedy policy"""
        if epsilon is None:
            epsilon = self.epsilon
            
        if random.random() < epsilon:
            return random.choice(self.env.actions)
        else:
            q_values = [self.q[(state, a)] for a in self.env.actions]
            max_q = max(q_values)
            count = q_values.count(max_q)
            if count > 1:
                best_actions = [a for a, q in zip(self.env.actions, q_values) if q == max_q]
                return random.choice(best_actions)
            else:
                return self.env.actions[q_values.index(max_q)]
    
    def run_episode(self):
        """Run one episode and update Q-values using SARSA"""
        state = self.env.random_state()
        action = self.choose_action(state)
        
        steps = 0
        while not self.env.is_terminal(state):
            next_state, reward = self.env.get_next_state(state, action)
            
            # Choose the next action using the same policy
            next_action = self.choose_action(next_state) if not self.env.is_terminal(next_state) else None
            
            # SARSA update rule
            if next_action:
                self.q[(state, action)] += self.alpha * (
                    reward + self.gamma * self.q[(next_state, next_action)] - self.q[(state, action)]
                )
            else:  # Terminal state
                self.q[(state, action)] += self.alpha * (reward - self.q[(state, action)])
            
            state = next_state
            action = next_action
            steps += 1
            
            # Break if episode takes too long
            if steps > 1000:
                break
        
        return steps
    
    def get_policy(self):
        """Get the optimal policy based on current Q-values"""
        policy = {}
        for x in range(self.env.cols):
            for y in range(self.env.rows):
                state = (x, y)
                q_values = [self.q[(state, a)] for a in self.env.actions]
                max_q = max(q_values)
                best_actions = [a for a, q in zip(self.env.actions, q_values) if q == max_q]
                policy[state] = random.choice(best_actions)
        return policy


def main():
    # Initialize with p1=1.0 and p2=0
    env = GridWorld(p1=1.0, p2=0)
    
    # Get p1, p2 from user input if needed
    try:
        p1 = float(input("Enter p1 (probability of moving in chosen direction): "))
        p2 = float(input("Enter p2 (probability of staying in same state): "))
        env = GridWorld(p1=p1, p2=p2)
    except:
        print("Using default values: p1=1.0, p2=0.0")
    
    # Test different values of alpha
    alphas = [0.05, 0.1, 0.2]
    num_episodes = 10000
    
    results = {}
    
    for alpha in alphas:
        print(f"Running SARSA with alpha = {alpha}")
        
        agent = SarsaAgent(env, gamma=0.9, epsilon=0.1, alpha=alpha)
        start_time = time.time()
        
        episode_lengths = []
        for i in range(num_episodes):
            length = agent.run_episode()
            episode_lengths.append(length)
            
            # Print progress
            if (i + 1) % 1000 == 0:
                print(f"Episode {i + 1}/{num_episodes}, Avg Length: {sum(episode_lengths[-100:]) / 100:.2f}")
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        policy = agent.get_policy()
        
        # Calculate average steps per episode
        avg_steps = sum(episode_lengths) / len(episode_lengths)
        
        results[alpha] = {
            "avg_steps": avg_steps,
            "computation_time": computation_time,
            "policy": policy
        }
        
        print(f"Alpha: {alpha}, Avg Steps: {avg_steps:.2f}, Time: {computation_time:.2f}s")
    
    # Test with decreasing epsilon
    print("Running SARSA with decreasing epsilon")
    agent = SarsaAgent(env, gamma=0.9, epsilon=0.1, alpha=0.1)
    start_time = time.time()
    
    episode_lengths = []
    for i in range(num_episodes):
        # Decrease epsilon over time
        current_epsilon = max(0.01, 0.1 * (1 - i / num_episodes))
        agent.epsilon = current_epsilon
        
        length = agent.run_episode()
        episode_lengths.append(length)
        
        # Print progress
        if (i + 1) % 1000 == 0:
            print(f"Episode {i + 1}/{num_episodes}, Epsilon: {current_epsilon:.4f}, Avg Length: {sum(episode_lengths[-100:]) / 100:.2f}")
    
    end_time = time.time()
    computation_time = end_time - start_time
    
    policy = agent.get_policy()
    
    # Calculate average steps per episode
    avg_steps = sum(episode_lengths) / len(episode_lengths)
    
    results["decreasing_epsilon"] = {
        "avg_steps": avg_steps,
        "computation_time": computation_time,
        "policy": policy
    }
    
    print(f"Decreasing Epsilon, Avg Steps: {avg_steps:.2f}, Time: {computation_time:.2f}s")
    
    # Print the optimal policy for the best parameters
    best_alpha = min(results, key=lambda x: results[x]["avg_steps"])
    print(f"Best parameters - Alpha: {best_alpha}")
    
    # Print the optimal policy in a grid format
    policy = results[best_alpha]["policy"]
    for y in range(env.rows - 1, -1, -1):
        row = ""
        for x in range(env.cols):
            if (x, y) == env.goal_state:
                row += "G  "
            else:
                action = policy[(x, y)]
                if action == "up":
                    row += "↑  "
                elif action == "down":
                    row += "↓  "
                elif action == "right":
                    row += "→  "
                elif action == "left":
                    row += "←  "
        print(row)


if __name__ == "__main__":
    main()
