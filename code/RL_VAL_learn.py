import pandas as pd
import numpy as np
import random

# 1. Read pre-generated demand data from CSV
demand_file = 'demand.csv'
demands = pd.read_csv(demand_file)['demand'].values

# 2. Define the inventory environment that uses CSV data
class InventoryEnv:
    """Inventory environment using demand from CSV."""
    def __init__(self, demands, max_inventory=50,
                 holding_cost=0.1, stockout_penalty=5):
        self.demands = demands
        self.max_inventory = max_inventory
        self.holding_cost = holding_cost
        self.stockout_penalty = stockout_penalty
        self.reset()

    def reset(self):
        self.inventory = self.max_inventory // 2
        self.step_idx = 0
        return self.inventory

    def step(self, action):
        # Reorder
        self.inventory = min(self.max_inventory, self.inventory + action)
        # Fetch demand from CSV sequence
        demand = int(self.demands[self.step_idx])
        self.step_idx += 1
        # Compute sales and leftover inventory
        sales = min(self.inventory, demand)
        self.inventory -= sales
        # Reward calculation
        reward = (
            sales
            - self.stockout_penalty * max(0, demand - sales)
            - self.holding_cost * self.inventory
        )
        return self.inventory, reward

# 3. Q-learning training function
def train_q_table(env, episodes=300, steps_per_episode=50,
                  alpha=0.1, gamma=0.95,
                  epsilon_start=1.0, epsilon_decay=0.995):
    actions = [0, 10, 20, 30, 40]
    Q = np.zeros((env.max_inventory + 1, len(actions)))
    epsilon = epsilon_start

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(steps_per_episode):
            # ε-greedy action
            if random.random() < epsilon:
                a_idx = random.randrange(len(actions))
            else:
                a_idx = int(Q[state].argmax())
            action = actions[a_idx]

            next_state, reward = env.step(action)
            total_reward += reward

            # Q update
            best_next = Q[next_state].max()
            Q[state, a_idx] += alpha * (
                reward + gamma * best_next - Q[state, a_idx]
            )
            state = next_state

        epsilon = max(0.1, epsilon * epsilon_decay)

        # Print progress
        if ep < 10 or (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{episodes} | "
                  f"Reward: {total_reward:.2f} | "
                  f"Epsilon: {epsilon:.3f}")

    return Q, actions

# 4. Evaluation function
def evaluate(env, policy, steps=1000):
    state = env.reset()
    total = 0.0
    for _ in range(steps):
        action = policy(state)
        state, reward = env.step(action)
        total += reward
    return total / steps

# 5. Run training and evaluation
env = InventoryEnv(demands)
Q, actions = train_q_table(env)

# Display Q-values for select states
for s in [0, 25, 50]:
    print(f"Q[{s}] = {Q[s]}")

# Define policies
q_policy = lambda s: actions[int(Q[s].argmax())]
naive_policy = lambda s: 20

# Final performance
naive_avg = evaluate(env, naive_policy)
q_avg = evaluate(env, q_policy)

print("\nFinal Performance:")
print(f"  Naïve policy (reorder 20 units): {naive_avg:.3f} avg reward/step")
print(f"  Q-Learning policy:             {q_avg:.3f} avg reward/step")
