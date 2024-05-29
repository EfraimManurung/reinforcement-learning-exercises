from ray.rllib.algorithms.algorithm import Algorithm

# Import supporting libraries
import sys
import os
import time

# Import libraries needed for PPO algorithm
from ray.rllib.algorithms.ppo import PPOConfig

# Import libraries
from classes.FrozenPond import FrozenPond

# Set the console encoding to UTF-8
if os.name == 'nt':
    os.system('chcp 65001')
    sys.stdout.reconfigure(encoding='utf-8')

# Clearing the output based on the operating system
def clear_output():
    os.system('cls' if os.name == 'nt' else 'clear')

# Use the Algorithm's `from_checkpoint` utility to get a new algo instance
# that has the exact same state as the old one, from which the checkpoint was
# created in the first place:

my_new_ppo = Algorithm.from_checkpoint('model/model-frozen-pond')

env = FrozenPond({"size": 4})

# Get the initial observation (should be: [0.0] for the starting position).
obs, info = env.reset()
terminated = truncated = False
total_reward = 0.0

# Play one episode
while not terminated and not truncated:
    # Render the step
    clear_output()
    env.render()
    
    # Make some delay, so it is easier to see
    time.sleep(1.0)
    
    # Compute a single action, given the current observation
    # from the environment.
    action = my_new_ppo.compute_single_action(obs)
    
    # Apply the computed action in the environment.
    obs, reward, terminated, _, info = env.step(action)

    # sum up rewards for reporting purposes
    total_reward += reward

# Report results.
print(f"Played 1 episode; total-reward={total_reward}")