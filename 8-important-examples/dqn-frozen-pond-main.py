'''
RL for FrozenPond case

Author: Efraim Manurung
MSc Thesis in Information Technology Group, Wageningen University

efraim.efraimpartoginahotasi@wur.nl
efraim.manurung@gmail.com

Main program
'''

# Import supporting libraries
import numpy as np
import sys
import os
import time

# Import libraries needed for PPO algorithm
from ray.rllib.algorithms.dqn.dqn import DQNConfig, DQN

from classes.FrozenPond import FrozenPond

# Set the console encoding to UTF-8
if os.name == 'nt':
    os.system('chcp 65001')
    sys.stdout.reconfigure(encoding='utf-8')

# Clearing the output based on the operating system
def clear_output():
    os.system('cls' if os.name == 'nt' else 'clear')

config = DQNConfig()

replay_config = {
        "type": "MultiAgentPrioritizedReplayBuffer",
        "capacity": 60000,
        "prioritized_replay_alpha": 0.5,
        "prioritized_replay_beta": 0.5,
        "prioritized_replay_eps": 3e-6,
    }

config = config.training(replay_buffer_config=replay_config)
config = config.resources(num_gpus=0)
config = config.env_runners(num_env_runners=1)
config = config.environment(
        env=FrozenPond,
        # Config dict to be passed to our custom env's constructor.
        env_config={"size": 4},
    )

# Construct the PPO algorithm object from the config
algo = config.build()

# Train the model for a number of iterations
for i in range(10):
    results = algo.train()
    print(f"Iter: {i}; avg. rewards={results['env_runners']['episode_return_mean']}")

# call `save()` to create a checkpoint.
save_result = algo.save('model/frozen-pond-dqn-model')

path_to_checkpoint = save_result.checkpoint.path
print(
    "An Algorithm checkpoint has been created inside directory: "
    f"'{path_to_checkpoint}'."
)

env = FrozenPond({"size": 4})

# Get the initial observation (should be: [0.0] for the starting position).
obs, info = env.reset()
done = False
total_reward = 0.0

# Play one episode
while not done:
    # Compute a single action, given the current observation
    # from the environment.
    action = algo.compute_single_action(obs)
    
    # Apply the computed action in the environment.
    obs, reward, done, _, info = env.step(action)
    
    # Render the step
    # clear_output()
    env.render()
    time.sleep(0.5)
    
    # sum up rewards for reporting purposes
    total_reward += reward

# Report results.
print(f"Played 1 episode; total-reward={total_reward}")
