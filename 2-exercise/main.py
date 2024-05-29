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
from IPython import display
import time

# Import libraries needed for PPO algorithm
import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig

# Set the console encoding to UTF-8
if os.name == 'nt':
    os.system('chcp 65001')
    sys.stdout.reconfigure(encoding='utf-8')

# Define our problem using python and Farama-Foundation's gymnasium API
class FrozenPond(gym.Env):
    """
    FrozenPond environment, a custom environment similar to FrozenLake-v1 from Gym.
    """

    def __init__(self, config):
        """
        Initialize the FrozenPond environment.
        
        Parameters:
        env_config (dict): Configuration dictionary for the environment.
        """
        self.size = config["size"] # env_config.get("size", 4)
        
        self.goal = (self.size - 1, self.size - 1)  # Goal is at the bottom-right
        self.player = (0, 0)  # The player starts at the top-left
        
        self.action_space = gym.spaces.Discrete(4) # up, down, left, right
        self.observation_space = gym.spaces.Discrete(self.size * self.size)
        
    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to the initial state.
        
        Returns:
        int: The initial observation of the environment.
        """
        self.player = (0, 0)  # The player starts at the top-left
        self.goal = (self.size - 1, self.size - 1)  # Goal is at the bottom-right
        
        if self.size == 4:
            self.holes = np.array([
                [0, 0, 0, 0],  # FFFF
                [0, 1, 0, 1],  # FHFH
                [0, 0, 0, 1],  # FFFH
                [1, 0, 0, 0]   # HFFF
            ])
        else:
            raise Exception("Frozen Pond only supports size 4")
        
        return self.observation(), {}

    def observation(self):
        """
        Get the current observation of the environment.
        
        Returns:
        int: The index representing the player's position.
        """
        return self.size * self.player[0] + self.player[1]

    def reward(self):
        """
        Get the reward for the current state.
        
        Returns:
        int: Reward, 1 if the player reaches the goal, otherwise 0.
        """
        return int(self.player == self.goal)

    def done(self):
        """
        Check if the episode is done.
        
        Returns:
        bool: True if the episode is done, otherwise False.
        """
        return self.player == self.goal or bool(self.holes[self.player] == 1)

    def is_valid_loc(self, location):
        """
        Check if the given location is valid (within bounds).
        
        Parameters:
        location (tuple): The location to check.
        
        Returns:
        bool: True if the location is valid, otherwise False.
        """
        return 0 <= location[0] < self.size and 0 <= location[1] < self.size

    def step(self, action):
        """
        Take a step in the environment.
        
        Parameters:
        action (int): The action to take (0=left, 1=down, 2=right, 3=up).
        
        Returns:
        tuple: A tuple containing the new observation, reward, done flag, and additional info.
        
        New:
        New observation, reward, terminated-flag, truncated-flag, info-dict(not-empty).
        """
        if action == 0:  # left
            new_loc = (self.player[0], self.player[1] - 1)
        elif action == 1:  # down
            new_loc = (self.player[0] + 1, self.player[1])
        elif action == 2:  # right
            new_loc = (self.player[0], self.player[1] + 1)
        elif action == 3:  # up
            new_loc = (self.player[0] - 1, self.player[1])
        else:
            raise ValueError("Action must be in {0,1,2,3}")

        if self.is_valid_loc(new_loc):
            self.player = new_loc
        
        truncated = False
        
        return self.observation(), self.reward(), self.done(), truncated, {"player": self.player, "goal": self.goal}

    def render(self):
        """
        Render the current state of the environment.
        """
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) == self.player:
                    print("🧑", end="")
                elif (i, j) == self.goal:
                    print("⛳️", end="")
                elif self.holes[i, j]:
                    print("🕳", end="")
                else:
                    print("🧊", end="")
            print()

# RL Configuration and Training
config = (
    PPOConfig().environment(
        env=FrozenPond,
        # Config dict to be passed to our custom env's constructor.
        env_config={"size": 4},
    )
    # Parallelize environment rollouts.
    .env_runners(num_env_runners=3)
)

# Construct the PPO algorithm object from the config
algo = config.build()

# Train the model for a number of iterations
for i in range(10):
    results = algo.train()
    print(f"Iter: {i}; avg. rewards={results['env_runners']['episode_return_mean']}")

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
    display.clear_output(wait=True)
    env.render()
    time.sleep(0.5)
    
    # sum up rewards for reporting purposes
    total_reward += reward

# Report results.
print(f"Played 1 episode; total-reward={total_reward}")
