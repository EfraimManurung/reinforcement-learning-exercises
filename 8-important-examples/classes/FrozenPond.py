# Import supporting libraries
import numpy as np
import sys
import os
import time

# Import libraries needed for PPO algorithm
import gymnasium as gym


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