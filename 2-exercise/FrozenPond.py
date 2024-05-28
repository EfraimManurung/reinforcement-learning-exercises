'''
Mostly we called the environment from the gym API  which is the FrozenLake-v1

Components of an environment
Conceptual decision:
- Observation space
- Action space
- Rewards

In Python we will need to implement, at least:
- Constructor
- reset()
- step()

In practice, we may also want other methods, such as render()
'''
# General libraries
import numpy as np
import distutils.spawn
# import utils

# Let us start do the code
import gym

# We start by subclassing gym.Env
# The concept about objects, inheritance, and subclasses
# This is a basic gym.Env and we can overwrite features of it

class FrozenPond(gym.Env):
    
    # The constructor gets called when we make a new FrozenPond object
    # Here is where we define the observation space and action space
    def __init__(self, env_config=None):
        if env_config is None:
            env_config = dict()
            
        self.size = env_config.get("size", 4)
        self.observation_space = gym.spaces.Discrete(self.size*self.size)
        self.action_space = gym.spaces.Discrete(self.size)
    
    # We need reset method
    # The constructor sets permanent parameters like the observation space
    # reset sets up each new episode
    # There is some freedom between the two, e.g. setting the goal location
    # If something could change, we'll put it in reset
    def reset(self):
        self.player = (0, 0)    # the playe starts at the top-left
        self.goal = (self.size-1, self.size-1)   # goal is at the bottom-right
        
        if self.size == 4:
            self.holes = np.array([
                [0, 0, 0, 0],  # FFFF
                [0, 1, 0, 1],  # FHFH
                [0, 0, 0, 1],  # FFFH
                [1, 0, 0, 0]   # HFFF
            ])
        else:
            raise Exception("Frozen Pond only supports size 4")
        
        # return 0 # to be changed to return self.observation()
        return self.observation()
    
    # Recall in default the observation is an index from 0 to 15
    # For example, if the player is at (2,1) then we return 
    # 4 * 2 + 1 = 9
    # So that is represent the position of player
    def observation(self):
        return self.size*self.player[0] + self.player[1]
    
    # Following the Frozen Lake example, the reward will be 1 if the agent reaches the goal, otherwise 0
    def reward(self):
        return int(self.player == self.goal) # return 1 if the position of player and goal is equal
    
    # We also need to know when an episode is done
    # Following Frozen Lake, the episode is done when the agent reaches the goal or falls into the pond
    def done(self):
        return self.player == self.goal or bool(self.holes[self.player] == 1)
        # cast from numpy.bool to bool because of the RLlib check_env
    
    # Finally, to make the step method simpler, we'll write a helper method called is_valid_loc
    # that checks whether a particular location is in bound (from 0 to 3 in each dimension).
    def is_valid_loc(self, location):
        return 0 <= location[0] < self.size and 0 <= location[1] < self.size
    
    # The last method we need is step
    # This is the most complicated method that contain sthe core logic
    # Recall that step returns 4 things:
    # 1. Observation
    # 2. Reward
    # 3. Done flag
    # 4. Extra info (we will ignore) or to be implemented
    # For clarity, we'll write helper methods for observation, reward, and done, plus one extra helper method
    
    # Using the aboves pieces, we can now write the step method.
    # step takes in an action, updates the state, and returns the observation, reward, done flag,
    # and extra info (ignored).
    # Recall how actions are encoded: 0 for left, 1 for down, 2 for right, 3 for up.
    # We will implement a non-slippery frozen pond; in other words, deterministic rather than stochastic.
    def step(self, action):
        # Compute the new player location
        if action == 0:     # left
            new_loc = (self.player[0], self.player[1]-1)
        elif action == 1:   # down
            new_loc = (self.player[0]+1, self.player[1])
        elif action == 2:   # right
            new_loc = (self.player[0], self.player[1]+1)
        elif action == 3:   # up
            new_loc = (self.player[0]-1, self.player[1])
        else:
            raise ValueError("Action must be in {0,1,2,3}")
        
        # Update the player location only if you stayed in bounds
        # (if you try to move out of bounds, the action does nothing)
        if self.is_valid_loc(new_loc):
            self.player = new_loc
        
        # Return observation/reward/done
        return self.observation(), self.reward(), self.done(), {"player" : self.player, "goal": self.goal}
    
    # That is it! We have implemented the necessary pieces in Frozen Pond:
    #   - constructor
    #   - reset
    #   - step
    # We will also add an optional render function so that we can draw the state
    def render(self):
        for i in range(self.size):
            for j in range(self.size):
                if (i,j) == self.player:
                    print("🧑", end="")
                elif (i,j) == self.goal:
                    print("⛳️", end="")
                elif self.holes[i,j]:
                    print("🕳", end="")
                else:
                    print("🧊", end="")
            print()