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
    
    