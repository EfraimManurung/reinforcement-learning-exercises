'''
You can pass either a string name or a Python class to specify an environment. By default, strings will be interpreted as a gym
environment name. Custom env classes passed directly to the algorithm must take asingle env_config parameters
in their constructor.
'''

import gymnasium as gym
import ray
from ray.rllib.algorithms import ppo

class MyEnv(gym.Env):
    def __init__(self, env_config):
        self.action_space = <gym.space>
        self.observation_space = <gym.space>
        
    def reset(self, seed, options):
        return <obs>, <info>
    
    def step(self, action):
        return <obs>, <reward: float>, <terminated: bool>, <truncated: bool>, <info: dict>
    
ray.init()
algo = ppo.PPO(env=MyEnv, config={
    "env_config": {}, # config to pass to env class
})

while True:
    print(algo.train())
        