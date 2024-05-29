from gym.envs.classic_control import CartPoleEnv
from gym.wrappers import TimeLimit

import utils

class MyCartPole(TimeLimit):
    def __init__(self, env_config=None):
        if isinstance(env_config, dict):
            env = CartPoleEnv(**env_config)
        else:
            env = CartPoleEnv()
            
        super().__init__(env, max_episode_steps=500)
        
    def render(self):
        utils.my_render_cartpole_matplotlib(self)
