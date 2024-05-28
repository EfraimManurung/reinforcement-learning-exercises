'''
RL for FrozenPond case

Author: Efraim Manurung
MSc Thesis in Information Technology Group, Wageningen University

efraim.efraimpartoginahotasi@wur.nl
efraim.manurung@gmail.com

Main program
'''

import sys
import os

from ray.rllib.algorithms.ppo import ppo, PPOConfig
from IPython import display

default_config = (
    PPOConfig()
    .framework("tf")
    .rollouts(create_env_on_local_worker=True)
    .debugging(seed=0, log_level="ERROR")
    .training(model={"fcnet_hiddens": [32, 32]})
)

# Set the console encoding to UTF-8
if os.name == 'nt':
    os.system('chcp 65001')
    sys.stdout.reconfigure(encoding='utf-8')

from FrozenPond import FrozenPond

algo = ppo.PPO(env=FrozenPond, config=default_config)

env = FrozenPond()
obs = env.reset()

done = False

for i in range(10):
    
    # action = algo.compute_single_action(obs, explore=False)
    action = algo.compute_single_action(obs, explore=False)
    obs, rewards, done, _ = env.step(action)
    
    display.clear_output(wait=True)
    env.render()
    print(i+1, done)

ppo.stop()