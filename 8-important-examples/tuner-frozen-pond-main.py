'''
RL for FrozenPond case

Author: Efraim Manurung
MSc Thesis in Information Technology Group, Wageningen University

efraim.efraimpartoginahotasi@wur.nl
efraim.manurung@gmail.com

Main program
'''

# Import supporting libraries
import sys
import os

# Import libraries needed for PPO algorithm
from ray.rllib.algorithms.ppo import PPOConfig

# Import tuner
from ray import air
from ray import tune

# Import environment
from classes.FrozenPond import FrozenPond

# Setup
config = PPOConfig()

# Update the config object with training parameters and environment parameters
config = config.training(
    lr=tune.grid_search([0.001]), clip_param=0.2
).environment(
    env=FrozenPond,
    env_config={"size": 4}  # You can change the size as needed
).env_runners(num_env_runners=3)

# Use to_dict() to get the old-style python config dict
# when running with tune.
tuner = tune.Tuner(
    "PPO",
    run_config=air.RunConfig(stop={"training_iteration": 1}),
    param_space=config.to_dict(),
)

results = tuner.fit()
