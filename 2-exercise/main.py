'''
RL for FrozenPond case

Author: Efraim Manurung
MSc Thesis in Information Technology Group, Wageningen University

efraim.efraimpartoginahotasi@wur.nl
efraim.manurung@gmail.com

Main program
'''

from FrozenPond import FrozenPond

from ray.rllib.algorithms.ppo import PPOConfig

# Create an RLlib Algorithm instance from a PPOConfig object.
config = (
    PPOConfig().environment(
        # Env class to use (here: our gym.Env sub-class from above).
        env=FrozenPond
    )
    # Parallelize environment rollouts.
    .env_runners(num_env_runners=3)
)

# Construct the actual (PPO) algorithm object from the config.
algo = config.build()

for i in range(5):
    results = algo.train()
    print(f"Iter: {i}; avg. return={results['env_runners']['episode_return_mean']}")
    
