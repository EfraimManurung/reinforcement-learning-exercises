from ray.rllib.algorithms.ppo import PPOConfig

# Create a PPOConfig object and set parameters
config = PPOConfig()
config = config.training(gamma=0.9, lr=0.01, kl_coeff=0.3, train_batch_size=128)
config = config.resources(num_gpus=0)
config = config.rollouts(num_envs_per_worker=1)

# Build an Algorithm object from the config and run 1 training iteration
algo = config.build(env="CartPole-v1")
algo.train()
