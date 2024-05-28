import gym
import ray
from ray import tune
from ray.rllib.algorithms import PPO
from ray.tune.registry import register_env

from FrozenPond import FrozenPond

# Register the custom environment
def frozen_pond_env_creator(env_config):
    return FrozenPond(env_config)

register_env("FrozenPond-v0", frozen_pond_env_creator)

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Set up the PPO trainer
config = {
    "env": "FrozenPond-v0",
    "env_config": {"size": 4},  # Passing the environment configuration
    "framework": "tf",  # You can use "torch" if you prefer PyTorch
    "num_workers": 1,  # Number of parallel environments
    "num_envs_per_worker": 1,
    "train_batch_size": 4000,
    "sgd_minibatch_size": 64,
    "num_sgd_iter": 10,
    "rollout_fragment_length": 200,
    "model": {
        "fcnet_hiddens": [256, 256],  # Neural network hidden layers
        "fcnet_activation": "tanh",
    },
    "lr": 5e-4,  # Learning rate
    "gamma": 0.99,  # Discount factor
}

# Create the trainer
trainer = PPO(config=config)

# Train the policy
for i in range(100):
    result = trainer.train()
    print(f"Iteration: {i}, Reward: {result['episode_reward_mean']}")

    # Save the model every 10 iterations
    if i % 10 == 0:
        checkpoint = trainer.save()
        print(f"Checkpoint saved at {checkpoint}")

# Evaluate the trained policy
env = FrozenPond()
obs = env.reset()
done = False
total_reward = 0

while not done:
    action = trainer.compute_single_action(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    total_reward += reward

print(f"Total reward: {total_reward}")

# Shut down Ray
ray.shutdown()
