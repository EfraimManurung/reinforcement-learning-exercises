from ray import tune
import gym 

from ray.rllib.algorithms.ppo import ppo
config = {
    "env": "CarRacing-v0",
    "framework": "torch",  # or "tf" for TensorFlow
    # Other RLlib parameters such as training parameters, neural network architecture, etc.
}

# Optional: Additional parameters can be added, such as the neural network architecture.

analysis = tune.run(
    "PPO",  # or any other RL algorithm supported by RLlib
    config=config,
    stop={"episode_reward_mean": 200},  # Stop training when the mean reward reaches a certain threshold
    num_samples=1,  # Number of training runs
    checkpoint_at_end=True,  # Save the checkpoint at the end of training
    local_dir="~/ray_results"  # Directory to store training results
)

checkpoint_path = analysis.get_best_checkpoint(trial=analysis.get_best_trial(metric="episode_reward_mean"))
agent = ppo.PPOTrainer(config=config)
agent.restore(checkpoint_path)

env = gym.make("CarRacing-v0")
obs = env.reset()
done = False
while not done:
    action = agent.compute_action(obs)
    obs, reward, done, _ = env.step(action)
    env.render()
