'''
RL for cartpole case

Author: Efraim Manurung
MSc Thesis in Information Technology Group, Wageningen University

efraim.efraimpartoginahotasi@wur.nl
efraim.manurung@gmail.com

Main program

Project sources:
- Algorithm: Model-free Off-policy RL 
  Deep Q Networks (DQN, Rainbow, Parametric DQN)
  https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#dqn

- Environment: CartPole gym environment that has a dict observation space.
  rllib/examples/envs/classes/cartpole_with_dict_observation_space.py
  https://github.com/ray-project/ray/blob/master/rllib/examples/envs/classes/cartpole_with_dict_observation_space.py
  

'''
# Import environment class
from classes.CartPoleWithDictObservationSpace import CartPoleWithDictObservationSpace

# Import algorithm
from ray.rllib.algorithms.dqn.dqn import DQNConfig, DQN

config = DQNConfig()

replay_config = {
        "type": "MultiAgentPrioritizedReplayBuffer",
        "capacity": 60000,
        "prioritized_replay_alpha": 0.5,
        "prioritized_replay_beta": 0.5,
        "prioritized_replay_eps": 3e-6,
    }

config = config.training(replay_buffer_config=replay_config)
config = config.resources(num_gpus=0)
config = config.env_runners(num_env_runners=1)
config = config.environment(env=CartPoleWithDictObservationSpace)

algo = DQN(config=config)

# Train the model for a number of iterations
for i in range(10):
    results = algo.train()
    print(f"Iter: {i}; avg. rewards={results['env_runners']['episode_return_mean']}")

# call `save()` to create a checkpoint.
save_result = algo.save('model/cartpole-model')

path_to_checkpoint = save_result.checkpoint.path
print(
    "An Algorithm checkpoint has been created inside directory: "
    f"'{path_to_checkpoint}'."
)
