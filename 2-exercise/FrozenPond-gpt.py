import numpy as np
import gym

class FrozenPond(gym.Env):
    def __init__(self, env_config=None):
        if env_config is None:
            env_config = dict()
        self.size = env_config.get("size", 4)
        self.observation_space = gym.spaces.Discrete(self.size*self.size)
        self.action_space = gym.spaces.Discrete(4)
        self.reset()

    def reset(self):
        self.player = (0, 0)
        self.goal = (self.size-1, self.size-1)
        self.holes = np.zeros((self.size, self.size))
        if self.size == 4:
            self.holes[1, 1] = 1
            self.holes[1, 3] = 1
            self.holes[2, 3] = 1
            self.holes[3, 0] = 1
        return self._get_observation()

    def _get_observation(self):
        return self.size * self.player[0] + self.player[1]

    def _get_reward(self):
        return int(self.player == self.goal)

    def _is_done(self):
        return self.player == self.goal or self.holes[self.player] == 1

    def _is_valid_location(self, location):
        return 0 <= location[0] < self.size and 0 <= location[1] < self.size

    def step(self, action):
        if action == 0:
            new_location = (self.player[0], self.player[1]-1)
        elif action == 1:
            new_location = (self.player[0]+1, self.player[1])
        elif action == 2:
            new_location = (self.player[0], self.player[1]+1)
        elif action == 3:
            new_location = (self.player[0]-1, self.player[1])
        else:
            raise ValueError("Action must be in {0, 1, 2, 3}")

        if self._is_valid_location(new_location):
            self.player = new_location

        observation = self._get_observation()
        reward = self._get_reward()
        done = self._is_done()
        return observation, reward, done, {}

    def render(self, mode='human'):
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) == self.player:
                    print("🧑", end="")
                elif (i, j) == self.goal:
                    print("⛳️", end="")
                elif self.holes[i, j]:
                    print("🕳", end="")
                else:
                    print("🧊", end="")
            print()
