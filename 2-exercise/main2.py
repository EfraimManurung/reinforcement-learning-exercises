import gym
from FrozenPond import FrozenPond

lake = gym.make("FrozenLake-v1", is_slippery=False)
pond = FrozenPond()

lake.reset()
pond.reset()

print("Iter | gym obs / our obs | gym reward / our reward | gym done / our done")
for i, a in enumerate([0, 2, 2, 1, 1, 1, 1, 2]):
    lake_obs, lake_rew, lake_done, _ = lake.step(a)
    pond_obs, pond_rew, pond_done, _ = pond.step(a)
    print("%2d   |      %2d / %2d      |          %d / %d        |      %5s / %5s" % \
          (i, lake_obs, pond_obs, lake_rew, pond_rew, lake_done, pond_done))