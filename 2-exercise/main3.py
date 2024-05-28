from FrozenPond import FrozenPond
from gym.wrappers import TimeLimit

pond = FrozenPond()
pond_5 = TimeLimit(pond, max_episode_steps=5)

pond.reset()
pond_5.reset()

for i in range(5):
    print(pond_5.step(0))