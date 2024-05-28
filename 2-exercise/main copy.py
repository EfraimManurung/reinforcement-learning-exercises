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

# Set the console encoding to UTF-8
if os.name == 'nt':
    os.system('chcp 65001')
    sys.stdout.reconfigure(encoding='utf-8')

from FrozenPond import FrozenPond

env = FrozenPond()
obs = env.reset()

done = False
for i in range(55):
    obs, rewards, done, _ = env.step(0)
    env.render()
    print(i+1, done)