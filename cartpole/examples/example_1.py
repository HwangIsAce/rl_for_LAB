import gym
from IPython import display
import matplotlib
import matplotlib.pyplot as plt

from utils import save_gif

"for visualization"

states = []

env = gym.make('CartPole-v1', render_mode="rgb_array") # env definition
env.reset() # env initialization

for _ in range(100): 
    state = env.render() # image render
    state
    
    action = env.action_space.sample() # Take a random action # 0 or 1 
    env.step(action) # forward to env 
    
    states.append(state)  # for visuliazation
    
# image to gif
save_gif(states, f'./gif/example_1.gif') 

env.close()


"env definition, random action"

