import gym
from IPython import display
import matplotlib
import matplotlib.pyplot as plt

from utils import save_gif

"for visualization"

screens = []

env = gym.make('CartPole-v1', render_mode="rgb_array") # env 정의 
env.reset() # env 초기화
img = plt.imshow(env.render()) # 이미지 render 
for _ in range(100): 
    
    screens.append(env.render()) 
    # display.clear_output(wait=True) 
    # display.display(plt.gcf()) 

    action = env.action_space.sample() # Take a random action
    env.step(action) # env update 
env.close()

# image to gif
save_gif(screens, f'./gif/example_1.gif') 


