import gym

from utils import *

"그렇다면 다른 알고리즘을 적용해보자."


states = []

env = gym.make('CartPole-v1', render_mode="rgb_array")
observation = env.reset()

for i in range(100):
    state = env.render()
    state
    # 알고리즘1:
    # 막대기가 오른쪽으로 기울어져 있다면, 오른쪽으로 힘을 가하고
    # 그렇지 않다면, 왼쪽으로 힘을 가하기.
    
    if i == 0:
        if observation[0][2] > 0:
            action = 1
        else: action = 0
    else:
        if observation[2] > 0:
            action = 1
        else: action = 0

    observation, reward, done, _, _ = env.step(action)
    print(observation, done)
    
    if done:
        print(i+1)
        break
    
    states.append(state)
    
save_gif(states, f'./gif/example_5.gif')
    
env.close()

"어쨌든 처음보다는 나아졌다."