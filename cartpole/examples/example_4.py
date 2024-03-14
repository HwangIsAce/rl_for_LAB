import gym
from utils import *

"""
다시 본론으로 돌아가서, 더 action 을 잘 할 수 있을까? 

그러면 action = 0 or 1 을 취했을 때 어떤 일이 일어날 지부터 생각해보자. 
"""

# 행동 0
env = gym.make('CartPole-v1', render_mode="rgb_array")
observation = env.reset()
print(observation)

observation, reward, terminated, truncated, info = env.step(0)
print(observation)

env.close()

" ***** [카트의 위치, 카트의 속도, 막대기의 각도, 막대기의 회전율] ****** "

"행동 0 -> 카트의 속도가 왼쪽 방향으로 증가하고, 막대기의 회전율이 오른쪽으로 기우는 방향으로 변한다." 

# 행동 0 을 반복해서 해보자.
import math

states_1 = []

env = gym.make('CartPole-v1', render_mode="rgb_array")
env.reset()

for i in range(10):
    state1 = env.render()
    state1
    
    observation, reward, done, _, _ = env.step(0)
    print(observation, done)
    
    if done:
        print(f'radian: {observation[2]}, degree: {math.degrees(observation[2])}') # radian -> 각도 표기법 # done (terminated, truncated) 이 True 가 될 때, 막대기의 각도가 12도보다 커지기 때문에 종료 조건 만족
        break
    
    states_1.append(state1)
    
save_gif(states_1, f'./gif/example_4_(1).gif')

env.close()

print('------' * 24)


# 행동 1을 반복해서 해보자.

states_2 = []

env = gym.make('CartPole-v1')
env.reset()

for i in range(10):
    state2 = env.render()
    
    observation, reward, done, _, _ = env.step(1)
    print(observation, done)
    
    if done:
        print(f'radian: {observation[2]}, degree: {math.degrees(observation[2])}') # radian -> 각도 표기법 # done (terminated, truncated) 이 True 가 될 때, 막대기의 각도가 12도보다 커지기 때문에 종료 조건 만족
        break
    
    states_2.append(states_2)
    
save_gif(states_2, f'./gif/example_4_(2).gif')
    
env.close()