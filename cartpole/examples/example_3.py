import gym
from gym import spaces


"""
이전의 예제에서 CartPole 게임 환경에서 주어지는 행동 공간 (action space)에서 "임의의" 행동을 선택했다. 모든 환경은 action_space와 observation_space를 가진다.

이러한 속성들은 공간의 한 유형이며, 유효한 행동과 관찰의 형식을 보여준다.

더 나은 행동을 취하기 위해서, env 와 action 에 대해서 살펴보자.
"""

env = gym.make('CartPole-v1', render_mode="rgb_array")

print(env.action_space) # 고정된 범위, 음이 아닌 숫자, 지금의 경우 0 or 1
print(env.observation_space) 

print(env.observation_space.high) # 해석 - -4.8  ~ 4.8 카트는 범위에서 움직일 수 있고, 막대기의 각도는 약 -0.419 ~ 0.419 의 범위에서 움직일 수 있다. 
print(env.observation_space.low) 

print('------' * 24)
space = spaces.Discrete(8)      # 8개의 요소를 갖는 세트 {0, 1, 2, ..., 7}
x = space.sample()

print(space.contains(x))
print(space.n == 8)


"action -> random action 말고 다른 거!?"

"random 하게 action 을 하긴 하지만 0 or 1 이 아니라, 0 ~ 7 사이의 값에서 sampling"