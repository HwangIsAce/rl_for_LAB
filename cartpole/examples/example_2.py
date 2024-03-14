import gym

from utils import *

"""
매번 임의의 action 을 하는 것보다 게임을 더 잘하고 싶다면, 우리의 action 이 게임 env 에 정확히 무엇을 하는지 이해하고 있어야 한다.

게임 환경 env 의 step() 함수는 우리가 필요로 하는 것을 반환하는데, 정확히는 observation, 보상, done, info 라는 네 개의 값을 반환하게 된다. 각각에 대한 설명은 아래와 같다.

1. 관찰 (observation) (object): 환경에 대한 관찰을 나타내는 객체이며, 환경에 따라 달라진다. 예를 들어, 카메라에서 얻어지는 픽셀값, 로봇 연결부의 각도 또는 속도, 그리고 보드 게임의 상태같은 것들이 될 수 있다.

2. 보상 (reward) (float): 이전의 행동을 통해 얻어지는 보상의 양. 그 크기는 환경에 따라 달라지지만 목표는 언제나 보상의 총량을 높이는 것이다.

3. done (boolean): 환경을 reset 해야할지 나타내는 진리값이다. done=True 라면 에피소드가 종료되었음을 나타낸다.

4. info (dict): 디버깅에 유용한 진단 정보이며, 때때로 학습에 있어서 유용하다. (예를 들어, 환경의 마지막 상태 변화를 위한 확률 같은 정보가 될 수 있다.) 하지만 학습에 있어서 에이전트가 공식적인 평가에 이것을 사용할 수는 없다.

5. 기존에는 step 함수가 (observation, reward, done, info)를 반환했는데 done이 truncated, terminated 두 의미를 모두 담고 있기 때문에 

그것을 구분하기 위하여 (observation, reward, terminated, truncated, info)로 4개에서 5개로 바뀌었다. 

과거의 코드에서 truncated 부분이 구현되어 있지 않다면 next_state, reward, done, _, _ 와 같이 truncated 부분을 무시하면 해결된다. 
"""

states = []

env = gym.make('CartPole-v1', render_mode="rgb_array")

for i_episode in range(20): # episode 정의
    observation = env.reset()                   # env reset

    for t in range(100):                        
        state = env.render()
        state
        
        print(observation) 
        action = env.action_space.sample()      # Take a random action
        observation, reward, terminated, truncated, info = env.step(action)
    
        if (terminated or truncated):                                # Finish the episode if done
            print('Episode finished after {} timesteps'.format(t+1))
            break
        
        states.append(state)
    
# image to gif
save_gif(states, f'./gif/example_2.gif')

env.close()


"action, environment (reward), 20 episode"

"20 번의 각 episode 마다 100 번씩 action - reward 가 반복된다."