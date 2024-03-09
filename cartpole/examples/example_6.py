import gym
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

import matplotlib.pyplot as plt

"그러면 더 action 을 잘할 수 있도록 modeling 을 하면 어떨까?"

# CartPole 환경 구성
env = gym.make('CartPole-v1')

# model 정의
class NNModel(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features=4, out_features=3, bias=True)
        self.fc2 = nn.Linear(in_features=3, out_features=2, bias=True)
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
            
model = NNModel() # model 불러오기

score = []

# 100회의 에피소드
for i in range(100):
    observation = env.reset()

    # 200개의 시간 스텝
    for t in range(200):

        # 뉴럴 네트워크의 선택
        if t == 0:
            predict = model(torch.tensor(observation[0]).reshape(1,4)) # numpy to tensor
        else:
            predict = model(torch.tensor(observation).reshape(1,4))
        predict = predict.detach().numpy() # tensor to numpy
        action = np.argmax(predict) 

        observation, reward, done, _, _ = env.step(action)

        if done:
            score.append(t + 1)
            break

env.close()
print(score)

# visulization
plt.plot(list(range(0, 100)), score)
plt.title('scores (CartPole-v1)')
plt.xlabel('Episodes')
plt.ylabel('Score')
plt.show()

"parameter() 확인하기"

"질문 - 현재 모델의 상태는?"

"과제 - 모델 수정하기"