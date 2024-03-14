import gym 
import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from example_6 import NNModel

model = NNModel()

# config
LR = 0.001
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR, amsgrad=True)


score = []
memory = deque(maxlen=2000)

# CartPole 환경 구성
env = gym.make('CartPole-v1', render_mode="rgb_array")

# 1000회의 에피소드 시작
for i in range(1000):
    
    state = env.reset()
    
    if type(state) == tuple:
        state = state[0]

    state = np.reshape(state, [1, 4])
    eps = 1 / (i / 50 + 10) # epsilon 

    # 200 timesteps
    for t in range(200):

        # Inference: e-greedy
        if np.random.rand() < eps:
            action = np.random.randint(0, 2)
        else:
            with torch.no_grad():
                predict = model(torch.tensor(state))
                predict = predict.detach().numpy()
                action = np.argmax(predict)
                        
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 4])

        memory.append((state, action, reward, next_state, done))
        state = next_state

        if done or t == 199:
            print('Episode', i, 'Score', t + 1)
            score.append(t + 1)
            break
        
    # Training

    if i > 10:
        
        minibatch = random.sample(memory, 16)

        for state, action, reward, next_state, done in minibatch:
            model.train()
            
            target = reward
            if not done:
                target = reward + 0.9 * np.amax((model(torch.tensor(next_state))[0]).detach().numpy())
            target_outputs = model(torch.tensor(state))
            target_outputs = target_outputs.detach().numpy()
            target_outputs[0][action] = target
            
            outputs = model(torch.tensor(state))
            
            loss = criterion(outputs, torch.tensor(target_outputs))
            optimizer.zero_grad()
            loss.backward()
            
            print(loss)
            
            # model.fit(state, target_outputs, epochs=1, verbose=0)

env.close()
print(score)


"어떻게 해야 문제를 해결할 수 있을지 discussion 하기"

