from ChessRL.environment import ChessEnv
from ChessRL.agent import Agent
from collections import deque
from tqdm import tqdm
import numpy as np


env = ChessEnv()
agent = Agent(env)
state = env.reset()

reward_history = []
last_reward_50 = deque(maxlen=50)

t = tqdm(range(500))
for epoch in t:
            
    done = False
    state = env.reset()

    ep_score = 0
    n_frame = 0 # juste for debug

    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        
        agent.step(state, action, reward, next_state, done)
        state = next_state
        
        ep_score += reward
        n_frame += 1

    agent.update_epsilon()

    reward_history.append(ep_score)
    last_reward_50.append(ep_score)
    
    current_avg_score = np.mean(last_reward_50) # get average of last 50 scores

    if current_avg_score >= 200: break
    
    t.set_postfix({
        'score': ep_score,
        'avg_score': current_avg_score,
        'epsilon': agent.epsilon,
        'n_frame': n_frame
    })