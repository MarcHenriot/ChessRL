from ChessRL.environment import ChessEnv
from ChessRL.model import CNN
from ChessRL.agent import Agent


env = ChessEnv()
agent = Agent(env)
state = env.reset()

model = CNN(env.observation_shape, env.action_size)

for _ in range(100):
    action = agent.get_action(state)
    next_state, reward, done, _ = env.step(action)
            
    agent.step(state, action, reward, next_state, done)
    env.opponent_step()