from ChessRL.environment import ChessEnv
from ChessRL.agent import DQN, DDQN
from ChessRL.util import plot_for_epochs

env = ChessEnv(opponent='stockfish')
agent = DDQN(env=env)
agent.learn(300, time_out=100)

#plot_for_epochs(agent.reward_history, 'no_warmup_reward', 'epochs', 'reward', './graphs')

'''
env = ChessEnv()
agent = Agent(env, warmup=True, pgn_path='ChessRL\data\pgns\Carlsen.pgn')
agent.learn(400, time_out=100)

plot_for_epochs(agent.reward_history, 'with_warmup_reward', 'epochs', 'reward', './graphs')
'''