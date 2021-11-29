from ChessRL.environment import ChessEnv
from ChessRL.agent import Agent
from ChessRL.model import Trainer

env = ChessEnv()
agent = Agent(env, warmup=True, pgn_path='ChessRL\data\pgns\Carlsen.pgn')
agent.learn(10)
