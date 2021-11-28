from ChessRL.environment import ChessEnv
from ChessRL.agent import Agent
from ChessRL.model import Trainer


env = ChessEnv()
agent = Agent(env)
# agent.learn(10)

trainer = Trainer('ChessRL\data\pgns\Berliner.pgn')
print(len(trainer.dataset))