from ChessRL.environment import ChessEnv
from ChessRL.agent import DQN, DDQN
from ChessRL.util import plot_for_epochs

env = ChessEnv(opponent='stockfish')
#agent = DDQN(env=env, checkpoint_path = "ChessRL/model_saved/checkpoint_1.pt")
'''
env = ChessEnv()
agent = Agent(env, warmup=True, pgn_path='ChessRL\data\pgns\Carlsen.pgn')
agent.learn(400, time_out=100)

plot_for_epochs(agent.reward_history, 'with_warmup_reward', 'epochs', 'reward', './graphs')
'''

agent = DQN(env=env)
agent.learn(1500, time_out=100, checkpoint_folder_path="ChessRL/model_saved/dqn")
plot_for_epochs(agent.reward_history, 'DQN Reward by epochs', 'epochs', 'reward', './graphs')
plot_for_epochs(agent.turnplay_history, 'DQN Turns played by epochs', 'epochs', 'nb of turns', './graphs')

agent = DDQN(env=env)
agent.learn(1500, time_out=100, checkpoint_folder_path="ChessRL/model_saved/ddqn")
plot_for_epochs(agent.reward_history, 'DDQN Reward by epochs', 'epochs', 'reward', './graphs')
plot_for_epochs(agent.turnplay_history, 'DDQN Turns played by epochs', 'epochs', 'nb of turns', './graphs')


agent = DQN(env=env, warmup = True, pgn_path='ChessRL/data/pgns/Carlsen.pgn')
agent.learn(1500, time_out=100, checkpoint_folder_path="ChessRL/model_saved/dqn_carlsen_warmup")
plot_for_epochs(agent.reward_history, 'DQN Reward by epochs with Carlsen warmup', 'epochs', 'reward', './graphs')
plot_for_epochs(agent.turnplay_history, 'DQN Turns played by epochs with Carlsen warmup', 'epochs', 'nb of turns', './graphs')

agent = DDQN(env=env, warmup = True, pgn_path='ChessRL/data/pgns/Carlsen.pgn')
agent.learn(1500, time_out=100, checkpoint_folder_path="ChessRL/model_saved/ChessRL/model_saved/ddqn_carlsen_warmup")
plot_for_epochs(agent.reward_history, 'DDQN Reward by epochs with Carlsen warmup', 'epochs', 'reward', './graphs')
plot_for_epochs(agent.turnplay_history, 'DDQN Turns played by epochs with Carlsen warmup', 'epochs', 'nb of turns', './graphs')


agent = DQN(env=env, warmup = True, pgn_path='ChessRL/data/pgns/Berliner.pgn')
agent.learn(1500, time_out=100, checkpoint_folder_path="ChessRL/model_saved/dqn_berliner_warmup")
plot_for_epochs(agent.reward_history, 'DQN Reward by epochs with Berliner warmup', 'epochs', 'reward', './graphs')
plot_for_epochs(agent.turnplay_history, 'DQN Turns played by epochs with Berliner warmup', 'epochs', 'nb of turns', './graphs')

agent = DDQN(env=env, warmup = True, pgn_path='ChessRL/data/pgns/Berliner.pgn')
agent.learn(1500, time_out=100, checkpoint_folder_path="ChessRL/model_saved/ChessRL/model_saved/ddqn_berliner_warmup")
plot_for_epochs(agent.reward_history, 'DDQN Reward by epochs with Berliner warmup', 'epochs', 'reward', './graphs')
plot_for_epochs(agent.turnplay_history, 'DDQN Turns played by epochs with Berliner warmup', 'epochs', 'nb of turns', './graphs')