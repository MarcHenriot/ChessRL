from ChessRL.environment import ChessEnv
from ChessRL.agent import Agent
from collections import deque
from tqdm import tqdm
import numpy as np


env = ChessEnv()
agent = Agent(env)
agent.learn(10)