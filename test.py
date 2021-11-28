from ChessRL.environment import ChessEnv
from ChessRL.agent import Agent
from ChessRL.model import Trainer, CNN

import poutyne as pt

env = ChessEnv()
agent = Agent(env)
# agent.learn(10)

trainer = Trainer(data_type='Carlsen')
train_loader, val_loader = trainer.init_dataloader(64)


net = CNN((8, 8, 8), 4096)
model = pt.Model(net, 'adam', 'cross_entropy', batch_metrics=['accuracy'], device='cuda')

model.fit_generator(train_loader, val_loader, epochs=20)
