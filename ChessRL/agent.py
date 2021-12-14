from ChessRL.environment import ChessEnv
from ChessRL.replayBuffer import ReplayBuffer
from ChessRL.model import CNN, Trainer

from collections import deque
from tqdm import tqdm
import numpy as np
import random
import os

import torch
import torch.nn.functional as F

class Agent(object):
    def __init__(self, env: ChessEnv, warmup=False, pgn_path=None, gamma=0.99, lr=0.001, tau=1e-3, eps_min=0.01, eps_decay=0.95, training_interval=4, buffer_size=1e5, checkpoint_path=None):
        # instances of the env
        self.env = env

        # instances of the device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # epsilon-greedy exploration strategy
        self.epsilon = 1
        self.epsilon_min = eps_min
        self.epsilon_decay = eps_decay

        # keep track of the number of times self.learn() was called for training_interval
        self.training_interval = training_interval
        self.learn_step_counter = 0

        # instances of the discount factor
        self.gamma = gamma

        # instances of tau for softupdate
        self.tau = tau

        # instances of the learning_rate
        self.lr = lr

        # instances of the network for current policy and its target
        self.policy_net = CNN(env.observation_shape, env.action_size).to(self.device)
        self.target_net = CNN(env.observation_shape, env.action_size).to(self.device).eval() # No need to train target model
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            self.policy_net.load_state_dict(checkpoint['model_state_dict'])
            self.target_net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']

        if warmup:
            trainer = Trainer(pgn_path, self.policy_net, max_game=500, device=self.device)
            warmed_net = trainer.warmup()
            self.policy_net = warmed_net
            self.target_net = warmed_net.eval()
        
        # instances of the replayBuffer
        self.memory = ReplayBuffer(env.action_size, int(buffer_size))
        self.memory.to(self.device)

        # history
        self.loss_history = []
        self.reward_history = []

    def update_epsilon(self):
        '''
        Update the value of epsilon with the decay. 
        '''
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_action(self, state):
        '''
        Returns the selected action according to an epsilon-greedy policy.
        '''
        if random.random() > self.epsilon:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            
            self.policy_net.eval()
            with torch.no_grad():
                action_values = self.policy_net(state).cpu()
                action_values = F.softmax(torch.reshape(action_values, (64, 64)), dim=0).numpy()
            self.policy_net.train()

            action_space = self.env.project_legal_moves()
            action_values = np.multiply(action_values, action_space)
            move_from = np.argmax(action_values, axis=None) // 64
            move_to = np.argmax(action_values, axis=None) % 64
            moves = [x for x in self.env.board.generate_legal_moves() if x.from_square == move_from and x.to_square == move_to]
            return random.choice(moves)
        else:
            return self.env.get_random_move()

    def step(self, state, action, reward, next_state, done, batch_size=64):
        '''
        Step the agent, train when needed.
        '''
        self.memory.add(state, action, reward, next_state, done)
        
        self.learn_step_counter = (self.learn_step_counter + 1) % self.training_interval
        
        if self.learn_step_counter == 0 and len(self.memory) > batch_size:
            self.fit(batch_size)

    def fit(self, batch_size):
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        '''
        print(states.size()) torch.Size([N, 8, 8, 8])
        print(actions.size()) torch.Size([N, 2])
        print(rewards.size()) torch.Size([N, 1])
        print(next_states.size()) torch.Size([N, 8, 8, 8])
        print(dones.size()) torch.Size([N, 1])
        '''
        q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + self.gamma * q_targets_next * (1 - dones)
        
        policy_out = self.policy_net(states).reshape((batch_size, 64, 64))
        q_expected = torch.zeros((batch_size, 1))
        for batch_idx, (x, y) in enumerate(actions):
            q_expected[batch_idx] = policy_out[batch_idx, x, y]   
        
        loss = F.mse_loss(q_expected.cpu(), q_targets.cpu())
        self.loss_history.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update() 

    def soft_update(self):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def learn(self, epochs, reward_look_back=50, early_stop_val=1000, checkpoint_folder_path=None, time_out=100000):
        last_reward = deque(maxlen=reward_look_back)
        t = tqdm(range(epochs))
        
        for epoch in t:
            done = False
            state = self.env.reset()
            ep_score = 0
            turn_play = 0
            
            while not done and turn_play < time_out:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                self.step(state, action, reward, next_state, done)
                
                state = next_state
                ep_score += reward
                turn_play += 1

            self.update_epsilon()

            self.reward_history.append(ep_score)
            last_reward.append(ep_score)
            current_avg_score = np.mean(last_reward) # get average of last 50 scores

            t.set_postfix({
                'score': ep_score,
                'avg_score': current_avg_score,
                'epsilon': self.epsilon,
                'turn_play': turn_play
            })

            if checkpoint_folder_path: 
                torch.save({
                    'model_state_dict': self.policy_net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epsilon': self.epsilon
                    }, 
                    os.path.join(checkpoint_folder_path, f'checkpoint_{epoch+1}.pt')
                )
            if current_avg_score >= early_stop_val: break

        if self.env.opponent != 'random': self.env.engine.quit()