import torch
import random
from collections import deque, namedtuple
import numpy as np

class ReplayBuffer():
    def __init__(self, action_size, buffer_size):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def to(self, device):
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        experience = self.experience(state, [action.from_square, action.to_square], reward, next_state, done)
        self.memory.append(experience)
    
    def sample(self, batch_size):
        '''
        Randomly sample a batch of experiences from memory.
        '''
        experiences = random.sample(self.memory, k=batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)