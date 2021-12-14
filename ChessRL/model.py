from ChessRL.environment import ChessEnv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

import chess.pgn
import numpy as np
import math

from tqdm import tqdm
from poutyne import Model


def conv2d_out_size(in_size, out_c, kernel_size=1, stride=1):
    in_size = torch.tensor(in_size)
    out_size = torch.div(in_size - (kernel_size - 1) - 1, stride, rounding_mode='trunc') + 1
    out_size[0] = out_c
    return out_size


class CNN(nn.Module):
    def __init__(self, observation_shape, action_size):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(observation_shape[0], 16, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(64)
            
        self.fc = nn.Linear(64 * 8 * 8, action_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.flatten(1)
        return self.fc(x)


class Trainer():
    def __init__(self, path, network, max_game=500, device='cpu'):
        self.pgn_games = self.load_pgn(path, max_game)
        self.dataset = TrainderDataset(*self.creat_moves_arrays())
        self.network = network
        self.device = device
    
    def load_pgn(self, path, max_game):
        acc = 0
        with open(path) as pgn: 
            games = []
            done = False
            while not done: 
                game = chess.pgn.read_game(pgn)
                if game and acc < max_game:
                    games.append(game)
                    acc += 1
                else:
                    done = True
            print(f'{len(games)} games loaded.')
        return games

    def get_moves(self, game):
        moves = []
        for move in game.mainline_moves():
            moves.append(move)
        return moves

    def creat_moves_arrays(self):
        layer_board_array = []
        project_moves_array = []
        for pgn_game in tqdm(self.pgn_games):
            board = pgn_game.board()
            moves = self.get_moves(pgn_game)
            layer_board_array.append(self.get_layer_board(board))
            
            for idx, move in enumerate(moves):
                board.push(move)
                project_moves_array.append(self.project_moves(move))
                if idx < len(moves) - 1:
                    layer_board_array.append(self.get_layer_board(board))

        return np.array(layer_board_array), np.array(project_moves_array)

    def get_layer_board(self, board):
        layer_board = np.zeros(shape=(8, 8, 8), dtype=np.float32)
        for i in range(64):
            row = i // 8
            col = i % 8
            piece = board.piece_at(i)
            if piece == None:
                continue
            elif piece.symbol().isupper():
                sign = 1
            else:
                sign = -1
            layer = ChessEnv.mapper[piece.symbol().lower()]
            layer_board[layer, row, col] = sign
        if board.turn:
            layer_board[6, :, :] = 1 / board.fullmove_number
        if board.can_claim_draw():
            layer_board[7, :, :] = 1
        return layer_board

    def project_moves(self, move):
        action_space = np.zeros((64, 64))
        idxs = (move.from_square, move.to_square)
        action_space[idxs] = 1
        return action_space

    def init_dataloader(self, batch_size, validation):
        if validation:
            self.train_loader, self.test_loader = self.train_valid_loaders(self.dataset, batch_size)
        else:
            self.train_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
            self.test_loader = None

    def train_valid_loaders(self, dataset, batch_size, train_split=0.8, shuffle=True, seed=None):
        num_data = len(dataset)
        indices = np.arange(num_data)

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)

        split = math.floor(train_split * num_data)
        train_idx, valid_idx = indices[:split], indices[split:]

        train_dataset = Subset(dataset, indices=train_idx)
        valid_dataset = Subset(dataset, indices=valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

        return train_loader, valid_loader

    def warmup(self, batch_size=128, epochs=10, validation=False):
        self.init_dataloader(batch_size,validation)
        model = Model(self.network, 'adam', 'cross_entropy', batch_metrics=['accuracy'], device=self.device)
        if validation:
            model.fit_generator(self.train_loader, self.test_loader, epochs=epochs)
        else:
            model.fit_generator(self.train_loader, epochs=epochs)
        return self.network


class TrainderDataset(Dataset):
    def __init__(self, layer_board_array, project_moves_array):
        super().__init__()
        self.X = layer_board_array
        self.y = project_moves_array
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        board = self.X[index]
        move = self.y[index]
        return torch.from_numpy(board), torch.from_numpy(move).reshape((4096)).argmax()