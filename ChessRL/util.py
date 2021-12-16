import numpy as np
from matplotlib import pyplot as plt
import os

def show_board(env):
    env.render()

def plot_for_epochs(tab, title, xlabel, ylabel, folder_path=None):
    epochs = [i+1 for i in range(len(tab))]
    plt.plot(epochs, tab)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if folder_path: plt.savefig(os.path.join(folder_path, f'{title}.png'))
    plt.close()

def get_layer_board(board, mapper):
        """
        Our state is represent by an 8x8x8 array
            Plane 0 represents pawns
            Plane 1 represents rooks
            Plane 2 represents knights
            Plane 3 represents bishops
            Plane 4 represents queens
            Plane 5 represents kings
            Plane 6 represents 1/fullmove number (needed for markov property)
            Plane 7 represents can-claim-draw
        White pieces have the value 1, black pieces are minus 1
        source : https://ai.stackexchange.com/questions/7979/why-does-the-policy-network-in-alphazero-work
        """
        layer_board = np.zeros(shape=(8, 8, 8), dtype=np.float32)
        for i in range(64):
            row = i // 8
            col = i % 8
            piece = board.piece_at(i)
            if piece == None:
                continue
            elif piece.symbol().isupper():
                sign = 1
            elif piece.symbol().islower():
                sign = -1
            layer = mapper[piece.symbol().lower()]
            layer_board[layer, row, col] = sign
        if board.turn:
            layer_board[6, :, :] = 1 / board.fullmove_number
        if board.can_claim_draw():
            layer_board[7, :, :] = 1
        return layer_board