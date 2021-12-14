from ChessRL.util import get_layer_board

import chess
import chess.engine
import random
import numpy as np

class ChessEnv():
    mapper = {
        'p': 0,
        'r': 1,
        'n': 2,
        'b': 3,
        'q': 4,
        'k': 5
    }

    '''
    B > N > 3P
    B + N = R + 1.5P
    Q + P = 2R
    source : https://www.chessprogramming.org/Simplified_Evaluation_Function
    '''
    piece_value = {
        'p': 10,
        'n': 32,
        'b': 33,
        'r': 50,
        'q': 90,
        'k': 2000
    }

    end_game_rewards = {
        '*':        0.0,  # Game not over yet
        '1/2-1/2':  0.0,  # Draw
        '1-0':  piece_value['k'],  # White wins
        '0-1':  -piece_value['k'],  # Black wins
    }

    '''
    src : https://www.chessprogramming.org/Simplified_Evaluation_Function
    '''
    piece_weight = {
        'p': np.array([
            [0,  0,  0,  0,  0,  0,  0,  0],
            [50, 50, 50, 50, 50, 50, 50, 50],
            [10, 10, 20, 30, 30, 20, 10, 10],
            [5,  5, 10, 25, 25, 10,  5,  5],
            [0,  0,  0, 20, 20,  0,  0,  0],
            [5, -5,-10,  0,  0,-10, -5,  5],
            [5, 10, 10,-20,-20, 10, 10,  5],
            [0,  0,  0,  0,  0,  0,  0,  0]
        ]),
        'n': np.array([
            [-50,-40,-30,-30,-30,-30,-40,-50],
            [-40,-20,  0,  0,  0,  0,-20,-40],
            [-30,  0, 10, 15, 15, 10,  0,-30],
            [-30,  5, 15, 20, 20, 15,  5,-30],
            [-30,  0, 15, 20, 20, 15,  0,-30],
            [-30,  5, 10, 15, 15, 10,  5,-30],
            [-40,-20,  0,  5,  5,  0,-20,-40],
            [-50,-40,-30,-30,-30,-30,-40,-50],
        ]),
        'b': np.array([
            [-20,-10,-10,-10,-10,-10,-10,-20],
            [-10,  0,  0,  0,  0,  0,  0,-10],
            [-10,  0,  5, 10, 10,  5,  0,-10],
            [-10,  5,  5, 10, 10,  5,  5,-10],
            [-10,  0, 10, 10, 10, 10,  0,-10],
            [-10, 10, 10, 10, 10, 10, 10,-10],
            [-10,  5,  0,  0,  0,  0,  5,-10],
            [-20,-10,-10,-10,-10,-10,-10,-20],
        ]),
        'r': np.array([
            [0,  0,  0,  0,  0,  0,  0,  0],
            [5, 10, 10, 10, 10, 10, 10,  5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [0,  0,  0,  5,  5,  0,  0,  0]
        ]),
        'q': np.array([
            [-20,-10,-10, -5, -5,-10,-10,-20],
            [-10,  0,  0,  0,  0,  0,  0,-10],
            [-10,  0,  5,  5,  5,  5,  0,-10],
            [-5,  0,  5,  5,  5,  5,  0, -5],
            [0,  0,  5,  5,  5,  5,  0, -5],
            [-10,  5,  5,  5,  5,  5,  0,-10],
            [-10,  0,  5,  0,  0,  0,  0,-10],
            [-20,-10,-10, -5, -5,-10,-10,-20]
        ]),
        'k': np.array([
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-20,-30,-30,-40,-40,-30,-30,-20],
            [-10,-20,-20,-20,-20,-20,-20,-10],
            [20, 20,  0,  0,  0,  0, 20, 20],
            [20, 30, 10,  0,  0, 10, 30, 20]
        ])
    }

    def __init__(self, opponent='random', FEN=None, limit_time=0.1):
        self.action_shape = (64, 64)
        self.action_size = 64 * 64
        self.observation_shape = (8, 8, 8)
        self.opponent = opponent
        self.FEN = FEN
        self.board = chess.Board(self.FEN) if self.FEN else chess.Board()
        self.reset_action_space()
        if opponent == 'stockfish':
            self.limit_time = chess.engine.Limit(time=limit_time)
            self.engine = chess.engine.SimpleEngine.popen_uci('ChessRL/engine/stockfish_14.1_win_x64_avx2.exe')
            print('Stockfish loaded !')

    def reset_action_space(self):
        '''
        The action  space consist of 64x64=4096 actions:
        There are 8x8 piece from where a piece can be picked up
        And another 64 pieces from where a piece can be dropped.
        '''
        self.action_space = np.zeros(shape=(64, 64), dtype=np.float32)

    def step(self, move):
        reward = (self.win_reward() + self.get_capture_reward(move) + self.get_placement_reward(move)) / ChessEnv.piece_value['k']
        self.board.push(move)
        self.opponent_step()
        done = self.board.is_game_over()
        return self.layer_board, reward, done, None

    def opponent_step(self):
        if not self.board.is_game_over():
            if self.opponent == 'random':
                opponent_move = self.get_random_move()
                self.board.push(opponent_move)

            elif self.opponent == 'stockfish':
                opponent_move = self.engine.play(self.board, self.limit_time).move
                self.board.push(opponent_move)

    def reset(self):
        self.board = chess.Board(self.FEN) if self.FEN else chess.Board()
        self.reset_action_space()
        return self.layer_board

    def render(self, mode='unicode', turn_number=True):
        if turn_number:
            turn_number = self.board.fen().split(' ')[-1]
            print('Turn number: ', turn_number)

        if mode == 'unicode':
            print(self.board.unicode())

        if mode == 'ascii':
            print(self.board)

        if mode == 'fancy':
            return self.board

    def get_random_move(self):
        return random.choice(self.legal_moves)

    def project_legal_moves(self):
        self.reset_action_space()
        for move in [(x.from_square, x.to_square) for x in self.board.legal_moves]:
            self.action_space[move] = 1
        return self.action_space

    def get_material_value(self, board):
        layer_board = get_layer_board(board, ChessEnv.mapper)
        pawns = ChessEnv.piece_value['p'] * np.sum(layer_board[0, :, :])
        rooks = ChessEnv.piece_value['r'] * np.sum(layer_board[1, :, :])
        knights = ChessEnv.piece_value['n'] * np.sum(layer_board[2, :, :])
        bishops = ChessEnv.piece_value['b'] * np.sum(layer_board[3, :, :])
        queen = ChessEnv.piece_value['q'] * np.sum(layer_board[4, :, :])
        return pawns + rooks + knights + bishops + queen

    def get_capture_reward(self, move):
        board_copy = self.board.copy()
        piece_balance_before = self.get_material_value(board_copy)
        board_copy.push(move)
        piece_balance_after = self.get_material_value(board_copy)
        return piece_balance_after - piece_balance_before

    def win_reward(self):
        result = self.board.result()
        return ChessEnv.end_game_rewards[result]

    def get_placement_reward(self, move):
        piece = self.board.piece_at(move.from_square).symbol().lower()
        end_square_index = move.to_square
        row = end_square_index // 8
        col = end_square_index % 8
        return ChessEnv.piece_weight[piece][row][col] * ChessEnv.piece_value[piece]

    @property
    def legal_moves(self):
        return list(self.board.legal_moves)

    @property
    def layer_board(self):
        '''
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
        '''
        layer_board = np.zeros(shape=(8, 8, 8), dtype=np.float32)
        for i in range(64):
            row = i // 8
            col = i % 8
            piece = self.board.piece_at(i)
            if piece == None:
                continue
            elif piece.symbol().isupper():
                sign = 1
            else:
                sign = -1
            layer = ChessEnv.mapper[piece.symbol().lower()]
            layer_board[layer, row, col] = sign
        if self.board.turn:
            layer_board[6, :, :] = 1 / self.board.fullmove_number
        if self.board.can_claim_draw():
            layer_board[7, :, :] = 1
        return layer_board



