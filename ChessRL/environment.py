import chess
import random
import numpy as np

class ChessEnv():
    end_game_rewards = {
        '*':        0.0,  # Game not over yet
        '1/2-1/2':  0.0,  # Draw
        '1-0':  +10.0,  # White wins
        '0-1':  -10.0,  # Black wins
    }

    mapper = {
        "p": 0,
        "r": 1,
        "n": 2,
        "b": 3,
        "q": 4,
        "k": 5
    }

    def __init__(self, opponent='random', FEN=None):
        self.action_shape = (64, 64)
        self.action_size = 64 * 64
        self.observation_shape = (8, 8, 8)
        self.opponent = opponent
        self.FEN = FEN
        self.board = chess.Board(self.FEN) if self.FEN else chess.Board()
        self.init_action_space()

    def init_action_space(self):
        """
        The action  space consist of 64x64=4096 actions:
        There are 8x8 piece from where a piece can be picked up
        And another 64 pieces from where a piece can be dropped.
        """
        self.action_space = np.zeros(shape=(64, 64), dtype=np.float32)

    def step(self, move):
        piece_balance_before = self.get_material_value()
        self.board.push(move)
        piece_balance_after = self.get_material_value()
        capture_reward = piece_balance_after - piece_balance_before
        reward = self.win_reward + capture_reward
        done = self.board.is_game_over()
        return self.layer_board, reward, done, None

    def opponent_step(self):
        if not self.board.is_game_over():
            if self.opponent == 'random':
                opponent_move = self.get_random_move()
                self.board.push(opponent_move)

    def reset(self):
        self.board = chess.Board(self.FEN) if self.FEN else chess.Board()
        self.init_action_space()
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
        self.init_action_space()
        for move in [(x.from_square, x.to_square) for x in self.board.legal_moves]:
            self.action_space[move] = 1
        return self.action_space

    def get_material_value(self):
        pawns = 1 * np.sum(self.layer_board[0, :, :])
        rooks = 5 * np.sum(self.layer_board[1, :, :])
        minor = 3 * np.sum(self.layer_board[2:4, :, :])
        queen = 9 * np.sum(self.layer_board[4, :, :])
        return pawns + rooks + minor + queen

    @property
    def win_reward(self):
        result = self.board.result()
        return ChessEnv.end_game_rewards[result]

    @property
    def legal_moves(self):
        return list(self.board.legal_moves)

    @property
    def layer_board(self):
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
