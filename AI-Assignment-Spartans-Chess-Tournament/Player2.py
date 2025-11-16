import sys
import copy
import time
from config import *
from board import GameEngine, Move

class Player2:
    """
    AI Player implementing adversarial search with minimax and alpha-beta pruning.
    """

    def __init__(self, engine, depth=3):
        # keep signature unchanged per your constraint
        self.engine = engine
        self.nodes_expanded = 0
        self.depth = depth   # cutoff depth for minimax

    def get_best_move(self):
        """
        Entry point: returns the best move for the current board state.
        This function determines the root player's objective from the current engine state
        (engine.white_to_move). Then it searches all legal moves and returns the best one.
        """
        best_move = None
        root_is_white = self.engine.white_to_move  # True => root is MAX (White); False => MIN (Black)

        # initial best value depends on whether root is MAX or MIN
        best_eval = -float("inf") if root_is_white else float("inf")
        alpha, beta = -float("inf"), float("inf")

        legal_moves = self.engine.get_legal_moves()
        if not legal_moves:
            return None

        # For each root move: apply it, search deeper, undo it, and pick best
        for move in legal_moves:
            self.engine.make_move(move)  # now engine.white_to_move flipped
            eval_score = self._minimax(self.depth - 1, alpha, beta)  # evaluate subtree
            self.engine.undo_move()

            # compare with best from root's perspective
            if root_is_white:
                if eval_score > best_eval:
                    best_eval = eval_score
                    best_move = move
                alpha = max(alpha, best_eval)
            else:
                if eval_score < best_eval:
                    best_eval = eval_score
                    best_move = move
                beta = min(beta, best_eval)

            # optional root-level pruning
            if beta <= alpha:
                break

        return best_move

    def _minimax(self, depth, alpha, beta):
        """
        Minimax driven by engine.white_to_move at each node.
        We *do not* pass an explicit is_max parameter; instead we check the engine state.
        This avoids mismatches between passed flags and the actual board state.
        """
        self.nodes_expanded += 1

        # cutoff or terminal
        if depth == 0 or self.engine.get_game_state() != "ongoing":
            return self.evaluate_board()

        legal_moves = self.engine.get_legal_moves()
        if not legal_moves:
            # no legal moves => terminal-ish (checkmate/stalemate) — evaluate directly
            return self.evaluate_board()

        # Determine whether current node is MAX (White) or MIN (Black)
        if self.engine.white_to_move:  # MAX node (White to move)
            v = -float("inf")
            for move in legal_moves:
                self.engine.make_move(move)
                val = self._minimax(depth - 1, alpha, beta)
                self.engine.undo_move()
                if val > v:
                    v = val
                if v >= beta:
                    return v   # beta cutoff
                if v > alpha:
                    alpha = v
            return v
        else:  # MIN node (Black to move)
            v = float("inf")
            for move in legal_moves:
                self.engine.make_move(move)
                val = self._minimax(depth - 1, alpha, beta)
                self.engine.undo_move()
                if val < v:
                    v = val
                if v <= alpha:
                    return v   # alpha cutoff
                if v < beta:
                    beta = v
            return v

    def evaluate_board(self):
        """
        Heuristic evaluation function using piece values and piece-square tables.
        Positive score = advantage to White, Negative = advantage to Black.
        """
        score = 0
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                piece = self.engine.board[r][c]
                if piece == EMPTY_SQUARE:
                    continue

                # base piece value (PIECE_VALUES in config uses positive for white, negative for black)
                piece_value = PIECE_VALUES.get(piece, 0)

                # positional bonuses: PSTs are defined from White's perspective
                if piece == WHITE_PAWN:
                    piece_value += PAWN_PST[r][c]
                elif piece == BLACK_PAWN:
                    piece_value -= PAWN_PST[BOARD_HEIGHT - 1 - r][c]
                elif piece == WHITE_KNIGHT:
                    piece_value += KNIGHT_PST[r][c]
                elif piece == BLACK_KNIGHT:
                    piece_value -= KNIGHT_PST[BOARD_HEIGHT - 1 - r][c]
                elif piece == WHITE_BISHOP:
                    piece_value += BISHOP_PST[r][c]
                elif piece == BLACK_BISHOP:
                    piece_value -= BISHOP_PST[BOARD_HEIGHT - 1 - r][c]
                elif piece == WHITE_KING:
                    piece_value += KING_PST_LATE_GAME[r][c]
                elif piece == BLACK_KING:
                    piece_value -= KING_PST_LATE_GAME[BOARD_HEIGHT - 1 - r][c]

                score += piece_value

        return score
