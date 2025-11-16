import sys
import copy
import time
from config import *
from board import GameEngine, Move

class Player3:
    """
    AI Player implementing adversarial search with minimax and alpha-beta pruning.
    """
    
    def __init__(self, engine, depth=4):  # Increased depth slightly
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
        
        # Simple move ordering: captures first, then others
        ordered_moves = self._order_moves(legal_moves)
        
        # For each root move: apply it, search deeper, undo it, and pick best
        for move in ordered_moves:
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
            
            
        
        return best_move
    
    def _order_moves(self, moves):
        """Simple move ordering: captures first for better pruning."""
        captures = []
        non_captures = []
        
        for move in moves:
            if move.piece_captured != EMPTY_SQUARE:
                # Sort captures by victim value (capture high-value pieces first)
                victim_value = abs(PIECE_VALUES[move.piece_captured])
                captures.append((victim_value, move))
            else:
                non_captures.append(move)
        
        # Sort captures by victim value (descending)
        captures.sort(key=lambda x: x[0], reverse=True)
        ordered_captures = [move for _, move in captures]
        
        return ordered_captures + non_captures
    
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
        
        # Simple move ordering at each level
        ordered_moves = self._order_moves(legal_moves)
        
        # Determine whether current node is MAX (White) or MIN (Black)
        if self.engine.white_to_move:  # MAX node (White to move)
            v = -float("inf")
            for move in ordered_moves:
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
            for move in ordered_moves:
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
        Enhanced heuristic evaluation function.
        Positive score = advantage to White, Negative = advantage to Black.
        """
        game_state = self.engine.get_game_state()
        
        # Handle terminal positions
        if game_state == "checkmate":
            # Return large score favoring the side that delivered checkmate
            return 10000 if not self.engine.white_to_move else -10000
        elif game_state == "stalemate":
            return 0  # Draw
        
        score = 0
        
        # Material and positional evaluation
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                piece = self.engine.board[r][c]
                if piece == EMPTY_SQUARE:
                    continue

                piece_value = PIECE_VALUES[piece]

                # positional bonus
                pst_bonus = 0
                if piece in (WHITE_PAWN, BLACK_PAWN):
                    pst_bonus = PAWN_PST[r][c]
                elif piece in (WHITE_KNIGHT, BLACK_KNIGHT):
                    pst_bonus = KNIGHT_PST[r][c]
                elif piece in (WHITE_BISHOP, BLACK_BISHOP):
                    pst_bonus = BISHOP_PST[r][c]
                elif piece in (WHITE_KING, BLACK_KING):
                    pst_bonus = KING_PST_LATE_GAME[r][c]

                # Apply PST: add if White, subtract if Black
                if piece_value > 0:   # White
                    piece_value += pst_bonus
                else:                 # Black
                    piece_value -= pst_bonus

        score += piece_value
        
        # Add small bonuses that align with tournament scoring
        
        # Bonus for being in check (aligns with +2 points for giving check)
        if self.engine.is_in_check():
            score += 50 if not self.engine.white_to_move else -50
        
        # Simple center control bonus
        center_squares = [(3, 1), (3, 2), (4, 1), (4, 2)]
        for r, c in center_squares:
            piece = self.engine.board[r][c]
            if piece.startswith('w'):
                score += 10
            elif piece.startswith('b'):
                score -= 10
        
        # Simple king safety: penalty for king on edges in early game
        total_pieces = sum(1 for r in range(BOARD_HEIGHT) for c in range(BOARD_WIDTH) 
                          if self.engine.board[r][c] != EMPTY_SQUARE)
        
        if total_pieces > 12:  # Early/mid game
            white_king_pos = self.engine._find_king('w')
            black_king_pos = self.engine._find_king('b')
            
            if white_king_pos:
                kr, kc = white_king_pos
                if kr == 0 or kr == 7 or kc == 0 or kc == 3:  # King on edge
                    score -= 15
                    
            if black_king_pos:
                kr, kc = black_king_pos
                if kr == 0 or kr == 7 or kc == 0 or kc == 3:  # King on edge
                    score += 15
        
        return score