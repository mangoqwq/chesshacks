from exceptiongroup import catch
from .utils import chess_manager, GameContext
from chess import Move
import random
import time
import chess
from torch import Tensor
from chess import Board, Move
import threading
from typing import Optional, Dict

from .utils.leela_cnn import LeelaCNN
from .utils.mcts import MCTS
from .utils.model_wrapper import ModelWrapper

# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis

from .utils import chess_manager, GameContext


import torch

# init logic
model_early_path = "./src/utils/early_checkpoint_6000_20251116_120458.pth"
model_early = LeelaCNN(10, 128)
model_early.load_state_dict(
    torch.load(model_early_path, weights_only=True, map_location=torch.device("cpu"))
)

model_late_path = "./src/utils/late_checkpoint_4200_20251116_115008.pth"
model_late = LeelaCNN(10, 128)
model_late.load_state_dict(
    torch.load(model_late_path, weights_only=True, map_location=torch.device("cpu"))
)

# wrap the model
wrapped_model_early = ModelWrapper(model_early)
wrapped_model_late = ModelWrapper(model_late)

# create mcts with the model
mcts_early = MCTS(wrapped_model_early, c_puct=3.0)
mcts_late = MCTS(wrapped_model_late, c_puct=1.0)

active_mcts = mcts_early


# Shared state for pondering thread
class PonderingState:
    def __init__(self):
        self.lock = threading.Lock()
        self.current_board: Optional[Board] = None
        self.current_temperature: float = 0.2
        self.current_mcts: Optional[MCTS] = None
        self.ponder_interval: float = 0.1  # Check for updates every 100ms
        self.should_ponder: bool = False

        # Results from pondering thread
        self.best_move: Optional[Move] = None
        self.probabilities: Optional[Dict[Move, float]] = None
        self.last_update_time: float = 0.0


pondering_state = PonderingState()
pondering_thread: Optional[threading.Thread] = None
pondering_thread_running = False


def pondering_worker():
    """Background thread that continuously ponders on the current board position."""
    global pondering_state, pondering_thread_running

    print("Pondering thread started")

    current_board = None
    current_temperature = 0.2
    current_mcts = None
    search_count = 0

    while pondering_thread_running:
        # Check for updates from main thread periodically
        with pondering_state.lock:
            if (pondering_state.current_board is not None and
                pondering_state.current_board != current_board):
                # Board has changed, update our local copy
                current_board = pondering_state.current_board.copy()
                current_temperature = pondering_state.current_temperature
                current_mcts = pondering_state.current_mcts
                search_count = 0  # Reset search count on board change

        # Always ponder if we have a board and MCTS instance
        if current_board is None or current_mcts is None:
            time.sleep(pondering_state.ponder_interval)
            continue

        # Run a single MCTS search iteration
        try:
            current_mcts.search(current_board)
            search_count += 1

            # Update results periodically (every 100ms or every 10 searches, whichever comes first)
            current_time = time.time()
            if (current_time - pondering_state.last_update_time >= 0.1 or
                search_count % 10 == 0):
                with pondering_state.lock:
                    # Re-check if board is still the same
                    if (pondering_state.current_board is not None and
                        pondering_state.current_board == current_board):
                        pondering_state.best_move = current_mcts.get_move(
                            current_board, temperature=current_temperature
                        )
                        pondering_state.probabilities = current_mcts.probabilities(
                            current_board, temperature=current_temperature
                        )
                        pondering_state.last_update_time = current_time
        except Exception as e:
            print(f"Error in pondering thread: {e}")
            time.sleep(pondering_state.ponder_interval)

    print("Pondering thread stopped")


def start_pondering_thread():
    """Start the pondering thread if it's not already running."""
    global pondering_thread, pondering_thread_running

    if pondering_thread_running:
        return

    pondering_thread_running = True
    pondering_thread = threading.Thread(target=pondering_worker, daemon=True)
    pondering_thread.start()
    print("Started pondering thread")


def stop_pondering_thread():
    """Stop the pondering thread."""
    global pondering_thread_running

    pondering_thread_running = False
    if pondering_thread is not None:
        pondering_thread.join(timeout=1.0)
    print("Stopped pondering thread")


# Start the pondering thread on module load
start_pondering_thread()


def random_move(ctx: GameContext) -> Move:
    print("Making random move!")
    legal_moves = list(ctx.board.legal_moves)
    move_weights = [1.0 for _ in legal_moves]  # Uniform weights for random choice
    return random.choices(legal_moves, weights=move_weights, k=1)[0]


def get_move_from_mcts(ctx: GameContext) -> Move:
    if ctx.timeLeft < 2000:
        return random_move(ctx)

    print("Making MCTS move!")
    global active_mcts
    board_pieces = len(ctx.board.piece_map())
    if board_pieces >= 20:
        active_mcts = mcts_early
        ponder_time = int(5e8)
        temperature = 0.2
    else:
        active_mcts = mcts_late
        ponder_time = int(1e9)
        temperature = 0.4

    ponder_time = min(ponder_time, int(ctx.timeLeft // 10 * 1e6))
    print("Ponder time (s):", f'{ponder_time/1e9:.2f}')

    # Update shared state for pondering thread with current board position
    # (This will be the position after opponent's move, or initial position)
    with pondering_state.lock:
        pondering_state.current_board = ctx.board.copy()
        pondering_state.current_temperature = temperature
        pondering_state.current_mcts = active_mcts
        pondering_state.should_ponder = True
        pondering_state.best_move = None
        pondering_state.probabilities = None
        pondering_state.last_update_time = 0.0

    # Wait for pondering thread to produce results
    start_time = time.time_ns()
    stop_time = start_time + ponder_time

    move = None
    while time.time_ns() < stop_time:
        # Poll the pondering thread for results
        with pondering_state.lock:
            if pondering_state.best_move is not None:
                move = pondering_state.best_move
                probabilities = pondering_state.probabilities

        if move is not None:
            # Continue pondering but we have a move ready
            time.sleep(0.01)  # Small sleep to avoid busy waiting
        else:
            # No result yet, wait a bit
            time.sleep(0.05)

    # Get final result from pondering thread
    with pondering_state.lock:
        if pondering_state.best_move is not None:
            move = pondering_state.best_move
            probabilities = pondering_state.probabilities
        else:
            # Fallback: get move directly if pondering thread hasn't produced results
            move = active_mcts.get_move(ctx.board, temperature=temperature)
            probabilities = active_mcts.probabilities(ctx.board, temperature=temperature)

    # Log probabilities
    if probabilities:
        ctx.logProbabilities(probabilities)

    print("MCTS selected move:", move)
    print("Root node visit count:", active_mcts.get_node(ctx.board).num_visits)

    # Update pondering state with board after our move (so we can think during opponent's turn)
    board_after_move = ctx.board.copy()
    board_after_move.push(move)
    
    # Determine temperature and MCTS for the position after our move
    board_pieces_after = len(board_after_move.piece_map())
    if board_pieces_after >= 20:
        mcts_after = mcts_early
        temperature_after = 0.2
    else:
        mcts_after = mcts_late
        temperature_after = 0.4
    
    # Update pondering state to continue pondering on the position after our move
    with pondering_state.lock:
        pondering_state.current_board = board_after_move
        pondering_state.current_temperature = temperature_after
        pondering_state.current_mcts = mcts_after
        pondering_state.should_ponder = True
        pondering_state.best_move = None
        pondering_state.probabilities = None
        pondering_state.last_update_time = 0.0

    return move


@chess_manager.entrypoint
def get_move(ctx: GameContext) -> Move:
    print("Getting move")
    move = None
    try:
        move = get_move_from_mcts(ctx)
    except Exception as e:
        print("ERROR OCCURRED:", str(e))
    return move


@chess_manager.reset
def reset_func(ctx: GameContext):
    print("Resetting game state")
    global moves_played
    moves_played = 0
