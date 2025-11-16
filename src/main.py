from exceptiongroup import catch
from .utils import chess_manager, GameContext
from chess import Move
import random
import time
import chess
from torch import Tensor
from chess import Board, Move

from .utils.leela_cnn import LeelaCNN
from .utils.mcts import MCTS
from .utils.model_wrapper import ModelWrapper

# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis

from .utils import chess_manager, GameContext


import torch

# init logic
model_early_path = "./src/utils/checkpoint_600_20251116_090407.pth"
model_early = LeelaCNN(10, 128)
model_early.load_state_dict(
    torch.load(model_early_path, weights_only=True, map_location=torch.device("cpu"))
)

model_late_path = "./src/utils/checkpoint_600_20251116_090407.pth"
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
        temperature = 0.6

    ponder_time = min(ponder_time, int(ctx.timeLeft // 10 * 1e6))
    print("Ponder time (ns):", ponder_time)

    move = active_mcts.ponder_time(
        board=ctx.board, ponder_time_ns=ponder_time, temperature=temperature
    )
    print("MCTS selected move:", move)

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
