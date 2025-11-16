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
model_path = "./src/utils/checkpoint_4000_20251116_060127.pth"
model = LeelaCNN(10, 128)
model.load_state_dict(
    torch.load(model_path, weights_only=True, map_location=torch.device("cpu"))
)

# wrap the model
wrapped_model = ModelWrapper(model)

# create mcts with the model
mcts = MCTS(wrapped_model, c_puct=3.0)
moves_played = 0


def random_move(ctx: GameContext) -> Move:
    print("Making random move!")
    legal_moves = list(ctx.board.legal_moves)
    move_weights = [1.0 for _ in legal_moves]  # Uniform weights for random choice
    return random.choices(legal_moves, weights=move_weights, k=1)[0]


def get_move_from_mcts(ctx: GameContext) -> Move:
    if ctx.timeLeft < 2000:
        return random_move(ctx)

    print("Making MCTS move!")
    global moves_played
    if moves_played < 10:
        ponder_time = int(3e8)
        temperature = 0.2
    elif moves_played < 30:
        ponder_time = int(5e8)
        temperature = 0.3
    else:
        ponder_time = int(1e9)
        temperature = 0.6

    ponder_time = min(ponder_time, int(ctx.timeLeft // 10 * 1e6))
    print("Ponder time (ns):", ponder_time)

    move = mcts.ponder_time(
        board=ctx.board, ponder_time_ns=ponder_time, temperature=temperature
    )
    print("MCTS selected move:", move)

    moves_played += 1
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
