from gobang.game import GobangGame
from gobang.players import HumanGobangPlayer, RandomPlayer, AlphaZeroPlayer
from net import NeuralNet
import torch

game = GobangGame()
human = HumanGobangPlayer(game)
greedy = RandomPlayer(game)

device = "cuda" if torch.cuda.is_available() else "cpu"
nn = NeuralNet(game).to(device)
alphago = AlphaZeroPlayer(game, nn)

while True:
    board = game.getInitBoard()
    player = 1
    while True:
        if player == 1:
            action = human.play(board)
        else:
            action = alphago.play(board)
        board, player = game.getNextState(board, player, action)
        game.display(board)
        result = game.getGameEnded(board, player)
        if result != 0:
            print("Result", result)
            break
    print("Game End")





