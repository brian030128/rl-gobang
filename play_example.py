from gobang.game import GobangGame
from gobang.players import HumanGobangPlayer, RandomPlayer, AlphaZeroPlayer
from net import NeuralNet
from train import Agent
from dotted_dict import DottedDict

import torch


args = DottedDict({
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

game = GobangGame()
human = HumanGobangPlayer(game)
greedy = RandomPlayer(game)

device = "cuda" if torch.cuda.is_available() else "cpu"
nn = NeuralNet(game).to(device)
alphago = AlphaZeroPlayer(game, nn)

#創一個 Agent object.learn()
#傳上面參數進Agent
c = Agent(game, nn, args)
c.learn()


# while True:
#     board = game.getInitBoard()
#     player = 1
#     while True:
#         if player == 1:
#             action = human.play(board)
#         else:
#             action = alphago.play(board)
#         board, player = game.getNextState(board, player, action)
#         game.display(board)
#         result = game.getGameEnded(board, player)
#         if result != 0:
#             print("Result", result)
#             break
#     print("Game End")





