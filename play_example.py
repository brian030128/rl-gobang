from gobang.game import GobangGame
from gobang.players import HumanGobangPlayer, RandomPlayer, AlphaZeroPlayer
from net import NeuralNet

game = GobangGame()
human = HumanGobangPlayer(game)
greedy = RandomPlayer(game)
alphago = AlphaZeroPlayer(game, NeuralNet(game))

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





