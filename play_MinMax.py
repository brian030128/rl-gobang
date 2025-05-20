from gobang.game import GobangGame
from gobang.MinMax import MinMaxPlayer
from gobang.players import HumanGobangPlayer
import numpy as np

game = GobangGame()
human = HumanGobangPlayer(game)
minmax = MinMaxPlayer(game, search_depth=2)


while True:
    board = game.getInitBoard()
    player = 1  # 1 for MinMax, -1 for human
    game.display(board)

    while True:
        if player == 1:
            print("MinMax turn")
            action = minmax.play(board, player)
            y, x = divmod(action, game.n)
            print(f"MinMax plays: {y} {x}")
        else:
            action = human.play(board)

        board, player = game.getNextState(board, player, action)
        game.display(board)

        result = game.getGameEnded(board, 1)  # Use 1 as perspective
        if result != 0:
            if result == 1:
                print("MinMax Player wins!")
            elif result == -1:
                print("Human wins!")
            else:
                print("It's a draw!")
            break

    again = input("Do you want to play again? (y/n): ").strip().lower()
    if again != 'y':
        print("Thanks for playing!")
        break
