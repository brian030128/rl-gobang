import numpy as np
from mcts import MCTS

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanGobangPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        self.game.display(board)
        valid = self.game.getValidMoves(board, 1)
        #for i in range(len(valid)):
        #    if valid[i]:
        #        print(int(i/self.game.n), int(i%self.game.n))
        while True:
            a = input()

            x,y = [int(x) for x in a.split(' ')]
            a = self.game.n * x + y if x!= -1 else self.game.n ** 2
            if valid[a]:
                break
            else:
                print('Invalid')

        return a
    

class AlphaZeroPlayer():
    def __init__(self, game, nnet):
        self.game = game
        self.nnet = nnet
        self.mcts = MCTS(self.game, self.nnet)

    def play(self, board, temp=1):
        pi = self.mcts.getActionProb(board, temp=temp)
        action = np.random.choice(len(pi), p=pi)
        return action