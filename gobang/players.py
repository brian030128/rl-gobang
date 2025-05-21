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

    def play(self, board, player):
        self.game.display(board)
        valid = self.game.getValidMoves(board, player)
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
        self.mcts = MCTS(self.game, self.nnet, args)


    def play(self, board, player, temp=1):
        board = self.game.getCanonicalForm(board, player)
        pi = self.mcts.getActionProb(board, temp=temp)
        action = np.random.choice(len(pi), p=pi)
        return action


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

args = dotdict({
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