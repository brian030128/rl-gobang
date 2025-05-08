from typing import List

class MCTS():
    def __init__(self, game, nnet):
        self.game = game
        self.nnet = nnet

    def getActionProb(self, canonicalBoard, temp=1) -> List[float]:
        pass

    