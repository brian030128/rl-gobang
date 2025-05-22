import copy


import torch
import numpy as np
from tqdm import tqdm


from gobang.game import GobangGame
from net import NeuralNet
from mcts import MCTS

device = "cuda" if torch.cuda.is_available() else "cpu"

def save_model(model, path):
    torch.save(model.state_dict(), path)


class Agent:

    def __init__(self, game: GobangGame, net: NeuralNet, args):
        self.memory = []
        self.game = game
        self.net = net
        self.current_best = copy.deepcopy(net)


    def episode(self):
        board = self.game.getInitBoard()
        player = 1
        mcts = MCTS(self.game, self.net, self.args)

        training_data = []
        while True:
            player_view_board = self.game.getCanonicalForm(board, player)
            pi = mcts.getActionProb(player_view_board, temp=1)
            training_data.append((player_view_board, pi, player))

            action = np.random.choice(len(pi), p=pi)

            board, player = game.getNextState(board, player, action)
            game.display(board)
            result = game.getGameEnded(board, player)
            if result != 0:
                training_data = [(x, y, result if player == 1 else - result) for x, y, player in training_data]
                break
        return training_data

    def learn(self):
        for i in range(1000):
            print(f"Starting iteration {i}")
            for j in tqdm(range(100)):
                training_data = self.episode()
                self.memory.extend(training_data)
                


game = GobangGame()
net = NeuralNet(game).to(device)
agent = Agent(game, net, None)

