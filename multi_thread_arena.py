from gobang.game import GobangGame
from gobang.players import AlphaZeroPlayer
from net import NeuralNet
import torch
import time
import copy

class MultiThreadedArena:
    def __init__(self, game, num_games):
        self.game = game
        self.num_games = num_games
        self.results = []
        self.manager = mp.Manager()
    
    def pk(self, player1, player2):
        results = manager.dict()
        processes = []

        for i in range(num_games):
            p1 = copy.deepcopy(player1)
            p2 = copy.deepcopy(player2)
            p = mp.Process(target=self.run, args=(p1, p2, i , results))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
        print("Wins for Player 1:", list(results.values()).count(1))
        print("Wins for Player -1:", list(results.values()).count(-1))
        print("Draws:", list(results.values()).count(1e-4))
        print("Total time taken:", time.time() - start)
    
    def run(self, p1, p2, i, results):
        result = self.play_single_game(p1, p2)
        results[i] = result

    def play_single_game(self, player1, player2):
        board = self.game.getInitBoard()
        player = 1

        while True:
            action = player1.play(board) if player == 1 else player2.play(board)
            board, player = self.game.getNextState(board, player, action)
            result = self.game.getGameEnded(board, player)
            if result != 0:
                return result
    


arena = MultiThreadedArena(GobangGame(), num_games=20)
device = "cuda" if torch.cuda.is_available() else "cpu"
game = GobangGame()

nn1 = NeuralNet(game).to(device)
nn1.eval()
nn2 = NeuralNet(game).to(device)
nn2.eval()

player1 = AlphaZeroPlayer(game, nn1)
player2 = AlphaZeroPlayer(game, nn2)

start = time.time()
arena.pk(player1, player2)
print("Total time taken:", time.time() - start)