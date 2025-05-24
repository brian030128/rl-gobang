from gobang.game import GobangGame
from gobang.players import AlphaZeroPlayer
from net import NeuralNet
import torch
import time
import copy
import multiprocessing as mp

class MultiThreadedArena:
    def __init__(self, game, num_games):
        self.game = game
        self.num_games = num_games
        self.results = []
        self.manager = mp.Manager()
    
    def pk(self, player1, player2):
        results = self.manager.dict()
        processes = []

        for i in range(self.num_games):
            p1 = copy.deepcopy(player1)
            p2 = copy.deepcopy(player2)
            p = mp.Process(target=run, args=(self.game, p1, p2, i , results))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
        print("Wins for Player 1:", list(results.values()).count(1))
        print("Wins for Player -1:", list(results.values()).count(-1))
        print("Draws:", list(results.values()).count(1e-4))
        print("Total time taken:", time.time() - start)
    
def run(game, p1, p2, i, results):
    result = play_single_game(game, p1, p2)
    results[i] = result

def play_single_game(game, player1, player2):
    board = game.getInitBoard()
    player = 1

    while True:
        action = player1.play(board) if player == 1 else player2.play(board)
        board, player = game.getNextState(board, player, action)
        result = game.getGameEnded(board, player)
        if result != 0:
            return result
    
import argparse
if __name__ == '__main__':
    arena = MultiThreadedArena(GobangGame(), num_games=20)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    game = GobangGame()

    nn1 = NeuralNet(game).to(device)
    nn1.eval()
    nn2 = NeuralNet(game).to(device)
    nn2.eval()

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int)
    parser.add_argument('--num_iterations', type=int)
    parser.add_argument('--keep_iters', type=int, default=20)
    parser.add_argument('--pk_episodes', type=int, default=40)
    parser.add_argument('--num_mcts_sims', type=int, default=25)
    parser.add_argument('--cpuct', type=int, default=1)
    args = parser.parse_args()


    player1 = AlphaZeroPlayer(game, nn1, copy.deepcopy(args))
    player2 = AlphaZeroPlayer(game, nn2, copy.deepcopy(args))

    start = time.time()
    arena.pk(player1, player2)
    print("Total time taken:", time.time() - start)