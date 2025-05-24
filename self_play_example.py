import threading
from gobang.game import GobangGame
from gobang.players import AlphaZeroPlayer
from net import NeuralNet
import torch
import time

def play_single_game(game_id, result_holder):
    game = GobangGame()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 每個執行緒有自己的模型與玩家實例
    nn1 = NeuralNet(game).to(device)
    nn1.eval()
    nn2 = NeuralNet(game).to(device)
    nn2.eval()
    
    player1 = AlphaZeroPlayer(game, nn1)
    player2 = AlphaZeroPlayer(game, nn2)

    board = game.getInitBoard()
    player = 1

    while True:
        action = player1.play(board) if player == 1 else player2.play(board)
        board, player = game.getNextState(board, player, action)
        result = game.getGameEnded(board, player)
        if result != 0:
            result_holder[game_id] = result
            break

# 執行 5 個執行緒
num_games = 3
threads = []
results = [0] * num_games

start_time = time.time()

for i in range(num_games):
    thread = threading.Thread(target=play_single_game, args=(i, results))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

end_time = time.time()

# 統計
print("All games completed.")
print(f"Time taken: {end_time - start_time:.2f} seconds")
print(f"Wins for Player 1: {results.count(1)}")
print(f"Wins for Player -1: {results.count(-1)}")
print(f"Draws: {results.count(1e-4)}")