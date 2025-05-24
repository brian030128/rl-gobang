import copy
from collections import deque
import random
import argparse
import multiprocessing as mp
import time
import os

import wandb
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from multi_thread_arena import MultiThreadedArena
from gobang.players import AlphaZeroPlayer
from gobang.game import GobangGame
from net import NeuralNet
from mcts import MCTS

device = "cuda" if torch.cuda.is_available() else "cpu"

def save_model(model, path):
    torch.save(model.state_dict(), path)



class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        board, pi, v = self.data[idx]
        b_tensor = torch.tensor(board, dtype=torch.float32, device=device)
        pi_tensor = torch.tensor(pi, dtype=torch.float32, device=device)
        v_tensor = torch.tensor(v, dtype=torch.float32, device=device)
        return b_tensor, pi_tensor, v_tensor

    def __len__(self):
        return len(self.data)

def train(model, optimizer, data, batch_size=3, train_epoches=10):
    model.train()

    total_loss = 0
    dataset = MyDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(train_epoches):
        for boards, pis, vs in dataloader:
            if boards.shape[0] == 1:
                break
            optimizer.zero_grad()

            pred_pi, pred_v = model(boards)
            loss_pi = -torch.mean(torch.sum(pis * torch.log(pred_pi + 1e-10), dim=1))
            loss_v = torch.mean((pred_v - vs) ** 2)
            loss = loss_pi + loss_v
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()

            wandb.log({
                "Loss": loss.item(),
            })

    return total_loss / len(dataloader)

def episode_worker(game: GobangGame, net, args, training_data, started_episodes,target_episodes):
    while True:
        with started_episodes.get_lock():
            if started_episodes.value >= target_episodes:
                break 
            started_episodes.value += 1
        
        board = game.getInitBoard()
        player = 1
        mcts = MCTS(game, net, args)
        episode_data = []
        while True:
            player_view = game.getCanonicalForm(board, player)
            pi = mcts.getActionProb(player_view, temp=1)
            episode_data.append((player_view, pi, player))

            action = np.random.choice(len(pi), p=pi)

            board, player = game.getNextState(board, player, action)
            result = game.getGameEnded(board, player)
            if result != 0:
                episode_data = [(x, y, result if player == 1 else - result) for x, y, player in episode_data]
                break
        
        training_data.extend(episode_data)


class Agent:

    def __init__(self, game: GobangGame, net: NeuralNet, args):
        self.args = args
        self.memory = deque()
        self.game = game
        self.net = net
        self.current_best = copy.deepcopy(net)
        self.improved_iters = []

        self.arena = MultiThreadedArena(game)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001, weight_decay=1e-4)

        self.best_model_iteration = 0
        self.train_count = 0

        self.manager = mp.Manager()


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
            result = game.getGameEnded(board, player)
            if result != 0:
                training_data = [(x, y, result if player == 1 else - result) for x, y, player in training_data]
                break
        return training_data
    
    def multi_thread_episode(self, num_episodes):
        processes = []
        training_data = self.manager.list()
        started_episodes = mp.Value('i', 0)
        for i in range(self.args.threads):
            p = mp.Process(target=episode_worker, args=(self.game, self.net, self.args, training_data, started_episodes, num_episodes))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        return training_data
            


    def learn(self):
        args = self.args
        for i in range(args.num_iterations):
            print(f"Starting iteration {i}")

            
            # Single thread for gathering data
            # iter_data = []
            # for j in tqdm(range(args.num_episodes)):
            #     training_data = self.episode()
            #     iter_data.extend(training_data)

            # Multi thread for gathering data
            start = time.time()
            iter_data = self.multi_thread_episode(args.num_episodes)
            print("Iteration took ", time.time() - start)

            self.memory.append(iter_data)
        
            while len(self.memory) > args.keep_iters:
                self.memory.popleft(0)
            
            # Update the neural network
            training_data = []
            for iter_data in self.memory:
                training_data.extend(iter_data)
            
            random.shuffle(training_data)

            loss = train(self.net, self.optimizer, training_data, args.batch_size)
            print(f"Iteration {i}, Loss: {loss}")

            
            # save the model every 10 iterations
            if i % 10 == 0:
                save_model(self.net, f"model_iter_{i}.pth")
                print(f"Model saved at model_iter_{i}.pth")

            # pk with the best model
            result = self.arena.pk(AlphaZeroPlayer(self.game, self.net, args),
                          AlphaZeroPlayer(self.game, self.current_best, args),
                          args.pk_episodes)
            print(result)
            winrate = result.count(1) / args.pk_episodes
            print(f"Winrate against current best: {winrate:.2f}")

            wandb.log({
                "Winrate": winrate,
                "Best Model Iteration": self.best_model_iteration
            })
            
            if winrate > args.pk_threshold:
                save_model(self.net, f"{args.save_dir}/best_{self.best_model_iteration}.pth")
                self.current_best.load_state_dict(self.net.state_dict())
                self.best_model_iteration += 1
                print(f"New best model found at iteration {i} with winrate {winrate:.2f}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_episodes', type=int , default=20)
    parser.add_argument('--batch_size', type=int , default=40)
    parser.add_argument('--train_epoches', type=int , default=10)
    parser.add_argument('--num_iterations', type=int, default=1000)
    parser.add_argument("--wandb-run-name", type=str, default="gobang-alpha-zero",)
    parser.add_argument('--keep_iters', type=int, default=20)
    parser.add_argument('--pk_episodes', type=int, default=40)
    parser.add_argument('--pk_threshold', type=float, default=0.6)
    parser.add_argument('--num_mcts_sims', type=int, default=25)
    parser.add_argument('--cpuct', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default="models")
    parser.add_argument('--threads', type=int, default=10)
    parser.add_argument("--seed", type=int, default=524126, help="Random seed for reproduction")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    game = GobangGame()
    net = NeuralNet(game).to(device)
    wandb.init(project="DLP-Lab5-DQN-CartPole", name=args.wandb_run_name, save_code=True)

    agent = Agent(game, net, args)
    agent.learn()



