import copy
from collections import deque
import random
import argparse


import wandb
import torch
import numpy as np
from tqdm import tqdm


from gobang.game import GobangGame
from net import NeuralNet
from mcts import MCTS

device = "cuda" if torch.cuda.is_available() else "cpu"

def save_model(model, path):
    torch.save(model.state_dict(), path)


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()

    total_loss = 0
    for board, pi, v in data:
        board = torch.from_numpy(board).float().to(device)
        pi = torch.from_numpy(pi).float().to(device)
        v = torch.tensor(v, dtype=torch.float32).to(device)

        pred_pi, pred_v = model(board)

        loss_pi = -torch.sum(pi * torch.log(pred_pi + 1e-10))
        loss_v = torch.mean((pred_v - v) ** 2)

        loss = loss_pi + loss_v
        total_loss += loss.item()

        loss.backward()

    optimizer.step()
    return total_loss / len(data)

class Agent:

    def __init__(self, game: GobangGame, net: NeuralNet, args):
        self.memory = deque()
        self.game = game
        self.net = net
        self.current_best = copy.deepcopy(net)
        self.improved_iters = []

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001, weight_decay=1e-4)


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
        for i in range(args.num_iterations):
            print(f"Starting iteration {i}")

            iter_memory = []
            for j in tqdm(range(args.num_episodes)):
                training_data = self.episode()
                iter_memory.append(training_data)

            self.memory.append(iter_memory)
        
            if len(self.memory) > args.keep_iters:
                self.memory.popleft(0)
            
            # Update the neural network
            training_data = []
            for episode in self.memory:
                training_data.extend(episode)
            
            random.shuffle(training_data)

            loss = train(self.net, self.optimizer, training_data)

            print(f"Iteration {i}, Loss: {loss}")
            wandb.log({
                "Train Loss": loss.item(),
                "Train Step": self.train_count
            })
            
            # save the model every 10 iterations
            if i % 10 == 0:
                save_model(self.net, f"model_iter_{i}.pth")
                print(f"Model saved at model_iter_{i}.pth")

            # pk with the best model
            wins = 0
            total_games = 0
            for i in range(args.pk_episodes):
                player = 1

        
    


            

                



if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument('--num_episodes', type=int)
    parser.add_argument('--num_iterations', type=int)
    parser.add_argument('--keep_iters', type=int, default=20)
    parser.add_argument('--pk_episodes', type=int, default=40)


    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    game = GobangGame()
    net = NeuralNet(game).to(device)
    wandb.init(project="DLP-Lab5-DQN-CartPole", name=args.wandb_run_name, save_code=True)


    agent = Agent(game, net, args)
    agent.learn()



