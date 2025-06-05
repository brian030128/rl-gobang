from gobang.board import Board 
from gobang.game import GobangGame
from torch import nn
from tqdm import tqdm
import numpy as np

import torch
import os
import torch.nn.functional as F

class NeuralNet(nn.Module):

    def __init__(self, game: GobangGame, num_channels: int = 512, dropout: float = 0.3):
        super(NeuralNet, self).__init__()
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=3, padding=1)  # same padding
        self.bn1 = nn.BatchNorm2d(num_channels)

        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=0)  # valid padding
        self.bn3 = nn.BatchNorm2d(num_channels)

        self.conv4 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=0)
        self.bn4 = nn.BatchNorm2d(num_channels)

        conv_output_size = num_channels * (self.board_x - 4) * (self.board_y - 4)

        self.fc1 = nn.Linear(conv_output_size, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)

        self.dropout = nn.Dropout(dropout)

        self.pi = nn.Linear(512, game.getActionSize())
        self.v = nn.Linear(512, 1)

    def forward(self, s):
        
        s = s.unsqueeze(1) # add channel dimension
        x = F.relu(self.bn1(self.conv1(s)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = x.view(x.size(0), -1)  # flatten

        x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn_fc2(self.fc2(x))))

        pi = F.softmax(self.pi(x), dim=1)  # policy output
        v = torch.tanh(self.v(x))          # value output

        return pi, v            # batch_size x 1


    def predict(self, board: Board):
        """
        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize
            v: a float in [-1,1] that gives the value of the current board
        """
                # game params
        board = torch.from_numpy(board).float().to(self.conv1.weight.device)
        board = board[None, :, :]
        
        pi, v = self(board)

        return pi[0].to("cpu").numpy(), v[0].to("cpu").numpy()

    def save_checkpoint(self, folder, filename):
        """
        Saves the current neural network (with its parameters) in
        folder/filename
        """
        filename = filename.split(".")[0] + ".pth"
        
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        
        torch.save(self.state_dict(), filepath)

    def load_checkpoint(self, folder, filename):
        """
        Loads parameters of the neural network from folder/filename
        """
        filename = filename.split('.')[0] + '.pth'

        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise(f'No model in path {filepath}')

        self.load_state_dict(torch.load(filepath, map_location=torch.device('cuda'))) #GPU
   
    def lossfunc(self, examples, batch_size=64, device='cuda'):
        """
        使用分批訓練處理 examples，避免 GPU OOM。
        
        examples: List of (board, pi, v)
        batch_size: 每一小批次的大小
        device: 'cuda' 或 'cpu'
        """
        self.train()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, weight_decay=1e-4)

        total_loss = 0
        num_batches = (len(examples) + batch_size - 1) // batch_size  # 向上取整

        for i in tqdm(range(0, len(examples), batch_size), desc="Training"):
            batch = examples[i:i+batch_size]

            # 拆 batch 並轉 tensor（加上 np.array 提高效能）
            boards, target_pi, target_v = zip(*batch)
            boards = torch.tensor(np.array(boards), dtype=torch.float32).to(device)
            target_pi = torch.tensor(np.array(target_pi), dtype=torch.float32).to(device)
            target_v = torch.tensor(np.array(target_v), dtype=torch.float32).to(device)

            optimizer.zero_grad()

            # 前向傳播
            out_pi, out_v = self(boards)

            # 損失計算
            p_loss = -torch.sum(target_pi * torch.log(out_pi + 1e-8)) / boards.size(0)
            v_loss = torch.mean((target_v - out_v.squeeze()) ** 2)
            loss = v_loss + p_loss

            # 反向傳播與參數更新
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 平均每 batch 的 loss
        avg_loss = total_loss / num_batches

        return avg_loss

"""     def lossfunc(self, examples, device='cuda'):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, weight_decay=1e-4)
        
        self.train()
        
        boards, target_pi, target_v = zip(*examples)
        boards = torch.tensor(boards, dtype=torch.float32).to(device)
        target_pi = torch.tensor(target_pi, dtype=torch.float32).to(device)
        target_v = torch.tensor(target_v, dtype=torch.float32).to(device)
        
        optimizer.zero_grad()

        out_pi, out_v = self(boards)

        p_loss = -torch.sum(target_pi * torch.log(out_pi + 1e-8)) / boards.size(0)
        v_loss =  torch.mean((target_v - out_v.squeeze()) ** 2)
        total_loss = v_loss + p_loss # L2 已含在 weight_decay 中
        total_loss.backward()
        
        optimizer.step()
        
        torch.save(self.state_dict(), "weight.pth")

        return total_loss.item() """



