from gobang.board import Board, GobangGame

import torch
from torch import nn
import torch.nn.functional as F

class NeuralNet():

    def __init__(self, game: GobangGame, num_channels: int = 512, dropout: float = 0.3):
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
        # s: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)  # add channel dim: batch_size x 1 x board_x x board_y

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
        board = board[torch.newaxis, :, :]
        
        pi, v = self.nnet.model.predict(board, verbose=False)

        return pi[0], v[0]