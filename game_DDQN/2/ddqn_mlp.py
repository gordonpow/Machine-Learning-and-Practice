import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNMLP(nn.Module):
    """
    Input: (9, 9, 6) one-hot observation
    Output: Q-values for 7 actions
    """

    def __init__(self, action_size=7):
        super(DQNMLP, self).__init__()

        self.flatten = nn.Flatten()  # 9*9*6 = 486

        self.fc1 = nn.Linear(486, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, x):
        # x: (B, 9, 9, 6)
        x = self.flatten(x)  # (B, 486)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q
