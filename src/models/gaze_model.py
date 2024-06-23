import torch
import torch.nn as nn
import torch.optim as optim

class GazeModel(nn.Module):
    def __init__(self):
        super(GazeModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(956, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 2)
        self.dropout = nn.Dropout(p=0.5)  # Dropout với tỷ lệ 0.5

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        x = self.fc4(x)  # Không cần hàm kích hoạt ở lớp cuối vì đây là bài toán regression
        return x