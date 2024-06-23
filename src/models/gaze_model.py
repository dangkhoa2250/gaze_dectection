import torch
import torch.nn as nn
import torch.optim as optim

class GazeModel(nn.Module):
    def __init__(self):
        super(GazeModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(956, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 2)  

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)  # Không cần hàm kích hoạt ở lớp cuối vì đây là bài toán regression
        return x

