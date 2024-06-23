import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Chỉ định thiết bị là GPU 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Định nghĩa mô hình phức tạp hơn
class ComplexModel(nn.Module):
    def __init__(self):
        super(ComplexModel, self).__init__()
        self.fc1 = nn.Linear(1000, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Khởi tạo mô hình và chuyển sang GPU
model = ComplexModel().to(device)

# Tạo dữ liệu giả và chuyển sang GPU
x = torch.randn(10000, 1000).to(device)
y = torch.randn(10000, 1).to(device)
dataset = TensorDataset(x, y)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Định nghĩa loss function và optimizer
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Vòng lặp huấn luyện
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Chuyển dữ liệu đầu vào và nhãn sang GPU
        data, target = data.to(device), target.to(device)

        # Reset gradient
        optimizer.zero_grad()
        # Forward pass
        output = model(data)
        # Tính loss
        loss = criterion(output, target)
        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

    # In GPU memory usage
    print(f'GPU memory usage: {torch.cuda.memory_allocated(device) / (1024 ** 2)} MB')

print("Training complete.")
