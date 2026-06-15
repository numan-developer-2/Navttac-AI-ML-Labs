# Lab 72: PyTorch fundamentals (NAVTTC syllabus requirement)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

print("-" * 42)
print("Lab 72: PyTorch Fundamentals")
print("-" * 42)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Tensor operations
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
print(f"\nTensor add: {(a + b).tolist()}")
print(f"Tensor mean: {a.mean().item():.2f}")

# Autograd demo
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1
y.backward()
print(f"dy/dx at x=3: {x.grad.item()}")  # 2x + 2 = 8

# Simple classifier on XOR-like data
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=torch.float32)
y = torch.tensor([[0.], [1.], [1.], [0.]], dtype=torch.float32)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=4, shuffle=True)


class XorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


model = XorNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

for epoch in range(300):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    preds = (model(X) > 0.5).float()
    acc = (preds == y).float().mean().item()

print(f"\nXOR classifier accuracy: {acc * 100:.0f}%")
print("Predictions:", preds.flatten().tolist())
print("Targets    :", y.flatten().tolist())

# Save and load model (persistence)
from pathlib import Path

Path("saved_models").mkdir(exist_ok=True)
torch.save(model.state_dict(), "saved_models/lab72_pytorch_xor.pt")
print(f"\nModel saved: saved_models/lab72_pytorch_xor.pt")

loaded = XorNet()
loaded.load_state_dict(torch.load("saved_models/lab72_pytorch_xor.pt", weights_only=True))
loaded.eval()
print("Reloaded model works:", loaded(X).shape)
