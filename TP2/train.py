import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import CardioDataset

torch.manual_seed(42)

dataset = CardioDataset("data/cardio_train.csv")
generator = torch.Generator().manual_seed(42)
train_set, val_set, test_set = random_split(dataset, [0.8, 0.1, 0.1], generator=generator)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_set, batch_size=64, shuffle=False)

batch = next(iter(train_loader))


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid() # Sortie binaire [0, 1]
        )

    def forward(self, x):
        return self.net(x)


# Initialisation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(input_size=batch['features'].shape[1], hidden_size=128).to(device)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

l1_lambda = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-4
l2_lambda = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
print(f"l1_lambda = {l1_lambda}, l2_lambda = {l2_lambda}")

for epoch in range(10):
    model.train()
    total_loss, total_base_loss, correct, n = 0.0, 0.0, 0, 0
    for batch in train_loader:
        inputs, targets = batch["features"].to(device), batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        base_loss = criterion(outputs, targets)

        # Calcul de la pénalité L1 (somme des valeurs absolues des poids)
        l1_penalty = sum(p.abs().sum() for p in model.parameters())

        # Calcul de la pénalité L2 (somme des carrés des poids)
        l2_penalty = sum((p ** 2).sum() for p in model.parameters())

        # Loss totale
        loss = base_loss + l1_lambda * l1_penalty + l2_lambda * l2_penalty

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(targets)
        total_base_loss += base_loss.item() * len(targets)
        correct += ((outputs > 0.5) == targets).sum().item()
        n += len(targets)

    model.eval()
    val_loss, val_correct, val_n = 0.0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch["features"].to(device), batch["labels"].to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item() * len(targets)
            val_correct += ((outputs > 0.5) == targets).sum().item()
            val_n += len(targets)

    print(f"Epoch {epoch+1:2d} | loss totale {total_loss/n:.4f} | BCE train {total_base_loss/n:.4f} "
          f"| acc train {correct/n:.4f} | BCE val {val_loss/val_n:.4f} | acc val {val_correct/val_n:.4f}")
