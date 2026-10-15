import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from dataset import get_loaders


# ==========================================
# DONNEES
# ==========================================

torch.manual_seed(42)

dataset, train_loader, val_loader, test_loader = get_loaders("data/cardio_train.csv", batch_size=64)
batch = next(iter(train_loader))
INPUT_SIZE = batch['features'].shape[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# MODELE
# ==========================================

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


def accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch["features"].to(device), batch["labels"].to(device)
            preds = (model(inputs) > 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total


# ==========================================
# PARTIE 2 : REGULARISATION L1 / L2
# ==========================================

def train_regularized(l1_lambda=1e-4, l2_lambda=1e-3, epochs=10, lr=0.01):
    model = MLP(input_size=INPUT_SIZE, hidden_size=128).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    writer = SummaryWriter(f'runs/cardio_reg_l1{l1_lambda}_l2{l2_lambda}')

    for epoch in range(epochs):
        model.train()
        running_base, running_total = 0.0, 0.0
        for batch in train_loader:
            inputs, targets = batch["features"].to(device), batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            base_loss = criterion(outputs, targets)

            # Calcul de la pénalité L1 (somme des valeurs absolues des poids)
            l1_penalty = sum(p.abs().sum() for p in model.parameters())

            # Calcul de la pénalité L2 (somme des carrés des poids)
            l2_penalty = sum(p.pow(2).sum() for p in model.parameters())

            # Loss totale
            loss = base_loss + l1_lambda * l1_penalty + l2_lambda * l2_penalty

            loss.backward()
            optimizer.step()

            running_base += base_loss.item()
            running_total += loss.item()

        n = len(train_loader)
        train_acc = accuracy(model, train_loader)
        val_acc = accuracy(model, val_loader)
        writer.add_scalar('BCE Loss', running_base / n, epoch)
        writer.add_scalar('Total Loss', running_total / n, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        print(f"[L1={l1_lambda} L2={l2_lambda}] epoch {epoch+1:2d} | BCE {running_base/n:.4f} | "
              f"total {running_total/n:.4f} | train acc {train_acc:.4f} | val acc {val_acc:.4f}")

    writer.close()
    return model


# ==========================================
# PARTIE 3 : COMPARAISON DES OPTIMISEURS
# ==========================================

def train_model(opt_name, learning_rate=0.001, epochs=30):
    model = MLP(input_size=INPUT_SIZE, hidden_size=128).to(device)
    criterion = nn.BCELoss()

    # Choix de l'optimiseur
    if opt_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif opt_name == "Momentum":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif opt_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter(f'runs/cardio_{opt_name}_lr{learning_rate}')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            inputs, targets = batch["features"].to(device), batch["labels"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Logging de la loss moyenne par epoch
        writer.add_scalar('Training Loss', running_loss / len(train_loader), epoch)
        writer.add_scalar('Validation Accuracy', accuracy(model, val_loader), epoch)
        print(f"[{opt_name}] epoch {epoch+1:2d} | loss {running_loss / len(train_loader):.4f}")

    writer.close()
    return model


# ==========================================
# PARTIE 4 : METRIQUES
# ==========================================

def evaluate_model(model, test_loader):
    model.eval()
    all_targets = []
    all_preds_probs = []

    with torch.no_grad(): # Désactiver le calcul des gradients
        for batch in test_loader:
            inputs, targets = batch["features"].to(device), batch["labels"].to(device)
            outputs = model(inputs)

            # Stocker les probabilités et les cibles pour sklearn
            all_targets.extend(targets.cpu().numpy())
            all_preds_probs.extend(outputs.cpu().numpy())

    # Conversion en numpy array
    all_targets = np.array(all_targets)
    all_preds_probs = np.array(all_preds_probs)

    # Prédictions binaires (seuil à 0.5)
    all_preds_classes = (all_preds_probs > 0.5).astype(int)

    # Calcul des métriques
    precision = precision_score(all_targets, all_preds_classes)
    recall = recall_score(all_targets, all_preds_classes)
    f1 = f1_score(all_targets, all_preds_classes)
    auc = roc_auc_score(all_targets, all_preds_probs)

    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    return precision, recall, f1, auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["reg", "optim", "all"], default="all")
    args = parser.parse_args()

    if args.part in ("reg", "all"):
        # Régularisation de référence puis régularisation L1 trop forte
        train_regularized(l1_lambda=1e-4, l2_lambda=1e-3)
        train_regularized(l1_lambda=0.1, l2_lambda=0)

    if args.part in ("optim", "all"):
        # Lancement des expériences
        models = {}
        for opt in ["SGD", "Momentum", "RMSprop", "Adam"]:
            models[opt] = train_model(opt, learning_rate=0.001)

        # Evaluation finale du meilleur modèle (meilleure accuracy de validation)
        best = max(models, key=lambda name: accuracy(models[name], val_loader))
        print(f"\nMeilleur modèle (val) : {best}")
        evaluate_model(models[best], test_loader)
        torch.save(models[best].state_dict(), f"cardio_{best}.pth")
