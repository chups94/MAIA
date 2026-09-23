import os
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets


# =========================
# ETAPE 1 : PREPARATION DES DONNEES
# =========================

# Normalisation "classique" pour CIFAR-10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


# CHARGEMENT DES DONNEES

trainset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

testset = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)


# Ne pas monopoliser les CPUs sur Slurm

def get_num_workers(default=2, cap=4):
    try:
        n = int(os.getenv("SLURM_CPUS_PER_TASK", default))
    except Exception:
        n = default

    return max(0, min(cap, n))


num_workers = get_num_workers()


trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=32,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True
)


testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)



# =========================
# ETAPE 2 : IMPLEMENTATION DU RESEAU
# =========================

class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        # Image 32x32 avec 3 canaux RGB = 3072 valeurs
        self.fc1 = nn.Linear(32 * 32 * 3, 128)

        # 128 neurones caches vers 10 classes
        self.fc2 = nn.Linear(128, 10)


    def forward(self, x):

        # Aplatir les images sans toucher a la dimension batch
        x = torch.flatten(x, 1)

        # Premiere couche + ReLU
        x = F.relu(self.fc1(x))

        # Couche de sortie
        x = self.fc2(x)

        # Pas de Softmax ici
        return x



# =========================
# ETAPE 3 : ENTRAINEMENT DU MODELE
# =========================

torch.manual_seed(0)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

random.seed(0)


# Choix du GPU si disponible
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")


# Creation du modele
model = MLP().to(device)


# Fonction de perte
criterion = nn.CrossEntropyLoss()


# Optimiseur
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)


# Nombre d'epochs
EPOCHS = 10


for epoch in range(EPOCHS):

    # Mode entrainement
    model.train()

    running_loss = 0.0
    running_correct = 0
    running_total = 0


    for inputs, labels in trainloader:

        # Envoyer les donnees sur le GPU
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)


        # 1. Reinitialiser les gradients
        optimizer.zero_grad(set_to_none=True)


        # 2. Passe avant
        outputs = model(inputs)


        # 3. Calcul de la perte
        loss = criterion(outputs, labels)


        # 4. Retropropagation
        loss.backward()


        # 5. Mise a jour des poids
        optimizer.step()


        # Statistiques
        running_loss += loss.item() * inputs.size(0)

        preds = outputs.argmax(dim=1)

        running_correct += (preds == labels).sum().item()

        running_total += labels.size(0)


    # Resultats de l'epoch
    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total


    print(
        f"Epoch {epoch+1:02d} | "
        f"loss={epoch_loss:.4f} | "
        f"acc={epoch_acc:.4f}"
    )



# =========================
# ETAPE 4 : EVALUATION SUR LE JEU DE TEST
# =========================

# Mode evaluation
model.eval()

classes = trainset.classes

total = 0
correct = 0


# Pas besoin de calculer les gradients pendant le test
with torch.no_grad():

    for images, labels in testloader:

        # Envoyer les donnees sur le GPU
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)


        # Predictions
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)


        # Statistiques
        total += labels.size(0)

        correct += (predicted == labels).sum().item()


# Accuracy finale sur le test
acc = correct / total

print(f"Test accuracy: {acc:.3f}")



# =========================
# ETAPE 5 : SAUVEGARDE DU MODELE
# =========================

# Sauvegarde des poids du modele
torch.save(model.state_dict(), "mlp_model.pth")

print("Modele sauvegarde dans mlp_model.pth")


# Exemple de chargement sur CPU
# A tester dans un script separe si besoin

# model2 = MLP().to("cpu")
# state = torch.load(
#     "mlp_model.pth",
#     map_location="cpu",
#     weights_only=True
# )
# model2.load_state_dict(state)
# model2.eval()
