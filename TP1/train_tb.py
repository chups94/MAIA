import os
import random
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms, datasets

from torch.utils.data import random_split, DataLoader

from torch.utils.tensorboard import SummaryWriter



# ==========================================
# ETAPE 1 : DONNEES
# ==========================================

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        CIFAR10_MEAN,
        CIFAR10_STD
    )

])


trainset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


testset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)



# ==========================================
# HYPERPARAMETRES
# ==========================================

hparams = dict(

    model="MLP",

    batch_size=32,

    lr=1e-2,

    seed=0,

    weight_decay=0.0

)



# ==========================================
# SEED
# ==========================================

torch.manual_seed(hparams["seed"])

random.seed(hparams["seed"])


if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        hparams["seed"]
    )



# ==========================================
# TENSORBOARD
# ==========================================

run_name = (
    f"{hparams['model']}/"
    f"bs{hparams['batch_size']}_"
    f"lr{hparams['lr']}_"
    f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
)


logdir = os.path.join(
    "runs",
    run_name
)


print("Logdir:", logdir)


writer = SummaryWriter(
    log_dir=logdir
)



# ==========================================
# TRAIN / VALIDATION SPLIT
# ==========================================

N = len(trainset)


val_size = int(
    0.1 * N
)


train_size = (
    N - val_size
)


train_subset, val_subset = random_split(

    trainset,

    [
        train_size,
        val_size
    ],

    generator=torch.Generator().manual_seed(
        hparams["seed"]
    )

)



trainloader = DataLoader(

    train_subset,

    batch_size=hparams["batch_size"],

    shuffle=True,

    pin_memory=True

)



valloader = DataLoader(

    val_subset,

    batch_size=hparams["batch_size"],

    shuffle=False,

    pin_memory=True

)



testloader = DataLoader(

    testset,

    batch_size=hparams["batch_size"],

    shuffle=False,

    pin_memory=True

)



# ==========================================
# ETAPE 2 : MODELE
# ==========================================

class MLP(nn.Module):

    def __init__(self):

        super().__init__()


        self.fc1 = nn.Linear(
            32 * 32 * 3,
            128
        )


        self.fc2 = nn.Linear(
            128,
            10
        )


    def forward(self, x):

        x = torch.flatten(
            x,
            1
        )


        x = F.relu(
            self.fc1(x)
        )


        x = self.fc2(x)


        return x



# ==========================================
# DEVICE
# ==========================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("Using device:", device)



# ==========================================
# CREATION DU MODELE
# ==========================================

model = MLP().to(
    device
)



criterion = nn.CrossEntropyLoss()



optimizer = torch.optim.SGD(

    model.parameters(),

    lr=hparams["lr"],

    momentum=0.9,

    weight_decay=hparams["weight_decay"]

)



# ==========================================
# ETAPE 2 : FONCTION METRIQUES
# ==========================================

@torch.no_grad()
def epoch_metrics(
    loader,
    model,
    criterion,
    device
):


    model.eval()


    loss_sum = 0.0

    correct = 0

    total = 0


    for x, y in loader:


        x = x.to(
            device,
            non_blocking=True
        )


        y = y.to(
            device,
            non_blocking=True
        )


        logits = model(x)


        loss = criterion(
            logits,
            y
        )


        loss_sum += (
            loss.item()
            * y.size(0)
        )


        pred = logits.argmax(1)


        correct += (
            pred == y
        ).sum().item()


        total += y.size(0)


    loss_avg = (
        loss_sum / total
    )


    acc = (
        correct / total
    )


    return loss_avg, acc



# ==========================================
# ETAPE 3 : ENTRAINEMENT + TENSORBOARD
# ==========================================

global_step = 0


EPOCHS = 10



for epoch in range(
    1,
    EPOCHS + 1
):


    model.train()


    running_loss_sum = 0.0

    running_total = 0


    for b, (x, y) in enumerate(
        trainloader
    ):


        x = x.to(
            device,
            non_blocking=True
        )


        y = y.to(
            device,
            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        logits = model(x)


        loss = criterion(
            logits,
            y
        )


        loss.backward()



        # LOG LOSS TOUS LES 10 BATCHS

        if b % 10 == 0:

            writer.add_scalar(

                "Loss/train_step",

                loss.item(),

                global_step

            )



        optimizer.step()



        running_loss_sum += (
            loss.item()
            * y.size(0)
        )


        running_total += (
            y.size(0)
        )


        global_step += 1



    # ======================================
    # METRIQUES FIN D'EPOCH
    # ======================================

    train_loss = (
        running_loss_sum
        / running_total
    )


    val_loss, val_acc = epoch_metrics(

        valloader,

        model,

        criterion,

        device

    )



    # ======================================
    # LOG TENSORBOARD
    # ======================================

    writer.add_scalar(

        "Loss/train",

        train_loss,

        epoch

    )


    writer.add_scalar(

        "Loss/val",

        val_loss,

        epoch

    )


    writer.add_scalar(

        "Accuracy/val",

        val_acc,

        epoch

    )



    print(

        f"Epoch {epoch:02d} | "

        f"train_loss={train_loss:.4f} | "

        f"val_loss={val_loss:.4f} | "

        f"val_acc={val_acc:.3f}"

    )



# ==========================================
# FERMETURE TENSORBOARD
# ==========================================

writer.close()



# ==========================================
# TEST FINAL
# ==========================================

test_loss, test_acc = epoch_metrics(

    testloader,

    model,

    criterion,

    device

)


print(
    f"Test loss={test_loss:.4f} | "
    f"Test accuracy={test_acc:.3f}"
)