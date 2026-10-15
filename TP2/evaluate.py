import sys
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix

from train_optimizers import train_model, test_loader, device


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

    print(f"Accuracy: {accuracy_score(all_targets, all_preds_classes):.4f}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    print("Matrice de confusion :")
    print(confusion_matrix(all_targets, all_preds_classes))


if __name__ == "__main__":
    opt = sys.argv[1] if len(sys.argv) > 1 else "Adam"
    model = train_model(opt, learning_rate=0.001)
    evaluate_model(model, test_loader)
