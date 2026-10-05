import json

import torch

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from .config import (
    DEVICE,
    MODEL_PATH,
    METRICS_DIR,
    CONFUSION_MATRIX_DIR
)

from .dataset import create_dataloaders

from .model import create_model


def evaluate_model():

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CONFUSION_MATRIX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        train_loader,
        validation_loader,
        test_loader,
        class_names
    ) = create_dataloaders()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model = create_model(
        num_classes=len(class_names)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)

    model.eval()

    all_predictions = []

    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    print("\n========== RESULTS ==========")

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=class_names,
            zero_division=0
        )
    )

    metrics = {

        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    metrics_path = (
        METRICS_DIR /
        "evaluation_metrics.json"
    )

    with open(
        metrics_path,
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    plt.figure(
        figsize=(10, 8)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.title(
        "Saree Design Confusion Matrix"
    )

    plt.tight_layout()

    matrix_path = (
        CONFUSION_MATRIX_DIR /
        "confusion_matrix.png"
    )

    plt.savefig(
        matrix_path
    )

    plt.close()

    print(
        f"\nConfusion matrix saved to: "
        f"{matrix_path}"
    )


if __name__ == "__main__":

    evaluate_model()