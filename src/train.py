import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from .config import (
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    MODEL_PATH,
    MODEL_DIR,
    PLOTS_DIR
)

from .dataset import create_dataloaders

from .model import create_model

from .utils import set_seed


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    progress_bar = tqdm(
        loader,
        desc="Training"
    )

    for images, labels in progress_bar:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

        progress_bar.set_postfix(
            loss=loss.item()
        )

    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def validate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    loss = running_loss / total

    accuracy = correct / total

    return loss, accuracy


def main():

    set_seed(42)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        train_loader,
        validation_loader,
        test_loader,
        class_names
    ) = create_dataloaders()

    print("\nClasses:")

    for index, name in enumerate(class_names):

        print(
            f"{index}: {name}"
        )

    model = create_model(
        num_classes=len(class_names)
    )

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_validation_accuracy = 0.0

    history = {

        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": []
    }

    for epoch in range(NUM_EPOCHS):

        print(
            f"\nEpoch {epoch + 1}/{NUM_EPOCHS}"
        )

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        validation_loss, validation_accuracy = validate(
            model,
            validation_loader,
            criterion
        )

        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["validation_loss"].append(
            validation_loss
        )

        history["validation_accuracy"].append(
            validation_accuracy
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: {train_accuracy:.4f}"
        )

        print(
            f"Validation Loss: {validation_loss:.4f}"
        )

        print(
            f"Validation Accuracy: {validation_accuracy:.4f}"
        )

        if validation_accuracy > best_validation_accuracy:

            best_validation_accuracy = validation_accuracy

            checkpoint = {

                "model_state_dict":
                    model.state_dict(),

                "class_names":
                    class_names,

                "validation_accuracy":
                    validation_accuracy
            }

            torch.save(
                checkpoint,
                MODEL_PATH
            )

            print(
                f"Best model saved: {MODEL_PATH}"
            )

    history_path = PLOTS_DIR / "training_history.json"

    with open(
        history_path,
        "w"
    ) as file:

        json.dump(
            history,
            file,
            indent=4
        )

    print("\nTraining completed.")

    print(
        f"Best validation accuracy: "
        f"{best_validation_accuracy:.4f}"
    )


if __name__ == "__main__":

    main()