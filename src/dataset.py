from pathlib import Path

from torchvision import datasets
from torch.utils.data import DataLoader

from .augmentation import (
    get_train_transform,
    get_validation_transform
)

from .config import (
    TRAIN_DIR,
    VALIDATION_DIR,
    TEST_DIR,
    BATCH_SIZE,
    NUM_WORKERS
)


def create_datasets():

    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=get_train_transform()
    )

    validation_dataset = datasets.ImageFolder(
        root=VALIDATION_DIR,
        transform=get_validation_transform()
    )

    test_dataset = datasets.ImageFolder(
        root=TEST_DIR,
        transform=get_validation_transform()
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset
    )


def create_dataloaders():

    (
        train_dataset,
        validation_dataset,
        test_dataset
    ) = create_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
        train_dataset.classes
    )