import random
import shutil

from pathlib import Path

from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_DIR,
    SPLIT_DATA_DIR
)


RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


def split_dataset():

    random.seed(
        RANDOM_SEED
    )

    if not RAW_DATA_DIR.exists():

        raise FileNotFoundError(
            f"Raw dataset not found: "
            f"{RAW_DATA_DIR}"
        )

    for class_dir in RAW_DATA_DIR.iterdir():

        if not class_dir.is_dir():

            continue

        class_name = class_dir.name

        images = [
            image
            for image in class_dir.iterdir()
            if image.suffix.lower()
            in [
                ".jpg",
                ".jpeg",
                ".png",
                ".webp"
            ]
        ]

        if len(images) < 10:

            print(
                f"Skipping {class_name}: "
                f"not enough images."
            )

            continue

        train_images, temp_images = train_test_split(
            images,
            test_size=(
                1 - TRAIN_RATIO
            ),
            random_state=RANDOM_SEED
        )

        validation_images, test_images = train_test_split(
            temp_images,
            test_size=0.50,
            random_state=RANDOM_SEED
        )

        splits = {

            "train":
                train_images,

            "validation":
                validation_images,

            "test":
                test_images
        }

        for split_name, split_images in splits.items():

            destination = (
                SPLIT_DATA_DIR
                / split_name
                / class_name
            )

            destination.mkdir(
                parents=True,
                exist_ok=True
            )

            for image in split_images:

                shutil.copy2(
                    image,
                    destination / image.name
                )

        print(
            f"{class_name}: "
            f"{len(train_images)} train, "
            f"{len(validation_images)} validation, "
            f"{len(test_images)} test"
        )


if __name__ == "__main__":

    split_dataset()