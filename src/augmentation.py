from torchvision import transforms
from .config import IMAGE_SIZE


def get_train_transform():
    """
    Training transformations.

    Strong color augmentation is used so that the model
    does not depend heavily on saree color.
    """

    return transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=15
        ),

        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.80, 1.0)
        ),

        # Important for color invariance
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.8,
            hue=0.5
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])


def get_validation_transform():

    return transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])


def get_grayscale_transform():

    """
    Converts image to grayscale while keeping
    3 channels so it can be used with ResNet.
    """

    return transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.Grayscale(
            num_output_channels=3
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])