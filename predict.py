import argparse
import json

import torch

from PIL import Image

from torchvision import transforms

from src.config import (
    DEVICE,
    MODEL_PATH,
    IMAGE_SIZE,
    CONFIDENCE_THRESHOLD
)

from src.model import create_model


def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    class_names = checkpoint[
        "class_names"
    ]

    model = create_model(
        num_classes=len(class_names)
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(DEVICE)

    model.eval()

    return model, class_names


def get_transform():

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


def predict(image_path):

    model, class_names = load_model()

    image = Image.open(
        image_path
    ).convert("RGB")

    transform = get_transform()

    image_tensor = transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(
        0
    )

    image_tensor = image_tensor.to(
        DEVICE
    )

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

    probabilities = probabilities[
        0
    ]

    top_probabilities, top_indices = torch.topk(
        probabilities,
        min(5, len(class_names))
    )

    top_predictions = []

    for probability, index in zip(
        top_probabilities,
        top_indices
    ):

        top_predictions.append({

            "design":
                class_names[index.item()],

            "confidence":
                float(probability.item())
        })

    best_prediction = top_predictions[0]

    print("\n========== PREDICTION ==========")

    print(
        f"Image: {image_path}"
    )

    print(
        f"Design: {best_prediction['design']}"
    )

    print(
        f"Confidence: "
        f"{best_prediction['confidence']:.2%}"
    )

    if (
        best_prediction["confidence"]
        < CONFIDENCE_THRESHOLD
    ):

        print(
            "\nWarning: Low confidence prediction."
        )

    print("\nTop Predictions:")

    for index, prediction in enumerate(
        top_predictions,
        start=1
    ):

        print(
            f"{index}. "
            f"{prediction['design']} - "
            f"{prediction['confidence']:.2%}"
        )

    return {
        "design":
            best_prediction["design"],

        "confidence":
            best_prediction["confidence"],

        "top_predictions":
            top_predictions
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Path to saree image"
    )

    args = parser.parse_args()

    predict(
        args.image
    )