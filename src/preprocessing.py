from pathlib import Path
from PIL import Image
import cv2
import numpy as np


def load_image(image_path):
    """
    Load image using PIL.
    """

    image = Image.open(image_path)

    return image.convert("RGB")


def resize_image(image, size=(224, 224)):
    """
    Resize PIL image.
    """

    return image.resize(size)


def convert_to_grayscale(image):
    """
    Convert image to grayscale.
    """

    return image.convert("L")


def convert_to_hsv(image):
    """
    Convert PIL image to HSV using OpenCV.
    """

    image_np = np.array(image)

    hsv = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2HSV
    )

    return hsv


def convert_to_lab(image):
    """
    Convert PIL image to LAB.
    """

    image_np = np.array(image)

    lab = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2LAB
    )

    return lab