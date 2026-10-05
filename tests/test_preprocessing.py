from PIL import Image

from src.preprocessing import (
    resize_image,
    convert_to_grayscale
)


def test_resize_image():

    image = Image.new(
        "RGB",
        (500, 500)
    )

    resized = resize_image(
        image,
        (224, 224)
    )

    assert resized.size == (
        224,
        224
    )


def test_grayscale():

    image = Image.new(
        "RGB",
        (224, 224)
    )

    grayscale = convert_to_grayscale(
        image
    )

    assert grayscale.mode == "L"