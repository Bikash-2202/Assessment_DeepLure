import torch

from src.model import create_model


def test_model_output():

    model = create_model(
        num_classes=5
    )

    sample = torch.randn(
        2,
        3,
        224,
        224
    )

    output = model(
        sample
    )

    assert output.shape == (
        2,
        5
    )