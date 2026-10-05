import json
import random

import numpy as np
import torch


def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


def save_class_names(
    class_names,
    file_path
):

    with open(
        file_path,
        "w"
    ) as file:

        json.dump(
            class_names,
            file,
            indent=4
        )


def load_class_names(file_path):

    with open(
        file_path,
        "r"
    ) as file:

        return json.load(file)