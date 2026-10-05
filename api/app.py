import io

import torch

from PIL import Image

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException
)

from torchvision import transforms

from src.config import (
    DEVICE,
    MODEL_PATH,
    IMAGE_SIZE,
    CONFIDENCE_THRESHOLD
)

from src.model import create_model


app = FastAPI(
    title="Saree Design Recognition API",
    description="Color-Invariant Saree Design Recognition",
    version="1.0.0"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

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

model = model.to(
    DEVICE
)

model.eval()


# --------------------------------------------------
# TRANSFORM
# --------------------------------------------------

transform = transforms.Compose([

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


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/")
def root():

    return {

        "message":
            "Saree Design Recognition API",

        "status":
            "running"
    }


@app.get("/health")
def health():

    return {

        "status":
            "healthy"
    }


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

@app.post("/predict")
async def predict(
    image: UploadFile = File(...)
):

    if not image.content_type.startswith(
        "image/"
    ):

        raise HTTPException(
            status_code=400,
            detail="Please upload an image."
        )

    contents = await image.read()

    try:

        pil_image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid image."
        )

    image_tensor = transform(
        pil_image
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
                round(
                    float(probability.item()),
                    4
                )
        })

    best = top_predictions[0]

    return {

        "filename":
            image.filename,

        "design":
            best["design"],

        "confidence":
            best["confidence"],

        "is_confident":
            best["confidence"]
            >= CONFIDENCE_THRESHOLD,

        "top_predictions":
            top_predictions
    }