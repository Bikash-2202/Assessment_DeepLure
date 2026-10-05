# Color-Invariant Saree Design Recognition

An AI-based computer vision system that recognizes saree designs while reducing dependency on saree color.

## Objective

The objective is to classify saree designs based on patterns, motifs, texture and structure rather than dominant color.

## Technology

- Python
- PyTorch
- Torchvision
- OpenCV
- Scikit-learn
- FastAPI
- Matplotlib
- Seaborn

## Model

The project uses ResNet18 transfer learning.

Color invariance is achieved using:

- ColorJitter
- Brightness augmentation
- Contrast augmentation
- Saturation augmentation
- Hue augmentation
- Rotation
- Cropping
- Horizontal flipping

## Dataset

The dataset follows the ImageFolder structure:

dataset/split/
    train/
    validation/
    test/

Each directory contains one folder per saree design.

Example:

dataset/split/train/
    banarasi/
    kanjivaram/
    paithani/
    chanderi/
    tant/

## Installation

Create a virtual environment:

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

## Training

Run:

python -m src.train

The best model will be saved inside:

models/

## Evaluation

Run:

python -m src.evaluate

Evaluation generates:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix

## Prediction

Run:

python predict.py --image sample_images/test.jpg

## API

Start the API:

uvicorn api.app:app --reload

Open:

http://127.0.0.1:8000/docs

## Project Architecture

Image
    ↓
Preprocessing
    ↓
Color Augmentation
    ↓
ResNet18
    ↓
Feature Extraction
    ↓
Classification
    ↓
Saree Design

## Color Invariance

The model is trained using strong color augmentation so that the same design with different colors is classified as the same design.

Example:

Red + Design A
Blue + Design A
Green + Design A

should all produce:

Design A

## Future Improvements

- EfficientNet
- Vision Transformer
- Grad-CAM
- Similarity search
- Vector embeddings
- Docker deployment
- Cloud deployment