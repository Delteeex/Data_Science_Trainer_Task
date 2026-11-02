
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


# Fashion-MNIST class labels
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


# Default model path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "models" / "improved_cnn_model.keras"
)


def preprocess_image(image_path):
    """
    Convert a real-world clothing image into
    Fashion-MNIST-compatible 28x28 grayscale format.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Open and convert image to grayscale
    image = Image.open(image_path).convert("L")

    # Improve contrast
    image = ImageOps.autocontrast(image)

    # Detect whether the image has a bright background
    image_array = np.array(image)

    border_pixels = np.concatenate([
        image_array[0, :],
        image_array[-1, :],
        image_array[:, 0],
        image_array[:, -1]
    ])

    # Fashion-MNIST uses a dark background
    if np.median(border_pixels) > 127:
        image = ImageOps.invert(image)

    image_array = np.array(image)

    # Find the approximate clothing region
    foreground_mask = image_array > 30

    if foreground_mask.any():
        rows, cols = np.where(foreground_mask)

        top, bottom = rows.min(), rows.max() + 1
        left, right = cols.min(), cols.max() + 1

        image = image.crop((left, top, right, bottom))

    # Resize while preserving aspect ratio
    image.thumbnail(
        (24, 24),
        Image.Resampling.LANCZOS
    )

    # Place clothing image in a black 28x28 canvas
    canvas = Image.new("L", (28, 28), color=0)

    x_offset = (28 - image.width) // 2
    y_offset = (28 - image.height) // 2

    canvas.paste(image, (x_offset, y_offset))

    # Normalize pixel values to 0-1
    image_array = np.array(
        canvas,
        dtype=np.float32
    ) / 255.0

    return image_array


def predict_image(
    image_path,
    model_path=DEFAULT_MODEL_PATH,
    top_k=3
):
    """
    Predict the top clothing classes for an image.

    Returns:
        List of dictionaries containing class names
        and prediction probabilities.
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    if not 1 <= top_k <= len(CLASS_NAMES):
        raise ValueError("top_k must be between 1 and 10.")

    # Load trained model
    model = tf.keras.models.load_model(model_path)

    # Preprocess input image
    processed_image = preprocess_image(image_path)

    # Adapt input shape to the loaded model
    if len(model.input_shape) == 4:
        model_input = processed_image.reshape(
            1, 28, 28, 1
        )
    elif len(model.input_shape) == 3:
        model_input = processed_image.reshape(
            1, 28, 28
        )
    else:
        raise ValueError(
            f"Unsupported model input shape: {model.input_shape}"
        )

    # Generate predictions
    probabilities = model.predict(
        model_input,
        verbose=0
    )[0]

    # Get top predictions
    top_indices = np.argsort(probabilities)[::-1][:top_k]

    results = []

    for index in top_indices:
        results.append({
            "class": CLASS_NAMES[index],
            "probability": float(probabilities[index])
        })

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Fashion-MNIST clothing image prediction"
    )

    parser.add_argument(
        "image_path",
        help="Path to JPG or PNG clothing image"
    )

    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to trained Keras model"
    )

    args = parser.parse_args()

    predictions = predict_image(
        args.image_path,
        model_path=args.model
    )

    print("\nFashion Image Predictions")
    print("-------------------------")

    for rank, result in enumerate(predictions, start=1):

        print(
            f"{rank}. {result['class']}: "
            f"{result['probability'] * 100:.2f}%"
        )
