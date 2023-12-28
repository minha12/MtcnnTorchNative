import argparse
import numpy as np
import torch
from PIL import Image
from mtcnn import MTCNN  # Replace with your actual MTCNN import
import torchvision.transforms as transforms


def detect_faces(image_path):
    # Load your MTCNN model
    mtcnn = MTCNN()

    # Load image
    image = Image.open(image_path)

    # Detect faces
    boxes, _ = mtcnn.detect_faces(image)

    # Process output as needed
    return boxes


def main():
    parser = argparse.ArgumentParser(description="Detect faces in an image using MTCNN")
    parser.add_argument("-p", "--image_path", type=str, help="Path to the image file")

    args = parser.parse_args()

    # Detect faces
    # boxes = detect_faces(args.image_path)
    boxes = detect_faces(args.image_path)
    # Print or process the result
    if boxes is not None:
        print(f"Detected faces at: {boxes}")
    else:
        print("No faces detected.")


if __name__ == "__main__":
    main()
