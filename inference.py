"""
Astronomy Image Classifier - Inference Script

This script performs inference on single astronomical images using a trained PyTorch model.
It loads a pre-trained ResNet18-based model and predicts whether an input image contains
a star or galaxy.

Usage:
    python inference.py --image path/to/image.jpg [--model model.pth] [--config config.yaml]

Examples:
    # Basic usage with default model
    python inference.py --image test_galaxy.jpg
    
    # Specify custom model and config
    python inference.py --image test_star.jpg --model my_model.pth --config my_config.yaml

Arguments:
    --image: Path to the input image file (required)
    --model: Path to the trained model checkpoint (default: best_astronomy_model.pth)
    --config: Path to the configuration YAML file (default: config.yaml)

Output:
    Prints the predicted class (Star/Galaxy) and confidence score.

Requirements:
    - torch, torchvision, PIL, pyyaml
    - astronomy_classifier.py (contains AstroCNN class)
    - Trained model checkpoint file
    - Configuration YAML file

Author: Edoardo Tesei
Date: August, 2025
"""

import torch
from torchvision import transforms
from PIL import Image
import yaml
import argparse
from astronomy_classifier import AstroCNN
from src.colours import *


def load_model(model_path, config_path="configs/config.yaml"):
    """Load trained model from checkpoint"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    model = AstroCNN(num_classes=config['model']['num_classes'])
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict_image(model, image_path):
    """Predict single image"""
    # Define transforms (same as validation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        _, predicted = torch.max(outputs, 1)
    
    class_names = ['Star', 'Galaxy']
    predicted_class = class_names[predicted.item()]
    confidence = probabilities[predicted.item()].item()
    
    return predicted_class, confidence

def main():
    parser = argparse.ArgumentParser(description='Astronomy Image Classifier Inference')
    parser.add_argument('--image', required=True, help='Path to image file')
    parser.add_argument('--model', default='best_astronomy_model.pth', help='Path to model checkpoint')
    parser.add_argument('--config', default='configs/config.yaml', help='Path to config file')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model = load_model(args.model, args.config)
    
    # Predict
    print(f"{bcolors.BOLD}{bcolors.DARK_GRAY}Predicting image: {args.image}{bcolors.ENDC}")
    predicted_class, confidence = predict_image(model, args.image)
    
    if predicted_class == 'Galaxy':
        print(f"{bcolors.DARK_CYAN}Prediction: {bcolors.BLUE}{predicted_class}{bcolors.ENDC}")
    elif predicted_class == 'Star':
        print(f"{bcolors.DARK_CYAN}Prediction: {bcolors.YELLOW}{predicted_class}{bcolors.ENDC}")
    print(f"{bcolors.DARK_RED_256}Confidence: {confidence:.4f} ({confidence*100:.2f}%){bcolors.ENDC}")

if __name__ == "__main__":
    main()