# OCR Model Testing Script
"""
Amanuel Mihiret
ML|CV|DL Expert and freelancer at Upwork

"""

# Import required packages:
import os
import torch
import argparse
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Default paths (DEFINED YOUR MATCH PATH)
DEFAULT_MODEL_PATH = '/home/aman-nvidia/My_files/ai_projects/ocr_one/models/best_ocr_model.pth'
DEFAULT_IMAGE_PATH = '/home/aman-nvidia/My_files/ai_projects/ocr_one/test_images/A_3.png'

def load_model(model_path):
    """Load model with EXACT architecture used during training"""
    print(f"Loading model from {model_path}...")
    
    # Create model with original training architecture
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    
    # This must match EXACTLY with the training script
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.25),
        nn.Linear(512, 36)  # 36 classes (0-9, A-Z)
    )
    
    # Load with strict=False to handle any minor mismatches
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()
    return model

def get_image_info(image):
    """Get detailed image information"""
    return {
        'size': image.size,
        'mode': image.mode,
        'channels': len(image.getbands()),
        'format': image.format,
        'dtype': np.array(image).dtype
    }

def preprocess_image(image_path, target_size=(37, 37)):
    """Preprocess image with direct resizing (no padding/cropping)"""
    try:
        # Open image and convert to grayscale
        img = Image.open(image_path)
        original_info = get_image_info(img)
        img = img.convert('L')
        
        # Direct resize to target dimensions
        processed_img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        return original_info, img, processed_img
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

def predict(image_path, model_path):
    """Run prediction on test image"""
    # Define class names
    class_names = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)]
    
    # Check files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load model
    model = load_model(model_path).to(device)
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(48),  # Slightly larger for pretrained models
        transforms.RandomCrop(37),  # Random 37x37 crops
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.RandomAffine(0, shear=10),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    try:
        # Get original and processed images
        original_info, gray_img, processed_img = preprocess_image(image_path)
        img_tensor = transform(processed_img).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top 5 predictions
        top5_prob, top5_indices = torch.topk(probabilities, 5)
        predicted_char = class_names[predicted.item()]
        
        # Create figure with better layout
        plt.figure(figsize=(16, 6))
        
        # Original image
        plt.subplot(1, 5, 1)
        original_img = Image.open(image_path)
        plt.imshow(original_img)
        plt.title(f"Original\n{original_info['size']} {original_info['mode']}", pad=10)
        plt.axis('off')
        
        # Grayscale version
        plt.subplot(1, 5, 2)
        plt.imshow(gray_img, cmap='gray')
        plt.title(f"Grayscale\n{gray_img.size} {gray_img.mode}", pad=10)
        plt.axis('off')
        
        # Processed image
        plt.subplot(1, 5, 3)
        plt.imshow(processed_img, cmap='gray')
        plt.title(f"Model Input\n{processed_img.size}", pad=10)
        plt.axis('off')
        
        # Prediction display
        plt.subplot(1, 5, 4)
        plt.text(0.5, 0.7, "Predicted:", 
                fontsize=24, ha='center', va='center',
                fontweight='bold', color='blue')
        plt.text(0.5, 0.5, predicted_char,
                fontsize=36, ha='center', va='center',
                color='red', fontweight='bold')
        plt.text(0.5, 0.3, "Confidence: {:.2%}".format(top5_prob[0][0].item()),
                fontsize=18, ha='center', va='center')
        plt.axis('off')

        # Top 5 predictions display - NEW SECTION
        plt.subplot(1, 5, 5)
        prediction_text = "Top 5 Predictions:\n" + "-"*20 + "\n"
        for i, (prob, idx) in enumerate(zip(top5_prob[0], top5_indices[0])):
            prediction_text += f"{i+1:>2}. {class_names[idx.item()]:<3} {prob.item():.2%}\n"
        
        plt.text(0.1, 0.9, prediction_text,
                fontfamily='monospace',
                fontsize=14,
                va='top')
        plt.axis('off')
        
        # Main title
        plt.suptitle(f"OCR Character Recognition Results", y=0.98, fontsize=16)
        plt.tight_layout()
        
        # Print console results
        print("\n" + "="*60)
        print(f"{'OCR PREDICTION RESULTS':^60}")
        print("="*60)
        print(f"Predicted class: {predicted_char}")
        print("\nTop 5 predictions:")
        print("-"*60)
        for i, (prob, idx) in enumerate(zip(top5_prob[0], top5_indices[0])):
            print(f"{i+1:>2}. {class_names[idx.item()]:<5} {prob.item():.2%}")
        print("="*60)
        
        # Show plot
        plt.show()
        
    except Exception as e:
        print(f"\nError during prediction: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OCR Model Tester')
    parser.add_argument('--image', type=str, default=DEFAULT_IMAGE_PATH, 
                       help='Path to test image')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                       help='Path to trained model')
    args = parser.parse_args()

    print("\n" + "="*30)
    print(f"{'OCR MODEL TESTER':^30}")
    print("="*30)
    predict(args.image, args.model)

                                    # End of the script