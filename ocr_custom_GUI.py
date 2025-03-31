import os
import torch
import argparse
import torch.nn as nn
from torchvision import models
from PIL import Image, ImageTk
import torchvision.transforms as transforms
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk

# Default paths
DEFAULT_MODEL_PATH = '/home/aman-nvidia/My_files/ai_projects/ocr_one/models/best_ocr_model.pth'
DEFAULT_IMAGE_PATH = '/home/aman-nvidia/My_files/ai_projects/ocr_one/test_images/A_3.png'

class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR Character Recognition")
        self.root.geometry("1200x700")
        
        # Create main frames
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(side=tk.TOP, pady=10)
        
        self.result_frame = tk.Frame(self.root)
        self.result_frame.pack(side=tk.BOTTOM, pady=10, fill=tk.BOTH, expand=True)
        
        # Setup image display labels
        self.original_label = tk.Label(self.image_frame)
        self.original_label.grid(row=0, column=0, padx=10)
        
        self.gray_label = tk.Label(self.image_frame)
        self.gray_label.grid(row=0, column=1, padx=10)
        
        self.processed_label = tk.Label(self.image_frame)
        self.processed_label.grid(row=0, column=2, padx=10)
        
        # Prediction display
        self.prediction_frame = tk.Frame(self.result_frame)
        self.prediction_frame.pack(side=tk.LEFT, padx=20)
        
        self.prediction_label = tk.Label(self.prediction_frame, text="Predicted:", font=('Helvetica', 16, 'bold'), fg='blue')
        self.prediction_label.pack()
        
        self.character_label = tk.Label(self.prediction_frame, text="", font=('Helvetica', 36, 'bold'), fg='red')
        self.character_label.pack()
        
        self.confidence_label = tk.Label(self.prediction_frame, text="", font=('Helvetica', 14))
        self.confidence_label.pack()
        
        # Top predictions display
        self.predictions_frame = tk.Frame(self.result_frame)
        self.predictions_frame.pack(side=tk.RIGHT, padx=20)
        
        self.predictions_title = tk.Label(self.predictions_frame, text="Top 5 Predictions:", font=('Helvetica', 14, 'bold'))
        self.predictions_title.pack()
        
        self.predictions_text = tk.Text(self.predictions_frame, height=8, width=25, font=('Courier', 12))
        self.predictions_text.pack()
        
        # Control buttons
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side=tk.BOTTOM, pady=10)
        
        self.load_button = tk.Button(self.control_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=10)
        
        self.predict_button = tk.Button(self.control_frame, text="Predict", command=self.predict)
        self.predict_button.pack(side=tk.LEFT, padx=10)
        
        # Initialize model
        self.model = self.load_model(DEFAULT_MODEL_PATH)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Current image
        self.current_image_path = DEFAULT_IMAGE_PATH
        self.display_image(self.current_image_path)
    
    def load_model(self, model_path):
        """Load model with EXACT architecture used during training"""
        print(f"Loading model from {model_path}...")
        
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(512, 36)  # 36 classes (0-9, A-Z)
        )
        
        model.load_state_dict(torch.load(model_path), strict=False)
        model.eval()
        return model
    
    def display_image(self, image_path):
        """Display the original, grayscale, and processed images"""
        try:
            # Open and process images
            img = Image.open(image_path)
            gray_img = img.convert('L')
            processed_img = gray_img.resize((37, 37), Image.Resampling.LANCZOS)
            
            # Resize for display (keeping aspect ratio)
            display_size = (300, 300)
            img_display = img.resize(display_size)
            gray_display = gray_img.resize(display_size)
            processed_display = processed_img.resize(display_size)
            
            # Convert to PhotoImage
            self.original_img = ImageTk.PhotoImage(img_display)
            self.gray_img = ImageTk.PhotoImage(gray_display)
            self.processed_img = ImageTk.PhotoImage(processed_display)
            
            # Update labels
            self.original_label.config(image=self.original_img)
            self.gray_label.config(image=self.gray_img)
            self.processed_label.config(image=self.processed_img)
            
            # Store current image
            self.current_image_path = image_path
            self.current_processed_img = processed_img
            
        except Exception as e:
            print(f"Error displaying image: {str(e)}")
    
    def load_image(self):
        """Open file dialog to load new image"""
        file_path = filedialog.askopenfilename()
        if file_path:
            self.display_image(file_path)
    
    def predict(self):
        """Run prediction on current image"""
        class_names = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)]
        
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(48),
            transforms.RandomCrop(37),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.RandomAffine(0, shear=10),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        try:
            # Convert PIL Image to tensor
            img_tensor = transform(self.current_processed_img).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(img_tensor)
                _, predicted = torch.max(outputs, 1)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get top 5 predictions
            top5_prob, top5_indices = torch.topk(probabilities, 5)
            predicted_char = class_names[predicted.item()]
            
            # Update prediction display
            self.character_label.config(text=predicted_char)
            self.confidence_label.config(text=f"Confidence: {top5_prob[0][0].item():.2%}")
            
            # Update top predictions
            self.predictions_text.delete(1.0, tk.END)
            self.predictions_text.insert(tk.END, "Rank  Char  Confidence\n")
            self.predictions_text.insert(tk.END, "---------------------\n")
            for i, (prob, idx) in enumerate(zip(top5_prob[0], top5_indices[0])):
                self.predictions_text.insert(tk.END, f"{i+1:>2}.   {class_names[idx.item()]:<3}   {prob.item():.2%}\n")
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()