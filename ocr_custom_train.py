# Import required packages
import os
import torch
import numpy as np 
import seaborn as sns 
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm 
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from IPython.display import clear_output
from sklearn.metrics import confusion_matrix

# Custom Dataset
class OCRDataset(Dataset):
    """Custom Dataset for loading OCR character images"""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the character folders (A-Z, 0-9)
            transform (callable, optional): Optional transform to be applied
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))  # Get class names from folder names
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.images = []
        
        # Load all image paths and their corresponding labels
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # Add other formats if needed
                    self.images.append((
                        os.path.join(cls_dir, img_name),
                        self.class_to_idx[cls]
                    ))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        
        # Load image (using PIL as it handles different image formats well)
        image = Image.open(img_path)
        
        # Convert to grayscale if not already
        if image.mode != 'L':
            image = image.convert('L')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    def get_class_names(self):
        """Helper method to get the ordered list of class names"""
        return self.classes

# Custom transformations for 37x37 images
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(48),  # Slightly larger for pretrained models
    transforms.RandomCrop(37),  # Random 37x37 crops
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.RandomAffine(0, shear=10),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(37),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Defined paths - MODIFY THESE TO YOUR ACTUAL PATHS
train_dir = '/home/aman-nvidia/My_files/ai_projects/ocr_one/ocr_dataset/training_data'
test_dir = '/home/aman-nvidia/My_files/ai_projects/ocr_one/ocr_dataset/testing_data'

# Create datasets
train_dataset = OCRDataset(train_dir, transform=train_transform)
test_dataset = OCRDataset(test_dir, transform=test_transform)

# Print class names
print("Training class names:", train_dataset.get_class_names())
print("Testing class names:", test_dataset.get_class_names())

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define model adaptation function
def get_pretrained_model(model_name='resnet18', num_classes=36, dropout_rate=0.5):
    """Enhanced pretrained model loader with better adaptation for OCR"""
    if model_name == 'resnet18':
        model = models.resnet18(weights='DEFAULT')
        model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate/2),
            nn.Linear(512, num_classes)
        )
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='DEFAULT')
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 256),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate/2),
            nn.Linear(256, num_classes)
        )
    elif model_name == 'mobilenetv3_small':
        model = models.mobilenet_v3_small(weights='DEFAULT')
        model.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: resnet18, efficientnet_b0, mobilenetv3_small")
    
    # Initialize weights
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
    
    return model

# Initialize model
model = get_pretrained_model('resnet18', num_classes=36)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training setup
num_epochs = 5
criterion = nn.CrossEntropyLoss()

# Freeze early layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze classifier
if hasattr(model, 'classifier'):
    for param in model.classifier.parameters():
        param.requires_grad = True
elif hasattr(model, 'fc'):
    for param in model.fc.parameters():
        param.requires_grad = True

optimizer = optim.AdamW([
    {'params': [p for p in model.parameters() if p.requires_grad]},
], lr=0.001)

scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 
                                       max_lr=0.01,
                                       steps_per_epoch=len(train_loader),
                                       epochs=num_epochs)

# Evaluation function
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return running_loss / len(loader), 100 * correct / total

# Training function
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs):
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    lrs = []
    
    best_acc = 0.0
    early_stop_counter = 0
    patience = 3
    
    plt.figure(figsize=(15, 5))
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        val_loss, val_acc = evaluate(model, test_loader, criterion)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_acc)
        else:
            scheduler.step()
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        
        clear_output(wait=True)
        plt.clf()
        
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Validation')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(train_accs, label='Train')
        plt.plot(val_accs, label='Validation')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(lrs)
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        
        plt.tight_layout()
        plt.show()
        
        if val_acc > best_acc + 0.001:
            best_acc = val_acc
            early_stop_counter = 0
            torch.save(model.state_dict(), 'best_ocr_model.pth')
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f'Training complete. Best validation accuracy: {best_acc:.2f}%')
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'lrs': lrs
    }

# Unfreeze all layers for final training
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.AdamW(model.parameters(), lr=0.0001)

# Start training
metrics = train_model(
    model, 
    train_loader, 
    test_loader, 
    criterion, 
    optimizer, 
    scheduler, 
    num_epochs
)