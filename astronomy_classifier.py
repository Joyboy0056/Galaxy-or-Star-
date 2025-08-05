import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from pathlib import Path
import time
import yaml

# ===============================
# GLOBAL SETUP
# ===============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_config(config_path="configs/config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# ===============================
# DATASET CLASS
# ===============================
class AstroDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Assume structure: root_dir/star/, root_dir/galaxy/
        classes = ['star', 'galaxy']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        for class_name in classes:
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.images.append(str(img_path))
                    self.labels.append(self.class_to_idx[class_name])
        
        print(f"Found {len(self.images)} images")
        print(f"Stars: {self.labels.count(0)}, Galaxies: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ===============================
# DATA TRANSFORMS
# ===============================

def get_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms

# ===============================
# MODEL DEFINITION
# ===============================

class AstroCNN(nn.Module):
    """
    A convolutional neural network for binary classification of astronomical images (e.g., stars vs galaxies),
    based on the ResNet-18 architecture with a customizable output layer.

    This model leverages transfer learning by optionally using pretrained weights from ImageNet. 
    The final fully connected layer is replaced with a Dropout + Linear combination to adapt 
    the model to a custom number of output classes.

    Args:
        num_classes (int): Number of output classes. Default is 2 (e.g., star, galaxy).
        pretrained (bool): If True, loads ResNet-18 with pretrained weights on ImageNet. Default is True.

    Attributes:
        backbone (nn.Module): The modified ResNet-18 model with a custom fully connected layer.

    Example:
        >>> model = AstronomyCNN(num_classes=2, pretrained=True)
        >>> output = model(images)  # where images is a batch of input tensors
    """

    def __init__(self, num_classes=2, pretrained=True):
        super(AstroCNN, self).__init__()
        
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Replace final layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.fc.in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# ===============================
# TRAINING FUNCTIONS
# ===============================

def train_model(model, train_loader, val_loader, num_epochs=25, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.cpu().numpy())
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
        
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)
        
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc.cpu().numpy())
        
        print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')
        
        # Save best model
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), 'best_astronomy_model.pth')
            print(f'New best model saved with val_acc: {best_val_acc:.4f}')
        
        scheduler.step()
        print()
    
    return train_losses, train_accs, val_losses, val_accs

# ===============================
# EVALUATION FUNCTIONS
# ===============================

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Classification report
    class_names = ['Star', 'Galaxy']
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return all_preds, all_labels

def plot_training_history(train_losses, train_accs, val_losses, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy plot
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


# ===============================
# MAIN EXECUTION
# ===============================

def main():
    # Load configuration
    config = load_config()
    
    # Set device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Using device: {device}")
    
    # Get transforms
    train_transforms, val_transforms = get_transforms()
    
    # Load datasets
    full_dataset = AstroDataset(config['data']['data_dir'], transform=train_transforms)
    
    # Split dataset
    train_size = int(config['data']['train_split'] * len(full_dataset))
    val_size = int(config['data']['val_split'] * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Apply different transforms to val and test
    val_dataset.dataset.transform = val_transforms
    test_dataset.dataset.transform = val_transforms
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Initialize model
    model = AstroCNN(num_classes=config['model']['num_classes'], pretrained=config['model']['pretrained'])
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    start_time = time.time()
    train_losses, train_accs, val_losses, val_accs = train_model(
        model, train_loader, val_loader, 
        config['training']['num_epochs'], 
        config['training']['learning_rate']
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time/60:.2f} minutes")
    
    # Plot training history
    plot_training_history(train_losses, train_accs, val_losses, val_accs)
    
    # Load best model for evaluation
    model.load_state_dict(torch.load('best_astronomy_model.pth'))
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_preds, test_labels = evaluate_model(model, test_loader)

if __name__ == "__main__":
    main()
