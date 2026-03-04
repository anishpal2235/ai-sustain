"""
Q3: Model Training & Supervised Evaluation
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm


class LandUseDataset(Dataset):
    """Custom Dataset for satellite images and land-use labels."""

    def __init__(self, dataframe, label_to_idx, transform=None):
        """
        Args:
            dataframe: DataFrame with 'image_path' and 'label' columns.
            label_to_idx: dict mapping label string -> integer index.
            transform: Image transformations to apply.
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label_idx = self.label_to_idx[row['label']]
        return image, label_idx


def get_data_transforms():
    """Define data transformations for training and testing."""
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform


class CustomCNN(nn.Module):
    """Simple custom CNN for land-use classification."""
    
    def __init__(self, num_classes=5):
        """
        Args:
            num_classes: Number of output classes
        """
        super(CustomCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_model(num_classes, model_type='custom', pretrained=True):
    """
    Create a model for land-use classification.

    Supported model_type values:
        custom, resnet18, resnet34, resnet50,
        efficientnet_b0, mobilenet_v3_small,
        convnext_tiny, vit_b_16
    """
    weights = 'IMAGENET1K_V1' if pretrained else None

    if model_type == 'custom':
        return CustomCNN(num_classes=num_classes)

    elif model_type == 'resnet18':
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_type == 'resnet34':
        model = models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_type == 'resnet50':
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_type == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_type == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    elif model_type == 'convnext_tiny':
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    elif model_type == 'vit_b_16':
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    else:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Choose from: custom, resnet18, resnet34, resnet50, "
            f"efficientnet_b0, mobilenet_v3_small, convnext_tiny, vit_b_16"
        )

    return model


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train model for one epoch.
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate_model(model, dataloader, device):
    """
    Evaluate model on a dataloader.  Returns (pred_indices, true_indices).
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def train_model(train_loader, test_loader, num_classes, num_epochs=20, 
                model_type='custom', device='cuda' if torch.cuda.is_available() else 'cpu',
                learning_rate=0.001):
    """
    Train and evaluate CNN model for land-use classification.
    
    Args:
        train_loader: Training dataloader
        test_loader: Test dataloader
        num_classes: Number of classes
        num_epochs: Number of training epochs
        model_type: Model architecture type
        device: Device to use (cuda/cpu)
        learning_rate: Learning rate for optimizer
        
    Returns:
        Trained model, predictions, metrics
    """
    # Create model
    model = create_model(num_classes, model_type=model_type, pretrained=(model_type != 'custom'))
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    history = {'train_loss': [], 'train_acc': [], 'epoch': []}
    
    print(f"Training on {device}")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['epoch'].append(epoch + 1)
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    
    # Evaluate
    test_preds, test_labels = evaluate_model(model, test_loader, device)
    
    return model, test_preds, test_labels, history


def compute_metrics(y_true, y_pred, class_names):
    """
    Compute accuracy, F1-score, and other metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        
    Returns:
        Dictionary with metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro
    }
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, 
                                  output_dict=True, zero_division=0)
    
    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    
    return metrics, report


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Path to save plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix - Land-Use Classification', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return cm


def plot_training_history(history, save_path=None):
    """Plot training loss history."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['train_loss'], marker='o', linewidth=2)
    plt.title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
