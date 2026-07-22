"""
04. PyTorch dataset class for TINTO images

Classes:
    1. TINTOImageDataset

Functions:
    1. load_fold_images
    2. get_default_transform
    3. get_simple_transform
    4. visualize_tinto_images
"""

import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# PyTorch Dataset for loading TINTO-generated images from disk
class TINTOImageDataset(Dataset):
    """
    PyTorch Dataset class for TINTO generated images.
    
    Args:
        image_paths: List of image file paths 
        labels: List of labels 
        transform: Image transformations 
    """

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Read image in RGB format
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR to RGB for pretrained models
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Load training and validation images from fold data into PyTorch Datasets
def load_fold_images(fold_data, transform=None):
    """
    Load training and validation images from fold data.
    
    Args:
        fold_data: Dictionary containing image paths and labels 
        transform: Image transformations 
    
    Returns:
        train_dataset: Training dataset 
        val_dataset: Validation dataset 
    """

    train_paths = fold_data['train']['images']['images'].tolist()
    train_labels = fold_data['train']['labels']

    val_paths = fold_data['val']['images']['images'].tolist()
    val_labels = fold_data['val']['labels']

    train_dataset = TINTOImageDataset(train_paths, train_labels, transform)
    val_dataset = TINTOImageDataset(val_paths, val_labels, transform)

    return train_dataset, val_dataset


# Get default ImageNet normalization transform chain
def get_default_transform(inputsize=224):
    """
    Get default ImageNet normalization transform.
    This is the standard transform for models pretrained on ImageNet.
    
    Args:
        input_size: Target image size 

    Returns:
        torchvision.transforms.Compose: Transform pipeline 
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((inputsize, inputsize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def get_simple_transform(input_size=224):
    """
    Get simple normalization transform for grayscale-like images.
    
    This is recommended for TINTO images which are grayscale-style.
    
    Args:
        input_size: Target image size 
    
    Returns:
        torchvision.transforms.Compose: Transform pipeline 
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                           std=[0.5, 0.5, 0.5])
    ])

def visualize_tinto_images(fold_data, 
                           fold_idx=0, 
                           num_samples=5):
    """
    Visualize TINTO generated images

    Args:
        fold_data: Fold data
        fold_idx: Fold index
        num_samples: Number of samples to display
    """

    # Get training images and labels
    train_paths = fold_data['train']['images']['images'].tolist()
    train_labels = fold_data['train']['labels']

    # Create subplot for visualizetion
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))

    # Display each sample
    for i in range(min(num_samples, len(train_paths))):
        # Read and convert image
        img = cv2.imread(train_paths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Display image with its class label
        axes[i].imshow(img)
        axes[i].set_title(f'Class: {train_labels[i]}')
        axes[i].axis('off')

    plt.suptitle(f'TINTO Images - Fold {fold_idx} (Train Set)')
    plt.tight_layout()
    plt.show()