"""
07.Training and evaluation functions for transfer learning

Functions:
    1. get_fold_dataloaders
    2. train_epoch
    3. evaluate_epoch
    4. train_transfer_learning
"""

import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm

from .config import Config
from .dataset import load_fold_images, get_default_transform, get_simple_transform
from .data_loader import create_5fold_tinto_images
from .models import get_model, get_model_config


# Dataloader functions
def get_fold_dataloaders(X, y, 
                         fold_idx=0, 
                         batch_size=16, 
                         method_name='tinto',
                         num_workers=0, 
                         feature_names=None,
                         use_simple_norm=False,
                         model_name='efficientnet_v2_m'):
    """
    Get DataLoader for a specific fold.
    
    Args:
        X: Feature data 
        y: Label data 
        fold_idx: Fold index (0-4)
        batch_size: Batch size
        method_name: TINTO method name 
        num_workers: Number of workers (0 for Windows) 
        feature_names: Feature names
        use_simple_norm: Use simple [-1,1] normalization 
    
    Returns:
        train_loader: Training DataLoader 
        val_loader: Validation DataLoader 
    """
    # Generate TINTO images for all folds (or load existing)
    tinto_data = create_5fold_tinto_images(
        X, y, method_name=method_name,
        feature_names=feature_names
    )

    # Extract the current fold's data from the 5-fold dictionary
    fold_data = tinto_data[f'fold_{fold_idx}']

    # Get input_size from model config
    model_config = get_model_config(model_name)
    input_size = model_config['input_size']

    if use_simple_norm:
        transform = get_simple_transform(input_size)
    else:
        transform = get_default_transform(input_size)

    train_dataset, val_dataset = load_fold_images(fold_data, transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        # pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        # pin_memory=True        
    )

    print(f"DataLoader for Fold {fold_idx} created successfully")
    print(f" - Training batches: {len(train_loader)}")
    print(f" - Validation batches: {len(val_loader)}")
    print(f" - Input size: {input_size}")
    print(f" - Normalization: {'Simple' if use_simple_norm else 'ImageNet'}")

    return train_loader, val_loader


# Training functions
def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model 
        dataloader: DataLoader 
        criterion: Loss function 
        optimizer: Optimizer 
        device: Device (cpu/cuda) 
    
    Returns:
        avg_loss: Average loss
        accuracy: Accuracy 
    """
    model.train()  # Enable training mode (Dropout active, BatchNorm tracking)
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        labels = labels.long()

        # Standard training step: forward → loss → backward → update
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(dataloader), correct / total


# Evaluate the model on the validation set
def evaluate_epoch(model, dataloader, criterion, device):
    """
    Evaluate one epoch. 
    Returns loss, accuracy, predictions, probabilities, true labels.
    """
    model.eval()  # Evaluation mode (Dropout disabled, BatchNorm frozen)
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():  # Disable gradient tracking for speed and memory
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.long()

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            # Convert logits to class probabilities for AUC-ROC computation
            probs = torch.softmax(outputs, dim=1)        # probabilities for AUC-ROC
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return (running_loss / len(dataloader), correct / total,
            all_preds, all_probs, all_labels)


# Main Training functions
def train_transfer_learning(X, y, fold_idx=0, epochs=50, method_name='tinto',
                            model_name='efficientnet_v2_m', save_model=False,
                            fine_tune_all=False, feature_names=None, device='cpu',
                            use_simple_norm=False):
    """
    Train a model using transfer learning on TINTO images.
    
    Args:
        X: Feature data 
        y: Label data 
        fold_idx: Fold index 
        epochs: Number of epochs 
        method_name: TINTO method name 
        model_name: Model name 
        save_model: Whether to save model weights 
        fine_tune_all: If True, fine-tune all layers
        feature_names: Feature names 
        device: Device (cpu/cuda) 
        use_simple_norm: Use simple normalization 
    
    Returns:
        history: Training history 
        model: Trained model
        best_val_acc: Best validation accuracy 
    """
    print(f"\n{'='*60}")
    print(f"Training: Fold {fold_idx}, Method: {method_name}, Model: {model_name}")
    print('='*60)

    # Get DataLoader
    train_loader, val_loader = get_fold_dataloaders(
        X, y, fold_idx, Config.BATCH_SIZE, method_name,
        num_workers=0, feature_names=feature_names,
        use_simple_norm=use_simple_norm,
        model_name=model_name
    )

    # Get model
    model = get_model(model_name, Config.NUM_CLASSES, device)

    # Setup optimizer with differential learning rates
    # Default: freeze backbone weights, train only the classifier head on top
    if fine_tune_all:
        # Fine-tune all layers: lower LR for backbone, higher LR for classifier
        for param in model.parameters():
            param.requires_grad = True

        backbone_params = []
        classifier_params = []
        for name, param in model.named_parameters():
            if 'classifier' in name or 'fc' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)

        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': Config.LEARNING_RATE / 10},  # Lower LR for backbone
            {'params': classifier_params, 'lr': Config.LEARNING_RATE}      # Higher LR for classifier
        ])
    else:
        # Train classifier head only, freeze all backbone weights
        for param in model.parameters():
            param.requires_grad = False

        # Use the correct classifier attribute name for this model
        # ResNeXt uses 'fc', all other models use 'classifier'
        classifier_attr = get_model_config(model_name)['classifier_name']
        classifier = getattr(model, classifier_attr)
        for param in classifier.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(classifier.parameters(), lr=Config.LEARNING_RATE)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Halve learning rate when validation accuracy stalls for 5 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    early_stop_patience = 10  # Stop if no improvement for 10 consecutive epochs

    # Train for fixed epochs, early stopping disabled for fair fold comparison
    for epoch in range(epochs):
        start_time = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, *_ = evaluate_epoch(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())  # True deep copy — safe against future mutations
        #     patience_counter = 0
        # else:
        #     patience_counter += 1

        scheduler.step(val_acc)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} [{epoch_time:.1f}s] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping intentionally disabled: all folds must run equal epochs
        # for a fair comparison across methods and models
        # # Early stopping
        # if patience_counter >= early_stop_patience:
        #     print(f"  Early stopping at epoch {epoch+1}")
        #     break

    # Restore the best model weights and compute final metrics
    if best_model_state:
        model.load_state_dict(best_model_state)

    # evaluate_epoch return: loss, accuracy, predictions, probabilities, true labels
    _, _, best_preds, best_probs, best_labels = evaluate_epoch(
        model, val_loader, criterion, device
    )

    # Compute F1 and AUC-ROC
    # F1 = 2 × (Precision × Recall) / (Precision + Recall)
    best_f1 = f1_score(best_labels, best_preds, average='weighted')

    try:
        # [tensor([0.9, 0.1]), tensor([0.2, 0.8]), ...]  -> array([[0.9, 0.1], [0.2, 0.8], ...])
        best_probs = np.array(best_probs)
        # Binary: use positive-class probability; Multi-class: one-vs-rest
        if best_probs.shape[1] == 2:
            best_auc = roc_auc_score(best_labels, best_probs[:, 1])
        else:
            best_auc = roc_auc_score(best_labels, best_probs, 
                                     multi_class='ovr', average='weighted')
    except ValueError:
        best_auc = float('nan')  # Some folds may have only 1 sample of a class

    print(f"\n Best: Acc={best_val_acc:.4f}  F1={best_f1:.4f}  AUC={best_auc:.4f}")

    # Optionally save model checkpoint to disk
    if save_model:
        model_path = Config.RESULTS_DIR / f'model_fold{fold_idx}_{method_name}_{model_name}.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,
            'best_val_acc': best_val_acc,
            'best_val_f1': best_f1,
            'best_val_auc': best_auc,
        }, model_path)
        print(f"Model saved to: {model_path}")

    return history, model, {
        'accuracy': best_val_acc,
        'f1_score': best_f1,
        'auc_roc': best_auc,
    }
