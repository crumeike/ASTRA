#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data handling utilities for post-tornado damage recognition experiments.
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class TornadoDamageDataset(Dataset):
    """Custom Dataset for loading tornado damage images."""
    
    def __init__(self, root_dir, transform=None, split='train'):
        """
        Args:
            root_dir (str): Root directory containing data folders
            transform (callable, optional): Transform to be applied to images
            split (str): Dataset split ('train', 'valid', or 'test')
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.split = split
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(class_dir, img_name), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_transforms(input_size, data_augmentation='basic'):
    """
    Get data transforms for training and validation.
    
    Args:
        input_size (int): Input image size
        data_augmentation (str): Augmentation strategy ('none', 'basic', 'standard', 'advanced')
    
    Returns:
        dict: Dictionary containing train and validation transforms
    """
    # Basic normalization for validation
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Train transformations with different augmentation levels
    if data_augmentation == 'none':
        train_transform = val_transform
    
    elif data_augmentation == 'basic':
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    elif data_augmentation == 'standard':
        train_transform = transforms.Compose([
            transforms.Resize((input_size + 32, input_size + 32)),
            transforms.RandomCrop((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    elif data_augmentation == 'advanced':
        train_transform = transforms.Compose([
            transforms.Resize((input_size + 32, input_size + 32)),
            transforms.RandomCrop((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomCrop((input_size, input_size), padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.4, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=True)
        ])
    
    else:
        raise ValueError(f"Unknown data augmentation type: {data_augmentation}")
    
    return {
        'train': train_transform,
        'valid': val_transform
    }


def get_data_loaders(data_dir, input_size, batch_size, data_augmentation='basic', num_workers=0):
    """
    Create data loaders for training and validation.
    
    Args:
        data_dir (str): Directory containing the data
        input_size (int): Input image size
        batch_size (int): Batch size
        data_augmentation (str): Augmentation strategy
        num_workers (int): Number of worker threads for data loading
    
    Returns:
        tuple: (train_loader, val_loader, class_names)
    """
    # Get transforms
    transforms_dict = get_data_transforms(input_size, data_augmentation)
    
    # Create datasets
    train_dataset = TornadoDamageDataset(
        root_dir=data_dir, 
        transform=transforms_dict['train'], 
        split='train'
    )
    
    val_dataset = TornadoDamageDataset(
        root_dir=data_dir, 
        transform=transforms_dict['valid'], 
        split='valid'
    )

    test_dataset = TornadoDamageDataset(
        root_dir=data_dir, 
        transform=transforms_dict['valid'], 
        split='test'
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.classes