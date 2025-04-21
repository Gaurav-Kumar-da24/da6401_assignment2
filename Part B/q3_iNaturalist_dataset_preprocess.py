#PART B
# q3_iNaturalist_dataset_preprocess.py
# Dataset Preprocessing: 
# 1. Spliting train data in train and validation dataset
# 2. Data augumentation on train dataset
# 3. Creating dataloader for train and validation dataset

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random
from collections import defaultdict

def stratified_split(dataset, val_split=0.2, seed=42):
    """
    Splits dataset indices into stratified train and validation subsets.
    :param dataset: Torch Dataset object (must have `targets` attribute).
    :param val_split: Fraction of data to use for validation.
    :param seed: Random seed for reproducibility.
    :return: train_indices, val_indices
    """
    targets = dataset.targets
    label_to_indices = defaultdict(list)

    # Group indices by class
    for idx, label in enumerate(targets):
        label_to_indices[label].append(idx)

    train_indices = []
    val_indices = []

    random.seed(seed)

    for label, indices in label_to_indices.items():
        n_total = len(indices)
        n_val = int(n_total * val_split)

        # Shuffle indices for this class
        random.shuffle(indices)

        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    # Shuffle final lists (optional, especially for training)
    random.shuffle(train_indices)
    random.shuffle(val_indices)

    return train_indices, val_indices



def prepare_inaturalist_data(data_dir, config, val_split=0.2):
    """
    Prepare the iNaturalist dataset with a custom stratified split.
    Augmentations are applied based on config.data_augmentation flag.
    Parameters:
    - data_dir: Path to data
    - config: wandb.config (expects config.batch_size, config.image_size, config.data_augmentation)
    - val_split: Fraction of training data used for validation
    """
    image_size = config.image_size
    batch_size = config.batch_size
    
    # Normal Transformation applied on train,valid and test all dataset
    base_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Data Augmentation transformation  will be applied on train dataset
    augment_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load full train dataset
    full_train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'))

    # Perform stratified split on complete train dataset to split in train and valid
    train_indices, val_indices = stratified_split(full_train_dataset, val_split=val_split, seed=42)

    # Create Subsets
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)

    # Assign  transforms conditionally
    if config.data_augment:
        train_dataset.dataset.transform = augment_transform
    else:
        train_dataset.dataset.transform = base_transform

    val_dataset.dataset.transform = base_transform

    
    # Train and Validation Dataset Loading
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Load the test dataset
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=base_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Class info
    num_classes = len(full_train_dataset.classes)
    class_names = full_train_dataset.classes

    return train_loader, val_loader, test_loader, num_classes, class_names