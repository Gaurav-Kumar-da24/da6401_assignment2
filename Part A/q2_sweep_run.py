#inding Best parameters using Hyperparameter Tuning
#Importing libraries
import numpy as np
import os
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer  
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random
from collections import defaultdict

#Importing custom modules 

from q1_CustomCNN import CustomCNN
from q2_iNaturalist_dataset_preprocess import prepare_inaturalist_data


# Dictionary of activation functions for flexibility
activation_functions = {
    'relu': nn.ReLU(),
    'gelu': nn.GELU(),
    'silu': nn.SiLU(),
    'mish': nn.Mish(),
}

#sweep configuration
sweep_config = {
    'method': 'bayes',  # Bayesian optimization for efficient search
    'metric': {
        'name': 'val_acc',
        'goal': 'maximize'
    },
    'parameters': {
        'image_size': {'value': 224},  # Fixed image size
        'batch_size': {'values': [32, 64]},
        'num_filters': {'values': [
            [32, 32, 32, 32, 32],     # Same in all layers
            [32, 64, 128, 256, 512],  # Doubling in each subsequent layer
            [128, 64, 32, 16, 8],    #  halving in each subsequent layer
            [64, 128, 256, 128, 64]   # Increasing then decreasing
        ]},
        'filter_size': {'value': 3},
        'activation': {'values': ['relu', 'gelu', 'silu', 'mish']},
        'dense_neurons': {'values': [128, 256, 512, 1024]},
        'dropout_rate': {'values': [0.2, 0.3]},
        'batch_norm': {'values': [True, False]},
        'data_augment': {'values': [True, False]},
        'learning_rate': {'values': [0.001, 0.0005, 0.0001]},
        'epochs': {'value': 10}  # Fixed number of epochs
    }
}

#train model
def train_model(config=None):
    """
    Train the model with given hyperparameters and log to wandb.
    """
    wandb.init(config=config)
    config = wandb.config

    # Create a descriptive run name based on hyperparameters
    run_name = f"bs_{config.batch_size}_act_{config.activation}_dense_{config.dense_neurons}_drop_{config.dropout_rate}_batchnorm_{config.batch_norm}"
    
    # Set the name in wandb
    wandb.run.name = run_name
    wandb.run.save()  # Save the run name early

    # Prepare data
    train_loader, val_loader, test_loader, num_classes, class_names = prepare_inaturalist_data(
        data_dir='/kaggle/input/dl-a2-dataset/inaturalist_12K',
        config=config
    )
    
    # Create model with current hyperparameters
    model = CustomCNN(
        input_shape=(3, config.image_size, config.image_size),
        num_classes=num_classes,
        num_filters=config.num_filters,
        filter_size=config.filter_size,
        activation=config.activation,
        dense_neurons=config.dense_neurons,
        dropout_rate=config.dropout_rate,
        batch_norm=config.batch_norm,
        data_augment=config.data_augment,
        learning_rate=config.learning_rate
    )
    
    # Set up wandb logger and model checkpoint
    wandb_logger = WandbLogger(log_model=True)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./checkpoints/',
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        deterministic=True,  # For reproducibility
        precision="16-mixed"
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader) # I doing training and validation here for hyperparameter tuning
    
    # Test the model
    trainer.test(model, test_loader) # I doing testing here to get the test accuracy but not required in Q


# Run the sweep
sweep_id = wandb.sweep(sweep_config, project='DL_A2_PARTA_CNN_sweep')
wandb.agent(sweep_id, train_model, count=50)  
