# DA6401 assignment 2
##DA24M006 GAURAV KUMAR M.Tech DSAI
### CNN Model for iNaturalist Dataset Classification

This repository contains the implementation of a Custom CNN model for image classification on the iNaturalist dataset. The project is structured to train a CNN model from scratch, perform hyperparameter tuning, and evaluate the best model on test data.

## File Structure

```
.
├── README.md                      # This file
├── PARTA/                         # Part A implementation
│   ├── q1_CustomCNN.py            # Custom CNN model implementation
│   ├── q2_iNaturalist_dataset_preprocess.py # Dataset preprocessing utilities
│   ├── q2_sweep_run.py            # Hyperparameter tuning with wandb
│   └── q4_best_model_test_plot.py # Evaluation of best model on test data and plotinng prediction plots.
├── PARTB/                         # Part B implementation (fine-tuning pre-trained models)
    ├──       
└── inaturalist_12K/               # Dataset directory
    ├── train/                     # Training images organized by class folders
    └── val/                       # Validation/test images organized by class folders
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Lightning
- Weights & Biases (wandb)
- torchvision
- matplotlib
- numpy

## Dataset Structure

The iNaturalist dataset should i organized as follows: (for kaggle i put inaturalist_12K in input folder)
```
inaturalist_12K/
├── train/
│   ├── Amphibia/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── Animalia/
│   │   └── ...
│   └── ...
└── val/
    ├── Amphibia/
    │   └── ...
    └── ...
```

## Implementation Details

### Part A: Training CNN from Scratch

### Part A Flow:

1. **Model Definition**: `q1_CustomCNN.py` defines the neural network architecture
2. **Data Preparation**: `q2_iNaturalist_dataset_preprocess.py` handles dataset splitting and preprocessing
3. **Model Training and Tuning**: `q2_sweep_run.py` trains multiple models with different hyperparameters
4. **Model Evaluation**: `q4_best_model_test_plot.py` evaluates the best model and visualizes predicted images


#### 1. Custom CNN Model (`PARTA/q1_CustomCNN.py`)

The q1_CustomCNN.py implements a flexible CustomCNN class with:

- 5 convolutional layers with customizable filters and sizes
- Activation function after each convolutional layer
- Max pooling after each activation
- Batch normalization (optional)
- A dense layer followed by dropout
- Output layer with 10 neurons (one for each class)

I implemented CNN model using PyTorch Lightning to facilitate training, validation, and testing.

#### 2. Dataset Preprocessing (`PARTA/q2_iNaturalist_dataset_preprocess.py`)

The q2_iNaturalist_dataset_preprocess.py provides utilities for:

- Stratified splitting of the training data into training and validation sets
- 20% of the training data is set aside for validation during hyperparameter tuning
- Data augmentation for the training set based on the hyperparameter configuration
- Making image size to 224x224 (resizing)
- Creating DataLoaders for train, validation, and test sets
- Normalizing images using ImageNet mean and standard deviation

#### 3. Hyperparameter Tuning (`PARTA/q2_sweep_run.py`)

I performed sweep run with Bayesian hyperparameter optimization using Weights & Biases to find the best model configuration. 
Parameters tuned include:

- Number of filters in each convolutional layer
- Filter organization strategy (same, doubling, halving, etc.)
- Activation function
- Number of neurons in the dense layer
- Dropout rate
- Batch normalization
- Data augmentation
- Learning rate
### sweep configuration
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

#### 4. Best Model Evaluation and Prection Plot as 10x3(`PARTA/q4_best_model_test_plot.py`)

 q4_best_model_test_plot.py script:

- Retrieves the best model from the W&B sweep
- Evaluates it on the test dataset
- Creates a visualization grid (10×3) showing sample predictions from each class
- Logs results and visualizations to W&B
- The test set (in the 'val' folder) is used only for final evaluation and not for hyperparameter tuning

## Usage

### 1. Train and Tune the Custom CNN Model (Part A)

```bash
cd PARTA
python q2_sweep_run.py
```

This will start the hyperparameter tuning process using W&B sweeps. The script will automatically:
- Create a W&B project
- Initialize a sweep with the defined configuration
- Run multiple experiments with different hyperparameter combinations
- Track and compare performance metrics
- The hyperparameter sweep is configured to use Bayesian optimization for efficient exploration of the parameter space
- The model is trained with mixed precision (FP16) for faster training
### 2. Evaluate the Best Custom CNN Model

```bash
cd PARTA
python q4_best_model_test_plot.py
```

This will:
- Retrieve the best model from the W&B sweep
- Evaluate it on the test dataset
- Generate and display a grid of predictions
- Log evaluation results to W&B


## Notes


- 



### Part B: Fine-tuning Pre-trained Models I used GoogLeNet Pretrained Model

The Part B implementation focuses on fine-tuning pre-trained models on GoogLeNet on the iNaturalist dataset. Code for this section is located in the PARTB directory.
