
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from q1_CustomCNN import CustomCNN

def get_best_model(entity, project, sweep_id):
    """Retrieve best model from W&B sweep"""
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    runs = sorted(sweep.runs, key=lambda run: run.summary.get('val_acc', 0), reverse=True)
    
    if not runs:
        raise ValueError("No runs found in the sweep")
    
    best_run = runs[0]
    print(f"Best Run: {best_run.name}\nValidation Accuracy: {best_run.summary['val_acc']:.2%}")

    # Find and download model artifact
    for artifact in best_run.logged_artifacts():
        if artifact.type == "model":
            artifact_dir = artifact.download()
            for f in Path(artifact_dir).iterdir():
                if f.suffix == ".ckpt":
                    return str(f), dict(best_run.config)
    
    raise FileNotFoundError("No model checkpoint found")

def prepare_test_data(config):
    """Prepare test data loader from specified directory"""
    test_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Load test dataset from location:/kaggle/input/dl-a2-dataset/inaturalist_12K/val
    test_dataset = datasets.ImageFolder(
        root='/kaggle/input/dl-a2-dataset/inaturalist_12K/val',
        transform=test_transform
    )
    
    return DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    ), test_dataset.classes

def visualize_predictions(model, test_loader, class_names):
    """Generate 10x3 grid of test predictions"""
    model.eval()
    device = next(model.parameters()).device
    
    # Collect 3 samples per class
    samples = {i: [] for i in range(len(class_names))}
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # Denormalize images
            images = images.cpu().numpy()
            images = images.transpose(0, 2, 3, 1)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            images = std * images + mean
            images = np.clip(images, 0, 1)
            
            for img, true, pred in zip(images, labels, preds.cpu()):
                if len(samples[true.item()]) < 3:
                    samples[true.item()].append((img, true.item(), pred.item()))
                
                if all(len(v) >= 3 for v in samples.values()):
                    break

    # Create plot
    fig, axes = plt.subplots(10, 3, figsize=(20, 30))
    for class_idx in range(10):
        for sample_idx in range(3):
            img, true, pred = samples[class_idx][sample_idx]
            ax = axes[class_idx, sample_idx]
            ax.imshow(img)
            ax.axis('off')
            
            true_name = class_names[true]
            pred_name = class_names[pred]
            color = 'green' if true == pred else 'red'
            ax.set_title(f"True: {true_name}\nPred: {pred_name}", color=color, fontsize=8)
    
    plt.tight_layout()
    return fig

def evaluate_model(checkpoint_path, config):
    """Evaluate model on test set and log results"""
    # Initialize W&B
    wandb.init(project="DL_A2_PARTA_Q4", entity="da24m006-iit-madras")
    wandb.run.name="Q4_test_acc_predict_grid_plot"
    wandb.run.save()
    # Load model
    model = CustomCNN.load_from_checkpoint(
        checkpoint_path,
        input_shape=(3, config['image_size'], config['image_size']),
        num_classes=10
    )
    
    # Prepare test data
    test_loader, class_names = prepare_test_data(config)
    
    # Evaluate
    trainer = pl.Trainer(
        accelerator='auto',
        logger=False,
        enable_checkpointing=False
    )
    
    # Test accuracy
    test_results = trainer.test(model, test_loader)
    test_acc = test_results[0]['test_acc']
    print(f"Test Accuracy: {test_acc:.2%}")
    
    # Generate predictions grid
    fig = visualize_predictions(model, test_loader, class_names)


    figures = visualize_predictions(model, test_loader, class_names)

    for class_name, fig in figures:
        wandb.log({f"Predictions/{class_name}": wandb.Image(fig)})
    
    plt.close()

if __name__ == "__main__":
    # Configuration
    ENTITY = "da24m006-iit-madras"
    PROJECT = "DL_A2_PARTA_CNN_sweep"
    SWEEP_ID = "cfxm050a"  # sweep ID from wandb sweep run    
    # Get best model
    checkpoint_path, config = get_best_model(ENTITY, PROJECT, SWEEP_ID)
    
    # Evaluate and visualize predictions
    # Test accuracy and Prediction grid plot 10x3
    evaluate_model(checkpoint_path, config)
    wandb.finish()
