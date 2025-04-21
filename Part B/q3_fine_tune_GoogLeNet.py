#PART B - GoogLeNet Fine-tuning on iNaturalist dataset
# q3_fine_tune_GoogLeNet.py

#I am  fine-tuning with . freezing all layers except the last layer ()'all-but-last')

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

def finetune_model(model, train_loader, val_loader, config, device):
    """
    Fine-tune the model using the 'all-but-last' freezing strategy.
    Args:
        model: The model to fine-tune
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Configuration with training parameters
        device: Device to train on (cuda/cpu)
    Returns:
        model: The fine-tuned model
        history: Training and validation metrics
    """
    # Initialize wandb
    run = wandb.init(project=config.project_name, name="GoogLeNet-all_but_last", config=vars(config))
    
    # Get parameters from config
    num_epochs = config.num_epochs
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay
    
    # Move model to the appropriate device
    model = model.to(device)
    
    # Log which parameters are being trained
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params} ({trainable_params/total_params:.2%})")
    wandb.log({"total_params": total_params, "trainable_params": trainable_params})
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Only optimize parameters that require gradients
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)
    
    # Training loop
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass - handle auxiliary outputs if they exist
            if model.training and hasattr(model, 'aux_logits') and model.aux_logits:
                outputs, aux_outputs = model(inputs)
                loss = criterion(outputs, labels)
                if aux_outputs is not None:
                    loss += 0.3 * criterion(aux_outputs, labels)
            else:
                outputs = model(inputs)
                # Handle case where model returns a tuple even when aux_logits is False
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                if isinstance(outputs, tuple):  # Handle auxiliary outputs
                    outputs = outputs[0]
                
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"googlenet_all_but_last_best.pth")
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
        
        # Log metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # Store in history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
    
    # Log final model metrics
    wandb.log({
        "final_val_acc": val_acc,
        "best_val_acc": best_val_acc
    })
    
    # Close wandb run
    wandb.finish()
    return model, history

def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test data
    
    Args:
        model: The trained model
        test_loader: DataLoader for test data
        device: Device to evaluate on (cuda/cpu)
        
    Returns:
        test_acc: Test accuracy
    """
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            _, predicted = torch.max(outputs.data, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_acc = 100 * test_correct / test_total
    
    wandb.init(project="DL_A2_PARTB_PretrainedGoogLeNet", name="test_evaluation")
    wandb.log({"test_accuracy": test_acc})
    wandb.finish()
    
    return test_acc