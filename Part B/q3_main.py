#PART B
#Q3 main function for fine-tuning GoogLeNet on iNaturalist dataset
import torch
import wandb

# Import custom modules
from q3_iNaturalist_dataset_preprocess import  prepare_inaturalist_data
from q1_GoogLeNet_pretrained import setup_googlenet_model, freeze_layers
from q3_fine_tune_GoogLeNet import finetune_model, evaluate_model

class Config:
    """Configuration class for model training"""
    def __init__(self):
        # Data parameters
        self.batch_size = 32
        self.image_size = 224  # GoogLeNet expects 224x224 images
        self.data_augment = True
        
        # Training parameters
        self.num_epochs = 15
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        
        # Project name for wandb
        self.project_name = "DL_A2_PARTB_PretrainedGoogLeNet"

def main():
    """Main function for fine-tuning GoogLeNet on iNaturalist dataset"""
    wandb.login()
    
    # GPU use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create configuration
    config = Config()
    
    # dataset direct location i put in kaggle input directory
    data_dir = "/kaggle/input/dl-a2-dataset/inaturalist_12K"
    
    # dataset prrprocessing and loading
    train_loader, val_loader, test_loader, num_classes, class_names = prepare_inaturalist_data(
        data_dir, config, val_split=0.2
    )
    
    print(f"Dataset loaded with {num_classes} classes: {class_names}")
    
    # Loading the pre-trained GoogLeNet model and  for iNaturalist dataset 
    googlenet_model = setup_googlenet_model(num_classes=10)
    
    # i Will use 'all-but-last' freezing strategy
    googlenet_model = freeze_layers(googlenet_model, strategy='all_but_last')
    
    # Fine-tune the model
    fine_tuned_model, history = finetune_model(
        googlenet_model, 
        train_loader,
        val_loader,
        config,
        device
    )
    
    # Evaluate on test set
    test_acc = evaluate_model(fine_tuned_model, test_loader, device)
    print(f"\nTest Accuracy with fine-tuned model (all-but-last): {test_acc:.2f}%")

if __name__ == "__main__":
    main()