# I am using GoogLeNet as Pretrained model with ImageNet weights
# Then I am modifying the final layers for the iNaturalist dataset

import torch
import torch.nn as nn
from torchvision.models import googlenet, GoogLeNet_Weights

def setup_googlenet_model(num_classes=10):
    """Sets up GoogLeNet model for iNaturalist classification."""
    #Step 1.Load pre-trained GoogLeNet with ImageNet weights
    # weights=GoogLeNet_Weights.IMAGENET1K_V1
    model = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    
    #Step 2. GoogLeNet has 1000 output classes (for ImageNet) We need to modify it for our classes in iNaturalist
    # Modify the final classifier layer 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    
    #Step 3.Adapt auxiliary classifiers for training (GoogLeNet specific)
    if hasattr(model, 'aux_logits') and model.aux_logits:
        # Check and modify aux1 if it exists
        if hasattr(model, 'aux1') and model.aux1 is not None:
            if hasattr(model.aux1, 'fc'):
                in_features = model.aux1.fc.in_features
                model.aux1.fc = nn.Linear(in_features, num_classes)
            elif hasattr(model.aux1, 'fc2'):
                in_features = model.aux1.fc2.in_features
                model.aux1.fc2 = nn.Linear(in_features, num_classes)
        
        # Check and modify aux2 if it exists
        if hasattr(model, 'aux2') and model.aux2 is not None:
            if hasattr(model.aux2, 'fc'):
                in_features = model.aux2.fc.in_features
                model.aux2.fc = nn.Linear(in_features, num_classes)
            elif hasattr(model.aux2, 'fc2'):
                in_features = model.aux2.fc2.in_features
                model.aux2.fc2 = nn.Linear(in_features, num_classes)
    
    # Handle newer PyTorch versions where InceptionAux is used differently
    if hasattr(model, 'AuxLogits') and model.AuxLogits is not None:
        in_features = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(in_features, num_classes)
    
    return model
