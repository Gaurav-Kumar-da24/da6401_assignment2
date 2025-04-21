#  q1_CustomCNN.py
"""
 Question 1 :
 Build a small CNN model consisting of 5 convolution layers.
 Each convolution layer would be followed by an activation and a
 max-pooling layer.
 After 5 such conv-activation-maxpool blocks, you should have
 one dense layer followed by the output layer containing 10
neurons (1 for each of the 10 classes). The input layer should be
 compatible with the images in the iNaturalist dataset dataset.
The code should be flexible such that the number of filters, size
 of filters, and activation function of the convolution layers and
 dense layers can be changed. You should also be able to change
 the number of neurons in the dense layer.
 """
import torch
import torch.nn as nn
 
 # Dictionary of activation functions for flexibility
activation_functions = {
    'relu': nn.ReLU(),
    'gelu': nn.GELU(),
    'silu': nn.SiLU(),
    'mish': nn.Mish(),
}
class CustomCNN(pl.LightningModule):
    def __init__(self, 
                 input_shape=(3, 224, 224),
                 num_classes=10,
                 num_filters=[32, 64, 128, 256, 512],
                 filter_size=3,
                 activation='relu',
                 dense_neurons=512,
                 dropout_rate=0.2,
                 batch_norm=True,
                 data_augment=True,
                 learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        
        # Store parameters
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.activation = activation_functions[activation]
        self.dense_neurons = dense_neurons
        self.dropout_rate = dropout_rate
        self.batch_norm =batch_norm
        self.data_augment=data_augment
        self.learning_rate = learning_rate
        
        # Build the network
        self.layers = nn.ModuleList()
        
        # Input channels for the first layer
        in_channels = input_shape[0]
        
        # Create 5 convolutional blocks
        for i in range(5):
            # Convolutional layer
            conv = nn.Conv2d(in_channels, num_filters[i], kernel_size=filter_size, padding=filter_size//2)
            self.layers.append(conv)
            
            # Batch normalization (optional)
            if batch_norm:
                self.layers.append(nn.BatchNorm2d(num_filters[i]))
            
            # Activation function
            self.layers.append(self.activation)
            
            # Max pooling
            self.layers.append(nn.MaxPool2d(2, 2))
            
            # Update input channels for next layer
            in_channels = num_filters[i]
        
        # Calculate size of the flattened feature map
        self.feature_size = self._get_feature_size()
        
        # Dense layer
        self.fc1 = nn.Linear(self.feature_size, dense_neurons)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output layer
        self.fc2 = nn.Linear(dense_neurons, num_classes)
    
    def _get_feature_size(self):
        # Determine output size after convolutions
        x = torch.zeros(1, *self.input_shape)
        for layer in self.layers:
            x = layer(x)
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        # Pass input through convolutional layers
        for layer in self.layers:
            x = layer(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Pass through dense layer
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.fc2(x)
        
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
