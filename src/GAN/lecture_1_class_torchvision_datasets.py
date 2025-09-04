"""
Lecture 1: Neural Network Fundamentals + MNIST Digits
A simple, refactored implementation using torchvision datasets and Accelerate.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchmetrics
from accelerate import Accelerator

def load_mnist_data():
    """Load and preprocess MNIST dataset"""
    # Download the Image Dataset
    # And Convert image to numerical value tensors
    transform  = transforms.ToTensor()
    train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_data  = datasets.MNIST('data', train=False, transform=transform)

    # Create DataLoader function for efficient data-reading
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_data, batch_size=64, shuffle=False)

    return train_loader, test_loader

class SimpleNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Network architecture: 784 -> 128 -> 64 -> 10
        self.flatten_image = nn.Flatten()               # Convert 28x28 image to 784 pixels
        self.hidden_layer_1 = nn.Linear(784, 128)       # First hidden layer
        self.relu_1 = nn.ReLU()                         # Activation function
        self.hidden_layer_2 = nn.Linear(128, 64)        # Second hidden layer
        self.relu_2 = nn.ReLU()                         # Activation function
        self.output_layer = nn.Linear(64, 10)           # Output layer (10 digits)

        print("Created neural network: 784 -> 128 -> 64 -> 10")

    def forward(self, x):
        """Forward pass through the network"""
        x = self.flatten_image(x)      # Flatten image
        x = self.relu_1(self.hidden_layer_1(x))  # First layer + activation
        x = self.relu_2(self.hidden_layer_2(x))  # Second layer + activation
        x = self.output_layer(x)       # Final output (no activation)
        return x


def train_model(model, train_loader, optimizer, accelerator, epochs=3):
    """Training loop with accelerate"""
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in train_loader:

            # Forward pass
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)

            # Backward pass
            optimizer.zero_grad()  # Clear gradients
            accelerator.backward(loss)  # Use accelerator for backward pass
            optimizer.step()       # Update weights

            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        accelerator.print(f'Epoch {epoch+1}: Average Loss = {total_loss/len(train_loader):.4f}')

def test_model(model, test_loader):
    """Test the recently trained model accuracy"""
    model.eval()
    correct = 0
    total = 0

    # Set Automatic gradient calculation OFF
    torch.set_grad_enabled(False)
    for x_batch, y_batch in test_loader:
        predictions = model(x_batch)
        accuracy = torchmetrics.functional.accuracy(predictions, y_batch, task='multiclass', num_classes=10)
        correct += accuracy * len(y_batch)
        total += len(y_batch)
    # Set Automatic gradient calculation Back On
    torch.set_grad_enabled(True)

    print(f'Test Accuracy: {(correct/total)*100:.2f}%')

if __name__ == "__main__":
    # Initialize Accelerator
    accelerator = Accelerator()

    # Step 1: Load data
    train_loader, test_loader = load_mnist_data()

    # Step 2: Create model and optimizer
    model = SimpleNeuralNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Prepare everything with Accelerate
    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )

    accelerator.print(f'Model has {sum(p.numel() for p in model.parameters())} parameters')

    # Step 3: Train the model
    train_model(model, train_loader, optimizer, accelerator)

    # Step 4: Test the model
    test_model(model, test_loader)


## Future Improvements
# TODO:ajinkyak: Simple Trainer Function, custom mix of features inspired by lightning. Shift divice management to accelerate
# TODO:ajinkyak: Flag: Overfit one batch.