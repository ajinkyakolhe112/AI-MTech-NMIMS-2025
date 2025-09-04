"""
Lecture 1: Neural Network Fundamentals + MNIST Digits
A simple, refactored implementation using Hugging Face datasets and Accelerate.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics
from datasets import load_dataset
from torchvision import transforms
from accelerate import Accelerator

def load_mnist_data():
    """Load MNIST dataset from Hugging Face"""
    # Load the dataset from the Hugging Face Hub
    dataset = load_dataset("mnist")

    # Define a transformation to convert images to PyTorch tensors
    transform = transforms.Compose([transforms.ToTensor()])

    def apply_transforms(examples):
        # Apply the tensor conversion
        for img in examples['image']:
            img = img.convert("RGB")
            img = transform(img)
            examples['pixel_values'].append(img)
        
        return examples

    # Apply the transformation to the entire dataset
    processed_dataset = dataset.with_transform(apply_transforms)
    
    # Create DataLoaders
    train_loader = DataLoader(processed_dataset['train'], batch_size=64, shuffle=True)
    test_loader = DataLoader(processed_dataset['test'], batch_size=64, shuffle=False)
    
    return train_loader, test_loader

class SimpleNeuralNet(nn.Module):
    """Simple MLP: 784 → 128 → 64 → 10"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

def train_model(model, train_loader, optimizer, accelerator, epochs=3):
    """Training loop simplified with Accelerate"""
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            # Data is automatically moved to the correct device by Accelerate
            x_batch = batch['pixel_values']
            y_batch = batch['label']
            
            # Forward pass
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        # accelerator.print only prints on the main process
        accelerator.print(f'Epoch {epoch+1}: Average Loss = {avg_loss:.4f}')

def test_model(model, test_loader):
    """Evaluate model accuracy (simplified for this example)"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            # Accelerate handles moving the batch to the correct device
            x_batch = batch['pixel_values']
            y_batch = batch['label']

            predictions = model(x_batch)
            
            # Note: In a true multi-GPU setup, this accuracy is only for one process.
            # For robust evaluation, one would use accelerator.gather_for_metrics().
            accuracy = torchmetrics.functional.accuracy(predictions, y_batch, task='multiclass', num_classes=10)
            correct += accuracy.item() * len(y_batch)
            total += len(y_batch)
    
    if total > 0:
        print(f'Test Accuracy: {(correct/total)*100:.2f}%')
    else:
        print('Test Accuracy: 0.00% (No data evaluated)')


if __name__ == "__main__":
    # Initialize Accelerator
    accelerator = Accelerator()

    # Load data
    train_loader, test_loader = load_mnist_data()
    
    # Create model and optimizer
    model = SimpleNeuralNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Prepare everything with Accelerate
    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )

    accelerator.print(f'Model has {sum(p.numel() for p in model.parameters())} parameters')
    
    # Train and evaluate the model
    train_model(model, train_loader, optimizer, accelerator)
    test_model(model, test_loader)
