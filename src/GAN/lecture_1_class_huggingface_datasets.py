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

def load_mnist_classification_data():
    """Load MNIST dataset from Hugging Face"""
    # Load the dataset from the Hugging Face Hub
    mnist_dataset = load_dataset("mnist")

    # Define a transformation to convert images to PyTorch tensors
    image_to_tensor_transform = transforms.Compose([transforms.ToTensor()])

    def apply_transforms_to_batch(batch_examples):
        # Apply the tensor conversion
        for image in batch_examples['image']:
            image = image.convert("RGB")
            image_tensor = image_to_tensor_transform(image)
            batch_examples['pixel_values'].append(image_tensor)
        
        return batch_examples

    # Apply the transformation to the entire dataset
    processed_mnist_dataset = mnist_dataset.with_transform(apply_transforms_to_batch)
    
    # Create DataLoaders
    train_dataloader = DataLoader(processed_mnist_dataset['train'], batch_size=64, shuffle=True)
    test_dataloader = DataLoader(processed_mnist_dataset['test'], batch_size=64, shuffle=False)
    
    return train_dataloader, test_dataloader

class DigitClassificationNetwork(nn.Module):
    """Neural network for classifying MNIST digits: 784 → 128 → 64 → 10"""
    def __init__(self):
        super().__init__()
        self.classification_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        print("Created classification network: 784 -> 128 -> 64 -> 10")
    
    def forward(self, input_images):
        classification_logits = self.classification_layers(input_images)
        return classification_logits

def train_classification_model(model, train_dataloader, optimizer, accelerator, epochs=3):
    """Training loop simplified with Accelerate"""
    loss_function = nn.CrossEntropyLoss()
    
    model.train()
    for current_epoch in range(epochs):
        total_training_loss = 0
        for data_batch in train_dataloader:
            # Data is automatically moved to the correct device by Accelerate
            batch_images = data_batch['pixel_values']
            batch_labels = data_batch['label']
            
            # Forward pass
            predicted_logits = model(batch_images)
            training_loss = loss_function(predicted_logits, batch_labels)
            
            # Backward pass
            optimizer.zero_grad()
            accelerator.backward(training_loss)
            optimizer.step()
            
            total_training_loss += training_loss.item()
        
        average_training_loss = total_training_loss / len(train_dataloader)
        # accelerator.print only prints on the main process
        accelerator.print(f'Epoch {current_epoch+1}: Average Loss = {average_training_loss:.4f}')

def evaluate_classification_model(model, test_dataloader):
    """Evaluate model accuracy (simplified for this example)"""
    model.eval()
    total_correct_predictions = 0
    total_test_samples = 0
    
    with torch.no_grad():
        for test_batch in test_dataloader:
            # Accelerate handles moving the batch to the correct device
            test_images = test_batch['pixel_values']
            test_labels = test_batch['label']

            test_predictions_logits = model(test_images)
            
            # Note: In a true multi-GPU setup, this accuracy is only for one process.
            # For robust evaluation, one would use accelerator.gather_for_metrics().
            batch_accuracy = torchmetrics.functional.accuracy(test_predictions_logits, test_labels, task='multiclass', num_classes=10)
            total_correct_predictions += batch_accuracy.item() * len(test_labels)
            total_test_samples += len(test_labels)
    
    if total_test_samples > 0:
        final_test_accuracy = (total_correct_predictions/total_test_samples)*100
        print(f'Test Accuracy: {final_test_accuracy:.2f}%')
    else:
        print('Test Accuracy: 0.00% (No data evaluated)')


if __name__ == "__main__":
    # Initialize Accelerator
    accelerator = Accelerator()
    accelerator.print("Starting MNIST Digit Classification with Hugging Face datasets")

    # Load data
    train_dataloader, test_dataloader = load_mnist_classification_data()
    accelerator.print(f"Loaded MNIST from HuggingFace: {len(train_dataloader)} train batches, {len(test_dataloader)} test batches")
    
    # Create model and optimizer
    classification_model = DigitClassificationNetwork()
    model_optimizer = torch.optim.SGD(classification_model.parameters(), lr=0.01)
    
    # Prepare everything with Accelerate
    classification_model, model_optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        classification_model, model_optimizer, train_dataloader, test_dataloader
    )

    total_parameters = sum(p.numel() for p in classification_model.parameters())
    accelerator.print(f'Model has {total_parameters:,} parameters')
    
    # Train and evaluate the model
    train_classification_model(classification_model, train_dataloader, model_optimizer, accelerator)
    evaluate_classification_model(classification_model, test_dataloader)
    
    accelerator.print("Training and evaluation completed!")
