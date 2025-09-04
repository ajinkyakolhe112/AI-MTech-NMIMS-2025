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

def load_mnist_classification_data():
    """Load and preprocess MNIST dataset for classification"""
    # Download the Image Dataset
    # And Convert image to numerical value tensors
    image_to_tensor_transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=image_to_tensor_transform)
    test_dataset = datasets.MNIST('data', train=False, transform=image_to_tensor_transform)

    # Create DataLoader function for efficient data-reading
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_dataloader, test_dataloader

class DigitClassificationNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Network architecture: 784 -> 128 -> 64 -> 10
        self.flatten_image_layer = nn.Flatten()               # Convert 28x28 image to 784 pixels
        self.first_hidden_layer = nn.Linear(784, 128)         # First hidden layer
        self.first_activation = nn.ReLU()                     # Activation function
        self.second_hidden_layer = nn.Linear(128, 64)         # Second hidden layer
        self.second_activation = nn.ReLU()                    # Activation function
        self.output_classification_layer = nn.Linear(64, 10)  # Output layer (10 digits)

        print("Created classification network: 784 -> 128 -> 64 -> 10")

    def forward(self, input_images):
        """Forward pass through the network"""
        flattened_images = self.flatten_image_layer(input_images)      # Flatten image
        first_layer_output = self.first_activation(self.first_hidden_layer(flattened_images))  # First layer + activation
        second_layer_output = self.second_activation(self.second_hidden_layer(first_layer_output))  # Second layer + activation
        classification_logits = self.output_classification_layer(second_layer_output)       # Final output (no activation)
        return classification_logits


def train_classification_model(model, train_dataloader, optimizer, accelerator, epochs=3):
    """Training loop with accelerate"""
    loss_function = nn.CrossEntropyLoss()

    model.train()
    for current_epoch in range(epochs):
        total_training_loss = 0
        for batch_images, batch_labels in train_dataloader:

            # Forward pass
            predicted_logits = model(batch_images)
            training_loss = loss_function(predicted_logits, batch_labels)

            # Backward pass
            optimizer.zero_grad()  # Clear gradients
            accelerator.backward(training_loss)  # Use accelerator for backward pass
            optimizer.step()       # Update weights

            total_training_loss += training_loss.item()
        average_training_loss = total_training_loss / len(train_dataloader)
        accelerator.print(f'Epoch {current_epoch+1}: Average Loss = {average_training_loss:.4f}')

def evaluate_classification_model(model, test_dataloader):
    """Test the recently trained model accuracy"""
    model.eval()
    total_correct_predictions = 0
    total_test_samples = 0

    # Set Automatic gradient calculation OFF
    torch.set_grad_enabled(False)
    for test_images, test_labels in test_dataloader:
        test_predictions_logits = model(test_images)
        batch_accuracy = torchmetrics.functional.accuracy(test_predictions_logits, test_labels, task='multiclass', num_classes=10)
        total_correct_predictions += batch_accuracy * len(test_labels)
        total_test_samples += len(test_labels)
    # Set Automatic gradient calculation Back On
    torch.set_grad_enabled(True)

    final_test_accuracy = (total_correct_predictions/total_test_samples)*100
    print(f'Test Accuracy: {final_test_accuracy:.2f}%')

if __name__ == "__main__":
    # Initialize Accelerator
    accelerator = Accelerator()
    accelerator.print("Starting MNIST Digit Classification Training")

    # Step 1: Load data
    train_dataloader, test_dataloader = load_mnist_classification_data()
    accelerator.print(f"Loaded MNIST: {len(train_dataloader)} train batches, {len(test_dataloader)} test batches")

    # Step 2: Create model and optimizer
    classification_model = DigitClassificationNetwork()
    model_optimizer = torch.optim.SGD(classification_model.parameters(), lr=0.01)
    
    # Prepare everything with Accelerate
    classification_model, model_optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        classification_model, model_optimizer, train_dataloader, test_dataloader
    )

    total_parameters = sum(p.numel() for p in classification_model.parameters())
    accelerator.print(f'Model has {total_parameters:,} parameters')

    # Step 3: Train the model
    train_classification_model(classification_model, train_dataloader, model_optimizer, accelerator)

    # Step 4: Test the model
    evaluate_classification_model(classification_model, test_dataloader)
    
    accelerator.print("Training and evaluation completed!")


## Future Improvements
# TODO:ajinkyak: Simple Trainer Function, custom mix of features inspired by lightning. Shift divice management to accelerate
# TODO:ajinkyak: Flag: Overfit one batch.