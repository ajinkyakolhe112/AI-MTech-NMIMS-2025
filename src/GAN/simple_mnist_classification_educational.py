"""
Educational MNIST Classification Implementation
A simple neural network for digit classification using Accelerate with clear variable names.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchmetrics
from accelerate import Accelerator
import matplotlib.pyplot as plt


def load_mnist_classification_data(batch_size=64):
    """Load and preprocess MNIST dataset for classification"""
    # Simple transformation: convert images to tensors
    image_to_tensor_transform = transforms.ToTensor()
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=image_to_tensor_transform)
    test_dataset = datasets.MNIST('data', train=False, transform=image_to_tensor_transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader


class DigitClassificationNetwork(nn.Module):
    """Neural network for classifying MNIST digits (0-9)"""
    
    def __init__(self, input_size=784, hidden_size_1=256, hidden_size_2=128, num_classes=10):
        super().__init__()
        
        # Network architecture with clear naming
        self.flatten_image_layer = nn.Flatten()  # Convert 28x28 to 784
        self.first_hidden_layer = nn.Linear(input_size, hidden_size_1)
        self.first_activation = nn.ReLU()
        self.first_dropout = nn.Dropout(0.2)  # Prevent overfitting
        
        self.second_hidden_layer = nn.Linear(hidden_size_1, hidden_size_2)
        self.second_activation = nn.ReLU()
        self.second_dropout = nn.Dropout(0.2)
        
        self.output_classification_layer = nn.Linear(hidden_size_2, num_classes)
        
        print(f"Created classification network: {input_size} -> {hidden_size_1} -> {hidden_size_2} -> {num_classes}")
    
    def forward(self, input_images):
        """Forward pass through the network"""
        # Flatten the 28x28 images to 784-dimensional vectors
        flattened_images = self.flatten_image_layer(input_images)
        
        # First hidden layer with activation and dropout
        first_layer_output = self.first_hidden_layer(flattened_images)
        first_activated_output = self.first_activation(first_layer_output)
        first_dropout_output = self.first_dropout(first_activated_output)
        
        # Second hidden layer with activation and dropout
        second_layer_output = self.second_hidden_layer(first_dropout_output)
        second_activated_output = self.second_activation(second_layer_output)
        second_dropout_output = self.second_dropout(second_activated_output)
        
        # Final output layer (no activation - raw logits)
        classification_logits = self.output_classification_layer(second_dropout_output)
        
        return classification_logits


def overfit_single_batch_classification(model, train_dataloader, optimizer, accelerator, iterations=200):
    """
    TESTING FUNCTION: Overfit classification model on a single batch to verify learning capability
    If the network can't overfit one batch, it won't work on full dataset
    """
    accelerator.print("=== OVERFITTING TEST: Training classifier on single batch ===")
    
    # Get one single batch and keep using it
    single_batch_images, single_batch_labels = next(iter(train_dataloader))
    accelerator.print(f"Using single batch with {single_batch_images.size(0)} images")
    
    # Print the true labels in the batch for reference
    unique_labels = torch.unique(single_batch_labels).cpu().numpy()
    accelerator.print(f"Digits in this batch: {unique_labels}")
    
    loss_function = nn.CrossEntropyLoss()
    model.train()  # Set model to training mode
    
    # Train on the same batch repeatedly
    for iteration in range(iterations):
        # Forward pass: get predictions on same batch
        predicted_logits = model(single_batch_images)
        training_loss = loss_function(predicted_logits, single_batch_labels)
        
        # Backward pass: update weights
        optimizer.zero_grad()
        accelerator.backward(training_loss)
        optimizer.step()
        
        # Calculate accuracy on this batch
        predicted_classes = torch.argmax(predicted_logits, dim=1)
        correct_predictions = (predicted_classes == single_batch_labels).sum().item()
        batch_accuracy = (correct_predictions / single_batch_labels.size(0)) * 100
        
        # Print progress every 25 iterations
        if (iteration + 1) % 25 == 0:
            accelerator.print(f'Iteration {iteration+1:3d}: Loss = {training_loss:.4f}, Accuracy = {batch_accuracy:.1f}%')
    
    # Final evaluation on the overfitted batch
    model.eval()
    with torch.no_grad():
        final_predictions = model(single_batch_images)
        final_predicted_classes = torch.argmax(final_predictions, dim=1)
        final_accuracy = (final_predicted_classes == single_batch_labels).sum().item() / single_batch_labels.size(0) * 100
    
    accelerator.print(f"Final overfitting accuracy: {final_accuracy:.1f}%")
    accelerator.print("If accuracy reaches ~100%, the network can learn. If not, check architecture/learning rate.")
    
    model.train()  # Set back to training mode


def train_classification_model(model, train_dataloader, optimizer, accelerator, epochs=10):
    """Training loop for the digit classification model"""
    loss_function = nn.CrossEntropyLoss()
    
    model.train()  # Set model to training mode
    
    for current_epoch in range(epochs):
        total_training_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        for batch_images, batch_labels in train_dataloader:
            # Forward pass: get predictions
            predicted_logits = model(batch_images)
            training_loss = loss_function(predicted_logits, batch_labels)
            
            # Backward pass: update weights
            optimizer.zero_grad()
            accelerator.backward(training_loss)
            optimizer.step()
            
            # Calculate training accuracy
            predicted_classes = torch.argmax(predicted_logits, dim=1)
            correct_predictions += (predicted_classes == batch_labels).sum().item()
            total_samples += batch_labels.size(0)
            total_training_loss += training_loss.item()
        
        # Calculate averages
        average_training_loss = total_training_loss / len(train_dataloader)
        training_accuracy = (correct_predictions / total_samples) * 100
        
        accelerator.print(f'Epoch {current_epoch+1:2d}: Training Loss = {average_training_loss:.4f}, Training Accuracy = {training_accuracy:.2f}%')


def evaluate_classification_model(model, test_dataloader, accelerator):
    """Evaluate the trained model on test data"""
    model.eval()  # Set model to evaluation mode
    
    total_correct_predictions = 0
    total_test_samples = 0
    all_predictions = []
    all_true_labels = []
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for test_images, test_labels in test_dataloader:
            # Get predictions
            test_predictions_logits = model(test_images)
            predicted_digit_classes = torch.argmax(test_predictions_logits, dim=1)
            
            # Collect predictions for detailed analysis
            all_predictions.extend(predicted_digit_classes.cpu().numpy())
            all_true_labels.extend(test_labels.cpu().numpy())
            
            # Calculate accuracy using torchmetrics
            batch_accuracy = torchmetrics.functional.accuracy(
                test_predictions_logits, test_labels, 
                task='multiclass', num_classes=10
            )
            
            total_correct_predictions += batch_accuracy * test_labels.size(0)
            total_test_samples += test_labels.size(0)
    
    # Calculate final test accuracy
    final_test_accuracy = (total_correct_predictions / total_test_samples) * 100
    accelerator.print(f'Final Test Accuracy: {final_test_accuracy:.2f}%')
    
    return all_predictions, all_true_labels


def visualize_predictions(model, test_dataloader, accelerator, num_samples=16):
    """Visualize some predictions to understand model performance"""
    model.eval()
    
    # Get one batch for visualization
    test_images, test_labels = next(iter(test_dataloader))
    
    with torch.no_grad():
        prediction_logits = model(test_images)
        predicted_classes = torch.argmax(prediction_logits, dim=1)
    
    # Convert to CPU for plotting
    images_cpu = test_images.cpu()
    true_labels_cpu = test_labels.cpu()
    predicted_labels_cpu = predicted_classes.cpu()
    
    # Create visualization
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            # Display image
            ax.imshow(images_cpu[i].squeeze(), cmap='gray')
            
            # Add title with true and predicted labels
            true_digit = true_labels_cpu[i].item()
            predicted_digit = predicted_labels_cpu[i].item()
            
            # Color: green if correct, red if incorrect
            title_color = 'green' if true_digit == predicted_digit else 'red'
            ax.set_title(f'True: {true_digit}, Pred: {predicted_digit}', color=title_color)
        
        ax.axis('off')
    
    plt.suptitle('MNIST Classification Results (Green=Correct, Red=Incorrect)', fontsize=14)
    plt.tight_layout()
    plt.savefig('mnist_classification_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    accelerator.print("Saved visualization to 'mnist_classification_results.png'")


def main():
    """Main function to run MNIST digit classification"""
    # Initialize Accelerator for device management
    accelerator = Accelerator()
    accelerator.print("Starting MNIST Digit Classification Training")
    
    # Step 1: Load MNIST data
    train_dataloader, test_dataloader = load_mnist_classification_data(batch_size=128)
    accelerator.print(f"Loaded MNIST: {len(train_dataloader)} train batches, {len(test_dataloader)} test batches")
    
    # Step 2: Create classification model
    classification_model = DigitClassificationNetwork(
        input_size=784,      # 28x28 pixels
        hidden_size_1=256,   # First hidden layer
        hidden_size_2=128,   # Second hidden layer  
        num_classes=10       # Digits 0-9
    )
    
    # Step 3: Create optimizer
    model_optimizer = torch.optim.Adam(classification_model.parameters(), lr=0.001)
    
    # Step 4: Prepare everything with Accelerate
    classification_model, model_optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        classification_model, model_optimizer, train_dataloader, test_dataloader
    )
    
    total_parameters = sum(p.numel() for p in classification_model.parameters())
    accelerator.print(f'Model has {total_parameters:,} trainable parameters')
    
    accelerator.print("\nChoose training mode:")
    accelerator.print("1. Overfit single batch (testing mode)")
    accelerator.print("2. Full dataset training")
    
    # OPTION 1: Test if network can learn by overfitting single batch
    overfit_single_batch_classification(
        classification_model, train_dataloader, model_optimizer, accelerator, iterations=200
    )
    
    # Uncomment the lines below to run full training instead:
    # train_classification_model(
    #     classification_model, train_dataloader, model_optimizer, accelerator, epochs=15
    # )
    # 
    # accelerator.print("\nEvaluating model on test data...")
    # all_predictions, all_true_labels = evaluate_classification_model(
    #     classification_model, test_dataloader, accelerator
    # )
    # 
    # visualize_predictions(classification_model, test_dataloader, accelerator)
    
    accelerator.print("Training completed!")


if __name__ == "__main__":
    main()