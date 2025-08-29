import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


class SimpleMNISTClassifier(nn.Module):
    def __init__(self, input_size=784, hidden_size=512, num_classes=10):
        super(SimpleMNISTClassifier, self).__init__()
        
        self.classifier_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, mnist_images):
        # Flatten image from (batch_size, 1, 28, 28) to (batch_size, 784)
        current_batch_size = mnist_images.size(0)
        flattened_images = mnist_images.reshape(current_batch_size, -1)
        
        # Pass through classifier network
        class_predictions = self.classifier_network(flattened_images)
        return class_predictions


class MNISTLightningModule(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.mnist_classifier = SimpleMNISTClassifier()
        
        # Keep track of training metrics
        self.training_step_outputs = []
        self.validation_step_outputs = []
    
    def forward(self, mnist_images):
        return self.mnist_classifier(mnist_images)
    
    def cross_entropy_loss(self, predicted_logits, true_labels):
        return F.cross_entropy(predicted_logits, true_labels)
    
    def calculate_accuracy(self, predicted_logits, true_labels):
        # Get predicted class (highest probability)
        predicted_classes = torch.argmax(predicted_logits, dim=1)
        
        # Calculate how many predictions are correct
        correct_predictions = (predicted_classes == true_labels).float()
        accuracy_percentage = correct_predictions.mean()
        return accuracy_percentage
    
    def training_step(self, batch, batch_idx):
        mnist_images, true_digit_labels = batch
        current_batch_size = mnist_images.shape[0]
        
        # Step 1: Get predictions from classifier
        predicted_digit_logits = self.mnist_classifier(mnist_images)
        
        # Step 2: Calculate classification loss
        classification_loss = self.cross_entropy_loss(predicted_digit_logits, true_digit_labels)
        
        # Step 3: Calculate accuracy for this batch
        batch_accuracy = self.calculate_accuracy(predicted_digit_logits, true_digit_labels)
        
        # Step 4: Log metrics
        self.log('train_loss', classification_loss, prog_bar=True)
        self.log('train_accuracy', batch_accuracy, prog_bar=True)
        
        # Store outputs for epoch-end calculations
        self.training_step_outputs.append({
            'loss': classification_loss,
            'accuracy': batch_accuracy
        })
        
        return classification_loss
    
    def validation_step(self, batch, batch_idx):
        mnist_images, true_digit_labels = batch
        
        # Step 1: Get predictions from classifier (no gradient needed)
        predicted_digit_logits = self.mnist_classifier(mnist_images)
        
        # Step 2: Calculate validation loss
        validation_loss = self.cross_entropy_loss(predicted_digit_logits, true_digit_labels)
        
        # Step 3: Calculate validation accuracy
        validation_accuracy = self.calculate_accuracy(predicted_digit_logits, true_digit_labels)
        
        # Step 4: Log validation metrics
        self.log('val_loss', validation_loss, prog_bar=True)
        self.log('val_accuracy', validation_accuracy, prog_bar=True)
        
        # Store outputs for epoch-end calculations
        self.validation_step_outputs.append({
            'loss': validation_loss,
            'accuracy': validation_accuracy
        })
        
        return validation_loss
    
    def on_train_epoch_end(self):
        # Calculate average training metrics for the epoch
        if self.training_step_outputs:
            average_train_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
            average_train_accuracy = torch.stack([x['accuracy'] for x in self.training_step_outputs]).mean()
            
            print(f"Epoch {self.current_epoch}: Train Loss: {average_train_loss:.4f}, Train Accuracy: {average_train_accuracy:.4f}")
            
            # Clear for next epoch
            self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        # Calculate average validation metrics for the epoch
        if self.validation_step_outputs:
            average_val_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
            average_val_accuracy = torch.stack([x['accuracy'] for x in self.validation_step_outputs]).mean()
            
            print(f"Epoch {self.current_epoch}: Val Loss: {average_val_loss:.4f}, Val Accuracy: {average_val_accuracy:.4f}")
            
            # Clear for next epoch
            self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        optimizer_for_classifier = torch.optim.Adam(
            self.mnist_classifier.parameters(), 
            lr=self.learning_rate
        )
        return optimizer_for_classifier


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, data_dir='./data'):
        super().__init__()
        self.batch_size = batch_size
        self.data_directory = data_dir
        
        # Same transform as GAN but without normalization to [-1, 1]
        self.transform_for_training = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
    
    def setup(self, stage=None):
        # Download and setup training data
        self.mnist_train_dataset = datasets.MNIST(
            root=self.data_directory,
            train=True,
            transform=self.transform_for_training,
            download=True
        )
        
        # Download and setup validation data
        self.mnist_validation_dataset = datasets.MNIST(
            root=self.data_directory,
            train=False,
            transform=self.transform_for_training,
            download=True
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.mnist_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.mnist_validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )


def demonstrate_predictions(model, datamodule, num_samples=16):
    """Show some predictions from the trained model"""
    model.eval()
    
    # Get a batch of validation data
    val_dataloader = datamodule.val_dataloader()
    sample_batch = next(iter(val_dataloader))
    sample_images, sample_true_labels = sample_batch
    
    # Get predictions
    with torch.no_grad():
        predicted_logits = model(sample_images[:num_samples])
        predicted_classes = torch.argmax(predicted_logits, dim=1)
    
    # Plot images with predictions
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            # Show image
            image_to_show = sample_images[i].squeeze().numpy()
            ax.imshow(image_to_show, cmap='gray')
            
            # Add title with true vs predicted label
            true_digit = sample_true_labels[i].item()
            predicted_digit = predicted_classes[i].item()
            title_color = 'green' if true_digit == predicted_digit else 'red'
            ax.set_title(f'True: {true_digit}, Pred: {predicted_digit}', color=title_color)
        
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_predictions_demo.png')
    plt.show()


def main():
    # Initialize the model and data module
    mnist_lightning_model = MNISTLightningModule(learning_rate=0.001)
    mnist_data_module = MNISTDataModule(batch_size=64)
    
    # Initialize trainer
    lightning_trainer = pl.Trainer(
        max_epochs=10,
        accelerator='auto',
        devices=1,
        log_every_n_steps=100,
        check_val_every_n_epoch=1
    )
    
    # Train the model
    print("Starting MNIST classification training...")
    lightning_trainer.fit(mnist_lightning_model, mnist_data_module)
    
    # Demonstrate predictions
    print("Generating prediction demonstrations...")
    demonstrate_predictions(mnist_lightning_model, mnist_data_module)
    
    print("Training completed! Check 'mnist_predictions_demo.png' for sample predictions.")


if __name__ == '__main__':
    main()