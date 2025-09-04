"""
Educational GAN Implementation v1: Simple Architecture
A beginner-friendly GAN implementation for MNIST digit generation using Accelerate.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from accelerate import Accelerator
import matplotlib.pyplot as plt
import os


def load_mnist_data(batch_size=64):
    """Load and preprocess MNIST dataset for GAN training"""
    # Simple normalization to [-1, 1] range (standard for GANs)
    transform_for_gan = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Convert [0,1] to [-1,1]
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform_for_gan)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    return train_dataloader


class SimpleGenerator(nn.Module):
    """Generator network: Creates fake images from random noise"""
    
    def __init__(self, noise_dimension=100):
        super().__init__()
        self.noise_dimension = noise_dimension
        
        # Simple feedforward network: 100 -> 256 -> 512 -> 784 (28x28)
        self.noise_to_image = nn.Sequential(
            nn.Linear(noise_dimension, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
            nn.Tanh()  # Output in [-1, 1] to match normalized data
        )
        
        print(f"Created Generator: {noise_dimension} -> 256 -> 512 -> {28*28}")
    
    def forward(self, random_noise):
        """Generate images from random noise"""
        flat_images = self.noise_to_image(random_noise)
        # Reshape from flat vector to image format (batch_size, 1, 28, 28)
        batch_size = flat_images.size(0)
        generated_images = flat_images.view(batch_size, 1, 28, 28)
        return generated_images


class SimpleDiscriminator(nn.Module):
    """Discriminator network: Distinguishes real from fake images"""
    
    def __init__(self):
        super().__init__()
        
        # Simple feedforward network: 784 -> 512 -> 256 -> 1
        self.image_to_decision = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.3),  # Prevent overfitting
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output probability [0, 1]
        )
        
        print(f"Created Discriminator: {28*28} -> 512 -> 256 -> 1")
    
    def forward(self, images):
        """Classify images as real (1) or fake (0)"""
        batch_size = images.size(0)
        flat_images = images.view(batch_size, -1)  # Flatten to (batch_size, 784)
        probability_real = self.image_to_decision(flat_images)
        return probability_real


def train_generator(generator, discriminator, real_images_batch, optimizer_generator, accelerator):
    """Train generator to fool the discriminator"""
    batch_size = real_images_batch.size(0)
    
    # Step 1: Generate fake images from random noise
    random_noise = torch.randn(batch_size, generator.noise_dimension, device=accelerator.device)
    generated_fake_images = generator(random_noise)
    
    # Step 2: Try to fool discriminator (we want it to say these fakes are real)
    discriminator_opinion_on_fakes = discriminator(generated_fake_images)
    target_labels_real = torch.ones_like(discriminator_opinion_on_fakes)  # We want 1s (real)
    
    # Step 3: Calculate loss and update generator
    loss_function = nn.BCELoss()
    generator_loss = loss_function(discriminator_opinion_on_fakes, target_labels_real)
    
    optimizer_generator.zero_grad()
    accelerator.backward(generator_loss)
    optimizer_generator.step()
    
    return generator_loss.item()


def train_discriminator(generator, discriminator, real_images_batch, optimizer_discriminator, accelerator):
    """Train discriminator to correctly identify real vs fake images"""
    batch_size = real_images_batch.size(0)
    loss_function = nn.BCELoss()
    
    # Step 1: Train on REAL images (should output 1)
    discriminator_opinion_on_real = discriminator(real_images_batch)
    target_labels_real = torch.ones_like(discriminator_opinion_on_real)
    loss_on_real_images = loss_function(discriminator_opinion_on_real, target_labels_real)
    
    # Step 2: Train on FAKE images (should output 0)
    random_noise = torch.randn(batch_size, generator.noise_dimension, device=accelerator.device)
    generated_fake_images = generator(random_noise).detach()  # Don't update generator
    discriminator_opinion_on_fakes = discriminator(generated_fake_images)
    target_labels_fake = torch.zeros_like(discriminator_opinion_on_fakes)
    loss_on_fake_images = loss_function(discriminator_opinion_on_fakes, target_labels_fake)
    
    # Step 3: Combined discriminator loss
    total_discriminator_loss = (loss_on_real_images + loss_on_fake_images) / 2
    
    optimizer_discriminator.zero_grad()
    accelerator.backward(total_discriminator_loss)
    optimizer_discriminator.step()
    
    return total_discriminator_loss.item()


def save_generated_samples(generator, epoch, accelerator, num_samples=16):
    """Save generated image samples to visualize progress"""
    generator.eval()
    
    with torch.no_grad():
        # Generate samples
        sample_noise = torch.randn(num_samples, generator.noise_dimension, device=accelerator.device)
        generated_samples = generator(sample_noise)
        
        # Convert to numpy for plotting
        samples_cpu = generated_samples.cpu()
        
        # Create grid plot
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            if i < num_samples:
                # Denormalize from [-1,1] to [0,1] for display
                img = (samples_cpu[i].squeeze() + 1) / 2
                ax.imshow(img, cmap='gray')
            ax.axis('off')
        
        plt.suptitle(f'Generated Images - Epoch {epoch}', fontsize=16)
        plt.tight_layout()
        
        # Save image
        os.makedirs('generated_images', exist_ok=True)
        if isinstance(epoch, str):
            plt.savefig(f'generated_images/{epoch}.png')
        else:
            plt.savefig(f'generated_images/epoch_{epoch:03d}.png')
        plt.close()
    
    generator.train()


def overfit_single_batch_gan(generator, discriminator, train_dataloader, accelerator, iterations=500):
    """
    TESTING FUNCTION: Overfit GAN on a single batch to verify learning capability
    If the networks can't overfit one batch, they won't work on full dataset
    """
    accelerator.print("=== OVERFITTING TEST: Training GAN on single batch ===")
    
    # Get one single batch and keep using it
    single_batch_images, _ = next(iter(train_dataloader))
    accelerator.print(f"Using single batch with {single_batch_images.size(0)} images")
    
    # Create optimizers
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Train on the same batch repeatedly
    for iteration in range(iterations):
        # Train discriminator on the same batch
        discriminator_loss = train_discriminator(
            generator, discriminator, single_batch_images, 
            optimizer_discriminator, accelerator
        )
        
        # Train generator on the same batch
        generator_loss = train_generator(
            generator, discriminator, single_batch_images,
            optimizer_generator, accelerator
        )
        
        # Print progress every 50 iterations
        if (iteration + 1) % 50 == 0:
            accelerator.print(f'Iteration {iteration+1:3d}: Gen Loss = {generator_loss:.4f}, Disc Loss = {discriminator_loss:.4f}')
        
        # Save samples every 100 iterations
        if (iteration + 1) % 100 == 0:
            save_generated_samples(generator, f'overfit_iter_{iteration+1}', accelerator)
    
    accelerator.print("Overfitting test completed! Check if losses decreased and images improved.")
    accelerator.print("If GAN can overfit one batch, it should work on full dataset.")


def train_gan(generator, discriminator, train_dataloader, accelerator, epochs=50):
    """Main training loop for the GAN"""
    # Create optimizers
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Training loop
    for epoch in range(epochs):
        total_generator_loss = 0
        total_discriminator_loss = 0
        num_batches = 0
        
        for real_images_batch, _ in train_dataloader:  # We don't need labels for GAN
            
            # Train discriminator first
            discriminator_loss = train_discriminator(
                generator, discriminator, real_images_batch, 
                optimizer_discriminator, accelerator
            )
            
            # Train generator second
            generator_loss = train_generator(
                generator, discriminator, real_images_batch,
                optimizer_generator, accelerator
            )
            
            total_generator_loss += generator_loss
            total_discriminator_loss += discriminator_loss
            num_batches += 1
        
        # Print progress
        avg_gen_loss = total_generator_loss / num_batches
        avg_disc_loss = total_discriminator_loss / num_batches
        accelerator.print(f'Epoch {epoch+1:3d}: Gen Loss = {avg_gen_loss:.4f}, Disc Loss = {avg_disc_loss:.4f}')
        
        # Save samples every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_generated_samples(generator, epoch + 1, accelerator)


def main():
    """Main function to run the educational GAN"""
    # Initialize Accelerator for device management
    accelerator = Accelerator()
    accelerator.print("Starting Simple Educational GAN Training")
    
    # Step 1: Load MNIST data
    train_dataloader = load_mnist_data(batch_size=128)
    accelerator.print(f"Loaded MNIST dataset with {len(train_dataloader)} batches")
    
    # Step 2: Create models
    noise_dimension = 100
    generator = SimpleGenerator(noise_dimension)
    discriminator = SimpleDiscriminator()
    
    # Step 3: Prepare everything with Accelerate
    generator, discriminator, train_dataloader = accelerator.prepare(
        generator, discriminator, train_dataloader
    )
    
    accelerator.print(f'Generator parameters: {sum(p.numel() for p in generator.parameters()):,}')
    accelerator.print(f'Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}')
    
    # OPTION 1: Test if networks can learn by overfitting single batch
    accelerator.print("\nChoose training mode:")
    accelerator.print("1. Overfit single batch (testing mode)")
    accelerator.print("2. Full dataset training")
    
    # For educational purposes, let's run overfitting test first
    overfit_single_batch_gan(generator, discriminator, train_dataloader, accelerator, iterations=500)
    
    # Uncomment the line below to run full training instead:
    # train_gan(generator, discriminator, train_dataloader, accelerator, epochs=100)
    
    accelerator.print("Training completed! Check 'generated_images' folder for results.")


if __name__ == "__main__":
    main()