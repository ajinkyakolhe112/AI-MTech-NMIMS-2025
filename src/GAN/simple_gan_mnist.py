import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512), 
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 28 * 28))
    
    def forward(self, input_batch):
        img = self.model(input_batch)
        # Reshape from flat vector to image format
        # Reshape from (batch_size, 784) to (batch_size, 1, 28, 28). 1 = number of channels (grayscale), 28x28 = image dimensions
        batch_size = img.size(0)
        img = img.reshape(batch_size, 1, 28, 28)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        # Flatten image from (batch_size, 1, 28, 28) to (batch_size, 784)
        batch_size = img.size(0)
        img_flat = img.reshape(batch_size, -1)  # -1 means infer the size (784)
        validity = self.model(img_flat)
        return validity


class GAN(pl.LightningModule):
    def __init__(self, latent_dim=100, lr=0.0002):
        super().__init__()
        self.save_hyperparameters()
        
        self.latent_dim = latent_dim
        self.lr = lr
        
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()
        
        self.validation_z = torch.randn(16, latent_dim)
    
    def forward(self, z):
        return self.generator(z)
    
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        real_mnist_images, _ = batch
        current_batch_size = real_mnist_images.shape[0]
        
        # Create random noise vector for generating fake images
        random_noise_vector = torch.randn(current_batch_size, self.latent_dim)
        random_noise_vector = random_noise_vector.type_as(real_mnist_images)
        
        # PyTorch Lightning alternates between two optimizers:
        # optimizer_idx=0: Train Generator
        # optimizer_idx=1: Train Discriminator
        
        if optimizer_idx == 0:
            # ===== TRAIN GENERATOR =====
            # Goal: Make discriminator think fake images are real
            
            # Step 1: Generate fake images from noise
            generated_fake_images = self.generator(random_noise_vector)
            
            # Step 2: Try to fool discriminator - we want it to output 1 (real)
            target_labels_for_real = torch.ones(current_batch_size, 1).type_as(real_mnist_images)
            discriminator_prediction_on_fake = self.discriminator(generated_fake_images)
            
            # Step 3: Calculate generator loss
            # High loss = discriminator correctly identified fakes as fake
            # Low loss = discriminator was fooled into thinking fakes are real
            generator_adversarial_loss = self.adversarial_loss(discriminator_prediction_on_fake, target_labels_for_real)
            
            self.log('g_loss', generator_adversarial_loss, prog_bar=True)
            return generator_adversarial_loss
        
        if optimizer_idx == 1:
            # ===== TRAIN DISCRIMINATOR =====
            # Goal: Correctly identify real vs fake images
            
            # Step 1: Test discriminator on REAL images (should output 1)
            target_labels_for_real = torch.ones(current_batch_size, 1).type_as(real_mnist_images)
            discriminator_prediction_on_real = self.discriminator(real_mnist_images)
            discriminator_loss_on_real_images = self.adversarial_loss(discriminator_prediction_on_real, target_labels_for_real)
            
            # Step 2: Test discriminator on FAKE images (should output 0)
            generated_fake_images = self.generator(random_noise_vector)
            target_labels_for_fake = torch.zeros(current_batch_size, 1).type_as(real_mnist_images)
            discriminator_prediction_on_fake = self.discriminator(generated_fake_images)
            discriminator_loss_on_fake_images = self.adversarial_loss(discriminator_prediction_on_fake, target_labels_for_fake)
            
            # Step 3: Total discriminator loss (average of both)
            total_discriminator_loss = (discriminator_loss_on_real_images + discriminator_loss_on_fake_images) / 2
            
            self.log('d_loss', total_discriminator_loss, prog_bar=True)
            return total_discriminator_loss
    
    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        return [opt_g, opt_d], []
    
    def on_epoch_end(self):
        if self.current_epoch % 10 == 0:
            z = self.validation_z.type_as(self.generator.model[0].weight)
            sample_imgs = self(z)
            
            # Log images
            grid = sample_imgs[:16]
            grid = grid.cpu().numpy()
            
            fig, axes = plt.subplots(4, 4, figsize=(8, 8))
            for i, ax in enumerate(axes.flat):
                ax.imshow(grid[i].squeeze(), cmap='gray')
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(f'generated_images_epoch_{self.current_epoch}.png')
            plt.close()


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def setup(self, stage=None):
        self.mnist_train = datasets.MNIST(
            root='./data', 
            train=True, 
            transform=self.transform, 
            download=True
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.mnist_train, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4
        )


def main():
    # Initialize the model and data module
    model = GAN()
    datamodule = MNISTDataModule(batch_size=64)
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='auto',
        devices=1,
        log_every_n_steps=50
    )
    
    # Train the model
    trainer.fit(model, datamodule)


if __name__ == '__main__':
    main()