# model_training.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize

# Define a simple CNN-based model for feature extraction
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.flatten_dim = None  # To calculate dynamically
        self.fc = None  # Initialize later after computing the flattened size

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        # Dynamically initialize the fully connected layer on first forward pass
        if self.fc is None:
            self.flatten_dim = x.numel() // x.shape[0]  # Calculate flattened dimension
            self.fc = nn.Linear(self.flatten_dim, 256).to(x.device)  # Create FC layer dynamically

        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Custom dataset class
class ImageMatchingDataset(Dataset):
    def __init__(self, images, image_size=(128, 128)):
        self.images = images
        self.resize = Resize(image_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img1 = self.resize(torch.tensor(self.images[idx]).permute(2, 0, 1))
        img2 = self.resize(torch.tensor(self.images[(idx + 1) % len(self.images)]).permute(2, 0, 1))
        return img1, img2

# Load the dataset
images = np.load("prepared_dataset.npy")
dataset = ImageMatchingDataset(images)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize the model, loss, and optimizer
model = FeatureExtractor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
print("Starting model training...")
for epoch in range(num_epochs):
    epoch_loss = 0  # Track total loss for this epoch
    for batch_idx, (img1, img2) in enumerate(dataloader):
        optimizer.zero_grad()
        features1 = model(img1.float())
        features2 = model(img2.float())
        loss = criterion(features1, features2)  # Minimize the difference between paired features
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    avg_epoch_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}\n")

# Save the model
torch.save(model.state_dict(), "feature_extractor.pth")
print("Model training completed successfully!")
