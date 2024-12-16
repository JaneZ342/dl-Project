# %%
import numpy as np
import torch.nn as nn
import torch
import torch.utils.data as data_utils

# %%
import gzip
import pickle

# %%
def load_data(path, batch_size):

    with gzip.open(path, 'rb') as f:
        dataset = pickle.load(f)

    train_data = torch.from_numpy(
        dataset["train"]["images"][:, None, :, :].astype(np.float32))
    train_labels = torch.from_numpy(
        dataset["train"]["labels"].astype(np.int64))

    # TODO normalize dataset
    # mean = train_data.mean()
    # stdv = train_data.std()

    train_dataset = data_utils.TensorDataset(train_data, train_labels)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_data = torch.from_numpy(
        dataset["test"]["images"][:, None, :, :].astype(np.float32))
    test_labels = torch.from_numpy(
        dataset["test"]["labels"].astype(np.int64))

    test_dataset = data_utils.TensorDataset(test_data, test_labels)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, train_dataset, test_dataset

# %%
train_loader, test_loader, train_dataset, test_dataset = load_data("s2_mnist.gz", 32)

# %%
import torch.nn as nn
import torch.nn.functional as F
from icoCNN import ConvIco, PoolIco, LNormIco

# %%
from icoCNN import icosahedral_grid_coordinates

# %%
def preprocess_to_icosahedral(images, resolution):
    """
    Projects images to the icosahedral grid with given resolution.
    Args:
        images: Tensor of shape [batch_size, 1, H, W]
        resolution: Int, resolution of the icosahedral grid
    Returns:
        Tensor of shape [batch_size, 1, 5, 2^r, 2^(r+1)]
    """
    # Generate icosahedral grid coordinates
    grid = icosahedral_grid_coordinates(resolution)  # Shape: [5, 2^r, 2^(r+1), 3]
    grid = torch.tensor(grid, dtype=torch.float32)  # Convert to tensor
    
    # Project grid from 3D to 2D normalized coordinates
    grid_2d = grid[..., :2]  # Keep only (x, y) components
    grid_2d = grid_2d / torch.max(torch.abs(grid_2d))  # Normalize to [-1, 1]

    # Add batch dimension and repeat grid for each image in the batch
    batch_size = images.size(0)
    grid_2d = grid_2d.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)  # Shape: [B, 5, 2^r, 2^(r+1), 2]

    # Reshape grid for grid_sample compatibility
    grid_2d = grid_2d.view(batch_size, 5 * grid_2d.size(2), grid_2d.size(3), 2)

    # Interpolate the input images to match the icosahedral grid
    projected = []
    for i in range(5):  # Loop over the 5 charts
        chart = grid_2d[:, i * grid_2d.size(1) // 5:(i + 1) * grid_2d.size(1) // 5, :, :]
        sampled = F.grid_sample(images, chart, align_corners=False)
        projected.append(sampled)

    # Stack all charts back together
    projected = torch.stack(projected, dim=2)  # Shape: [B, C, 5, 2^r, 2^(r+1)]
    return projected


# %%
# Preprocess train and test datasets
resolution = 4  # Adjust resolution
train_loader_ico = torch.utils.data.DataLoader([
    (preprocess_to_icosahedral(img.unsqueeze(0), resolution), label)
    for img, label in train_dataset
], batch_size=32, shuffle=True)

test_loader_ico = torch.utils.data.DataLoader([
    (preprocess_to_icosahedral(img.unsqueeze(0), resolution), label)
    for img, label in test_dataset
], batch_size=32, shuffle=False)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from icoCNN import ConvIco, PoolIco, LNormIco

class IcoCNNModel(nn.Module):
    def __init__(self, resolution=4):
        super(IcoCNNModel, self).__init__()
        self.resolution = resolution

        # Convolutional layers
        self.conv1 = ConvIco(r=resolution, Cin=1, Cout=8, Rin=1)
        self.lnorm1 = LNormIco(C=8, R=6)

        self.conv2 = ConvIco(r=resolution, Cin=8, Cout=16, Rin=6)
        self.lnorm2 = LNormIco(C=16, R=6)
        self.pool2 = PoolIco(r=resolution, R=6)

        self.conv3 = ConvIco(r=resolution-1, Cin=16, Cout=16, Rin=6)
        self.lnorm3 = LNormIco(C=16, R=6)

        self.conv4 = ConvIco(r=resolution-1, Cin=16, Cout=24, Rin=6)
        self.lnorm4 = LNormIco(C=24, R=6)
        self.pool4 = PoolIco(r=resolution-1, R=6)

        self.conv5 = ConvIco(r=resolution-2, Cin=24, Cout=24, Rin=6)
        self.lnorm5 = LNormIco(C=24, R=6)

        self.conv6 = ConvIco(r=resolution-2, Cin=24, Cout=32, Rin=6)
        self.lnorm6 = LNormIco(C=32, R=6)
        self.pool6 = PoolIco(r=resolution-2, R=6)

        self.conv7 = ConvIco(r=resolution-3, Cin=32, Cout=64, Rin=6)
        self.lnorm7 = LNormIco(C=64, R=6)

        # Fully connected layers (initialized dynamically)
        self.fc1 = None
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)  # Output for 10 classes

    def forward(self, x):
        # Convolutional layers with normalization and pooling
        x = F.relu(self.lnorm1(self.conv1(x)))
        x = F.relu(self.lnorm2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.lnorm3(self.conv3(x)))
        x = F.relu(self.lnorm4(self.conv4(x)))
        x = self.pool4(x)

        x = F.relu(self.lnorm5(self.conv5(x)))
        x = F.relu(self.lnorm6(self.conv6(x)))
        x = self.pool6(x)

        x = F.relu(self.lnorm7(self.conv7(x)))

        # Global pooling
        x = torch.mean(x, dim=(-3, -2, -1))  # Average over spatial, charts, and orientation dimensions
        x = x.view(x.size(0), -1)  # Flatten the tensor

        # Dynamically initialize fc1 based on the input size
        if self.fc1 is None:
            input_size = x.shape[1]  # Determine the flattened size dynamically
            self.fc1 = nn.Linear(input_size, 64).to(x.device)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)


# %%
import torch.optim as optim

# %%
# Initialize model, optimizer, and loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IcoCNNModel(resolution=5).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Fix the input shape to [batch_size, channels, R, charts, H, W]
        data = data.unsqueeze(2).unsqueeze(3)  # Add orientation and charts dimensions
        data = data.repeat(1, 1, 1, 5, 1, 1)  # Repeat along charts dimension

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")


# Evaluation function
def evaluate(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print(f"Test Accuracy: {100. * correct / len(test_loader.dataset):.2f}%")

# Run training and evaluation
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader_resized, optimizer, criterion, epoch)
    evaluate(model, device, test_loader_resized)

# %%
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Define the resize transformation
resize_transform = transforms.Compose([
    transforms.Resize((32, 64))  # Resize to 32x32
])

# Custom Dataset Wrapper to Apply Resize
class ResizeDataset(Dataset):
    def __init__(self, original_dataset, transform):
        self.dataset = original_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]  # Get the original image and label
        image = self.transform(image)     # Apply the resize transformation
        return image, label

# Apply the resizing to train and test datasets
train_dataset_resized = ResizeDataset(train_dataset, resize_transform)
test_dataset_resized = ResizeDataset(test_dataset, resize_transform)

# DataLoader to check the resized images
train_loader_resized = DataLoader(train_dataset_resized, batch_size=32, shuffle=True)
test_loader_resized = DataLoader(test_dataset_resized, batch_size=32, shuffle=False)

# Verify the new image shape
for images, labels in train_loader_resized:
    print(f"Resized image shape: {images.shape}")  # Should be [32, 1, 32, 32]
    print(f"Label shape: {labels.shape}")
    break

# %%
for batch_idx, (data, target) in enumerate(train_loader_ico):
    print(data.shape)

# %%
def print_model_parameters(model):
    """
    Prints the total number of parameters in a PyTorch model,
    along with trainable and non-trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())


    print("Model Parameter Summary:")
    print(f"Total Parameters: {total_params}")

# %%
print_model_parameters(model)


