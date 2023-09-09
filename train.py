import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import imgaug.augmenters as iaa
import pandas as pd
import numpy as np
import os

# -----------------------
# Dataset Loading
# -----------------------

class LegoDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.dataframe.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

# -----------------------
# Image Augmentation
# -----------------------

augmenters = [
    iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, rotate=0, translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),
    iaa.Multiply((0.8, 1.2)),
    iaa.LinearContrast((0.8, 1.2)),
    iaa.AddToHueAndSaturation((-10, 10)),
]
augmentation_sequence = iaa.Sequential(augmenters, random_order=True)

transform = transforms.Compose([
    transforms.Lambda(lambda img: augmentation_sequence.augment_image(np.array(img))),
    transforms.ToTensor(),
])

# -----------------------
# Model Definition
# -----------------------

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32*32*32, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

# -----------------------
# Training
# -----------------------

def train(model, dataloader, optimizer, criterion, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")

    print("Finished Training")

# -----------------------
# Main Code
# -----------------------

if __name__ == "__main__":
    # Constants
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    NUM_CLASSES = 10  # Adjust based on your dataset
    LEARNING_RATE = 0.001

    # Load Dataset
    train_dataset = LegoDataset(csv_file="path_to_csv", image_dir="path_to_images", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize Model, Loss, and Optimizer
    model = SimpleCNN(num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    train(model, train_loader, optimizer, criterion, num_epochs=NUM_EPOCHS)
