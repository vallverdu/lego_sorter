
import os
import numpy as np
from PIL import Image
import pandas as pd

import argparse
import glob
import datetime
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

from torch.utils.tensorboard import SummaryWriter

import imgaug.augmenters as iaa
import imgaug as ia

import matplotlib.pyplot as plt


# Configuration
class Config:
    OUTPUT = 'results'
    DATASET_PATH = './dataset'
    LABELS_PATH = 'dataset.csv'
    IMAGES_PATH = 'images'
    IMAGE_RESIZE = 128
    AUGMENTATION_FACTOR = 5
    TRAIN_SPLIT = 0.8
    BATCH_SIZE = 10
    EPOCHS = 20
    FREEZE_BACKBONE = True

# Data Augmentation using imgaug
augmentation = iaa.Sequential([
    iaa.CropAndPad(percent=(0, 0.1)),
    iaa.Affine(rotate=(-20, 20)),
    iaa.Resize({"height": Config.IMAGE_RESIZE, "width": Config.IMAGE_RESIZE}),
    iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
    iaa.GammaContrast((0.5, 2.0), per_channel=True),
    iaa.GaussianBlur((0, 3.0))
])

# Define the dataset class
class LegoDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None, do_aug=False, augmentation_factor=1):
        self.data = pd.read_csv(os.path.join(root_dir,csv_file))
        self.root_dir = root_dir
        self.transform = transform
        self.do_aug = do_aug
        self.brick_type_dict = {
                        '6*2': 0,
                        '3*1': 1,
                        '3*2': 2,
                        '2*1': 3,
                        '4*2': 4,
                        '1*1': 5,
                        '8*2': 6,
                        '4*1': 7,
                        '2*1_pyramid': 8,
                        '2*2': 9,
                        '6*1': 10
                    }
        self.seq_train = iaa.Sequential([
            iaa.CropAndPad(percent=(0, 0.1)),
            iaa.Affine(rotate=(-20, 20)),
            iaa.Resize({"height": Config.IMAGE_RESIZE, "width": Config.IMAGE_RESIZE}),
            iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.GaussianBlur((0, 3.0))
        ])

        self.seq_test = iaa.Sequential([
            iaa.Resize({"height": Config.IMAGE_RESIZE, "width": Config.IMAGE_RESIZE})
        ])

    def __len__(self):
        return len(self.data) * self.augmentation_factor

    def __getitem__(self, idx):

        # Determine the original index in the annotations
        original_index = idx // self.augmentation_factor
        sample = self.data.iloc[original_index]

        img_name = sample['filename']
        img_path = os.path.join(self.root_dir, img_name + '.png')
        image = np.array(Image.open(img_path))[..., :3]

        # Extract target variables
        brick_type = self.brick_type_dict[sample['brick_type']]

        rotation_x = sample['rotation_x']
        rotation_y = sample['rotation_y']
        rotation_z = sample['rotation_z']

        color_r = sample['color_r']
        color_g = sample['color_g']
        color_b = sample['color_b']

        
        # Apply augmentations
        if self.do_aug:
            image = augmentation(image=image)
        
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
        
        return (
            image,
            torch.tensor(brick_type).long(),
            torch.tensor(rotation_x).float(),
            torch.tensor(rotation_y).float(),
            torch.tensor(rotation_z).float(),
            torch.tensor(color_r).float(),
            torch.tensor(color_g).float(),
            torch.tensor(color_b).float()
        )

# The sections for Model, Training, and Evaluation will be added next...

# Multi-output ResNet Model
class LegoModel(nn.Module):
    def __init__(self, num_brick_types = 11, inference = True):
        super(LegoModel, self).__init__()

        self.inference = inference

        # Load ResNet18 pre-trained on ImageNet
        self.resnet = models.resnet18(pretrained=True)
        
        # Modify the final layer to handle multiple outputs
        num_features = self.resnet.fc.in_features
        
        # Remove the final layer
        self.resnet.fc = nn.Identity()  
        
        
        # Freeze backbone
        # optional for speed or few-shot-learning
        if Config.FREEZE_BACKBONE :
            for param in self.backbone.parameters() :
                param.requires_grad = False
        
        # Define the new final layers for our multi-output prediction
        self.fc_brick_type = nn.Linear(num_features, num_brick_types)
        self.fc_rotation = nn.Linear(num_features, 3)
        self.fc_color = nn.Linear(num_features, 3)


    def forward(self, x):
        x = self.resnet(x)

        brick_type = self.fc_brick_type(x)
        rotation = self.fc_rotation(x)
        color = self.fc_color(x)

        if self.inference:
            brick_type = torch.softmax(brick_type, dim=1)
            rotation = torch.sigmoid(rotation)
            color = torch.sigmoid(color)

        return brick_type, rotation, color



# Training and Evaluation Functions

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for images, brick_types, rotation_x, rotation_y, rotation_z, color_r, color_g, color_b in dataloader:
        images = images.to(device)
        brick_types = brick_types.to(device)
        rotations = torch.stack((rotation_x, rotation_y, rotation_z), dim=1).to(device)
        colors = torch.stack((color_r, color_g, color_b), dim=1).to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        out_brick_type, out_rotation, out_color = model(images)
        
        # Calculate loss
        loss1 = criterion(out_brick_type, brick_types)
        loss2 = criterion(out_rotation, rotations)
        loss3 = criterion(out_color, colors)
        loss = loss1 + loss2 + loss3
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for images, brick_types, rotation_x, rotation_y, rotation_z, color_r, color_g, color_b in dataloader:
            images = images.to(device)
            brick_types = brick_types.to(device)
            rotations = torch.stack((rotation_x, rotation_y, rotation_z), dim=1).to(device)
            colors = torch.stack((color_r, color_g, color_b), dim=1).to(device)
            
            # Forward pass
            out_brick_type, out_rotation, out_color = model(images)
            
            # Calculate loss
            loss1 = criterion(out_brick_type, brick_types)
            loss2 = criterion(out_rotation, rotations)
            loss3 = criterion(out_color, colors)
            loss = loss1 + loss2 + loss3
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Main Execution
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data, create DataLoader
    # Initialize model, criterion, optimizer
    # Training loop with TensorBoard logging
    
    writer = SummaryWriter()

    # Placeholder for the above steps...

    writer.close()

