
import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from torch.utils.tensorboard import SummaryWriter


from imgaug import augmenters as iaa

# Data Augmentation using imgaug
augmentation = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-20, 20)),
    iaa.Crop(percent=(0, 0.1)),
    iaa.Add((-10, 10), per_channel=0.5),
])

# PyTorch Dataset and DataLoader
class LegoDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None, do_aug=False):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.do_aug = do_aug
        self.brick_type_dict = {brick: idx for idx, brick in enumerate(dataframe['brick_type'].unique())}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0] + '.png')
        image = Image.open(img_name)
        
        # Apply augmentations
        if self.do_aug:
            image = np.array(image)
            image = augmentation(image=image)
            image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
            
        # Extract target variables
        brick_type = self.brick_type_dict[self.dataframe.iloc[idx]['brick_type']]
        rotations = [self.dataframe.iloc[idx][col] for col in ['rotation_x', 'rotation_y', 'rotation_z']]
        colors = [self.dataframe.iloc[idx][col] for col in ['color_r', 'color_g', 'color_b']]

        return (
            image,
            torch.tensor(brick_type).long(),
            torch.tensor(rotations).float(),
            torch.tensor(colors).float()
        )


# Model Definition
class MultiOutputCNN(nn.Module):
    def __init__(self, num_brick_types):
        super(MultiOutputCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(128 * 16 * 16, 512)

        self.fc_brick = nn.Linear(512, num_brick_types)
        self.fc_rotation = nn.Linear(512, 3)
        self.fc_color = nn.Linear(512, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        out_brick = self.fc_brick(x)
        out_rotation = self.fc_rotation(x)
        out_color = self.fc_color(x)

        return out_brick, out_rotation, out_color

# Training and Evaluation
# Note: Add necessary code for training and evaluation here

# If script is the main module, execute training and evaluation
if __name__ == "__main__":
    # Load data, create DataLoader, initialize model, train, and evaluate
    pass
