
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

# from torch.utils.tensorboard import SummaryWriter

import imgaug.augmenters as iaa
import imgaug as ia

import matplotlib.pyplot as plt


# Configuration
class Config:
    OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    DATASET_PATH = "/Users/jordivallverdu/Documents/360code/apps/lego_sorter/data"
    LABELS_PATH = os.path.join(DATASET_PATH, "data.csv")
    IMAGES_PATH = os.path.join(DATASET_PATH, "images")
    SUMMARY_PATH = os.path.join(DATASET_PATH, "summary")
    DEBUG_PATH = os.path.join(DATASET_PATH, "examples")
    IMAGE_RESIZE = 128
    AUGMENTATION_FACTOR = 30
    TRAIN_SPLIT= 0.8
    BATCH_SIZE = 64
    FREEZE_BACKBONE = True
    EPOCHS = 2
    DEBUG = False
    CKPT_SAVE_INTERVAL = 5
    LR = 1e-4
    EPS = 1e-6


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
            iaa.GaussianBlur((0, 3.0)),
            iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
            iaa.ContrastNormalization((0.75, 1.5)),
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            iaa.AddToHueAndSaturation((-20, 20)),
            iaa.Affine(shear=(-15, 15)),
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
            iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
            iaa.Flipud(0.3),
            iaa.Fliplr(0.3),
            iaa.Dropout((0.01, 0.1), per_channel=0.5)
        ])

        self.seq_test = iaa.Sequential([
            iaa.Resize({"height": Config.IMAGE_RESIZE, "width": Config.IMAGE_RESIZE})
        ])

        self.augmentation_factor = augmentation_factor
        self.do_aug = do_aug

    def __len__(self):
        return len(self.data) * self.augmentation_factor

    def __getitem__(self, idx):

        # Determine the original index in the annotations
        original_index = idx // self.augmentation_factor
        sample = self.data.iloc[original_index]

        img_name = sample['filename']
        img_path = os.path.join(self.root_dir,'images', img_name + '.png')
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
            image = self.seq_train(image=image)
        else :
            image = self.seq_test(image=image)
        
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


class EarlyStopping:
    def __init__(self, patience=7, delta=0, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0



# The sections for Model, Training, and Evaluation will be added next...

# Multi-output ResNet Model
class LegoModel(nn.Module):
    def __init__(self, num_brick_types = 11, inference = True):
        super(LegoModel, self).__init__()

        self.inference = inference

         # Load resnet34 pre-trained on ImageNet
        model = models.resnet34(pretrained=True)

        self.backbone = nn.Sequential(
            model.conv1, 
            model.bn1, 
            model.relu, 
            model.maxpool, 
            model.layer1, 
            model.layer2, 
            model.layer3, 
            model.layer4
        )
        
        # Modify the final layer to handle multiple outputs
        num_features = model.fc.in_features
        
        # Freeze backbone
        # optional for speed or few-shot-learning
        if Config.FREEZE_BACKBONE :
            for param in self.backbone.parameters() :
                param.requires_grad = False
        
        # Define the new final layers for our multi-output prediction
        self.fc_brick_type = nn.Linear(num_features, num_brick_types)


    def forward(self, x):
        x = self.backbone(x)
        x = x.mean(dim=(2, 3))

        brick_type = self.fc_brick_type(x)

        if self.inference:
            brick_type = torch.softmax(brick_type, dim=1)

        return brick_type



# Training and Evaluation Functions

def train(model, data_loader, optimizer, device, epoch):

    data_loader.dataset.dataset.do_aug = True
    data_loader.dataset.dataset.augmentation_factor = Config.AUGMENTATION_FACTOR

    model.train()

    train_loss_brick_type = 0.0
    train_loss_rotation = 0.0
    train_loss_color = 0.0
    train_loss = 0.0
    
    pbar = tqdm(data_loader)

    for batch_idx, (
        image,
        brick_type,
        rotation_x,
        rotation_y,
        rotation_z,
        color_r,
        color_g,
        color_b

    ) in enumerate(pbar, 0):
        
        image = image.to(device)
        brick_type = brick_type.to(device)
        rotation = torch.stack((rotation_x, rotation_y, rotation_z), dim=1).to(device)
        color = torch.stack((color_r, color_g, color_b), dim=1).to(device)
        
        # Optimizer zero
        optimizer.zero_grad()
        
        # Model inference
        # out_brick_type, out_rotation, out_color = model(image)
        out_brick_type = model(image)
        
        # Calculate loss
        loss_brick = criterion_brick_type(out_brick_type, brick_type)
        # loss_rotation = criterion_values(out_rotation, rotation)
        # loss_color = criterion_values(out_color, color)
        # loss = 1 * loss_brick + 1 * loss_rotation + 1 * loss_color # we could penalize more each parameter
        loss = loss_brick
        
        # pbar.set_description(f"TRAIN loss={float(loss)} | loss_brick={float(loss_brick)} | loss_rotation={float(loss_rotation)} | loss_color={float(loss_color)}")
        pbar.set_description(f"TRAIN loss={float(loss)}")


        # Backward pass and optimization
        loss.backward()
        
        # Optimizer
        optimizer.step()

        # train_loss_brick_type += loss_brick.item()
        # train_loss_rotation += loss_rotation.item()
        # train_loss_color += loss_color.item()
        train_loss += loss.item()
        
    # loss_brick_type = train_loss_brick_type / len(data_loader)
    # loss_rotation = train_loss_rotation / len(data_loader)
    # loss_color = train_loss_color / len(data_loader)
    avg_loss = train_loss / len(data_loader)

    # return avg_loss, loss_brick_type, loss_rotation, loss_color
    return avg_loss

def test(model, data_loader, device, epoch):

    data_loader.dataset.dataset.do_aug = False
    data_loader.dataset.dataset.augmentation_factor = 1

    model.eval()

    test_loss_brick_type = 0.0
    test_loss_rotation = 0.0
    test_loss_color = 0.0
    test_loss = 0.0
    
    # Initialize accumulators
    total_type_correct = 0
    total_rotation_mae = np.zeros(3)
    total_rotation_mse = np.zeros(3)
    total_color_mae = np.zeros(3)
    total_color_mse = np.zeros(3)

    pbar = tqdm(data_loader)

    with torch.no_grad():

        for batch_idx, (
            image,
            brick_type,
            rotation_x,
            rotation_y,
            rotation_z,
            color_r,
            color_g,
            color_b

        ) in enumerate(pbar, 0):
            
            image = image.to(device)
            brick_type = brick_type.to(device)
            # rotation = torch.stack((rotation_x, rotation_y, rotation_z), dim=1).to(device)
            # color = torch.stack((color_r, color_g, color_b), dim=1).to(device)
       
            # Forward pass
            # out_brick_type, out_rotation, out_color = model(image)
            out_brick_type = model(image)

            # Calculate loss
            loss_brick = criterion_brick_type(out_brick_type, brick_type)
            # loss_rotation = criterion_values(out_rotation, rotation)
            # loss_color = criterion_values(out_color, color)
            # loss = 1 * loss_brick + 1 * loss_rotation + 1 * loss_color
            loss = loss_brick

            # pbar.set_description(f"TEST loss={float(loss)} | loss_brick={float(loss_brick)} | loss_rotation={float(loss_rotation)} | loss_color={float(loss_color)}")
            pbar.set_description(f"TEST loss={float(loss)}")

            # test_loss_brick_type += loss_brick.item()
            # test_loss_rotation += loss_rotation.item()
            # test_loss_color += loss_color.item()
            test_loss += loss.item()

            # total_type_correct
            # total_rotation_mae
            # total_rotation_mse
            # total_color_mae
            # total_color_mse

            # Gender accuracy
            type_correct = (torch.argmax(out_brick_type, dim=1) == brick_type).float().sum()
            total_type_correct += type_correct

            # # Age metrics
            # age_diff = np.array(true_age) - np.array(pred_age)
            # total_age_mae += np.sum(np.abs(age_diff))
            # total_age_mse += np.sum(age_diff**2)

            # # Eye position metrics
            # eye_position_diff = np.array(true_eye) - np.array(pred_eye)
            # total_eye_position_mae += np.sum(np.abs(eye_position_diff), axis=0)
            # total_eye_position_mse += np.sum(eye_position_diff**2, axis=0)

    # loss_brick = test_loss_brick_type / len(data_loader)
    # loss_rotation = test_loss_rotation / len(data_loader)
    # loss_color = test_loss_color / len(data_loader)
    avg_loss = test_loss / len(data_loader)

    # return avg_loss, loss_brick, loss_rotation, loss_color, type_correct
    return avg_loss, type_correct

def criterion_values(pred, true):
    return F.mse_loss(torch.sigmoid(pred), true)
  
criterion_brick_type = nn.CrossEntropyLoss()

# Main Execution
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train algorithm')
    parser.add_argument('-e', '--experiment', type=str, help='Name of the experiment', required=True)
    parser.add_argument('-d', '--device', type=str, help='Device for the training', default=0)
    parser.add_argument('-s', '--seed', type=int, help='Seed for the training', default=108)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # define Device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    

    # Create project folder and names
    OUTPUT_FOLDER = os.path.join(Config.OUTPUT, args.experiment, datetime.datetime.now().strftime("%d_%m_%Y__%H_%M_%S"))
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print('OUTPUT_FOLDER',OUTPUT_FOLDER)

    os.makedirs(Config.SUMMARY_PATH, exist_ok=True)
    os.makedirs(Config.DEBUG_PATH, exist_ok=True)

    CHECKPOINTS_FOLDER = os.path.join(OUTPUT_FOLDER, 'checkpoints')
    os.makedirs(CHECKPOINTS_FOLDER, exist_ok=True)

    # Define the data transformations
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define the dataset and data loader
    lego_dataset = LegoDataset('data.csv', './data', transform=data_transform)
    train_size = int(Config.TRAIN_SPLIT * len(lego_dataset))
    test_size = len(lego_dataset) - train_size
    print('train_size',train_size)
    print('test_size',test_size)
    train_dataset, test_dataset = torch.utils.data.random_split(lego_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Initialize the model and define the loss function and optimizer
    model = LegoModel(inference=False)

    optimizer = optim.Adam(model.parameters(), lr=Config.LR, eps = Config.EPS)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model.to(device)

    # Epochs
    best_val_loss = float('inf')
    best_epoch = 0
    best_model = None
    best_type_accuracy = None

    # Initialize accumulators
    train_losses = []
    val_losses = []
    epochs_type_accuracy  = []


    early_stopping = EarlyStopping(patience=10, verbose=True)

    # Train the model
    for epoch in range(Config.EPOCHS):
        
        
        train_loss = train(model, train_loader, optimizer, device, epoch)
        val_loss, total_type_correct = test(model, test_loader, device, epoch)

        # Append losses for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Compute average metrics for the epoch
        epoch_type_accuracy = total_type_correct / len(lego_dataset) * 100

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model = model.state_dict()
            best_gender_accuracy = epoch_type_accuracy
        
        # Checkpointing
        if epoch > 0 and epoch % Config.CKPT_SAVE_INTERVAL == 0 :
            print(f"Saved checkpoint {epoch}\n")
            model_path_solve = os.path.join(CHECKPOINTS_FOLDER, f"ckpt_{epoch}_{val_loss}.pth")
            optimizer_path_solve = os.path.join(CHECKPOINTS_FOLDER, model_path_solve.replace('ckpt_', 'optim_'))
            torch.save(model.state_dict(), model_path_solve)
            torch.save(optimizer.state_dict(), optimizer_path_solve)

        # Print progress
        print(f"Epoch {epoch + 1}/{Config.EPOCHS}\n"
          f"Type accuracy: {epoch_type_accuracy:.2f}%\n"
          f"Validation loss: {val_loss:.4f}")
        

        # Step the scheduler
        scheduler.step()

        # Check for early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping")
            break
        
       

    # Save the best model
    torch.save(best_model, os.path.join(CHECKPOINTS_FOLDER, f"best_{best_epoch}_{best_val_loss}.pth"))

        


    # Plot the learning curves
    epochs = range(1, Config.EPOCHS + 1)

    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_losses, label="Training")
    plt.plot(epochs, val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Type Output Learning Curve")
    plt.legend()
    plt.show()



