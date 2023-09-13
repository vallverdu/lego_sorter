import pandas as pd
import cv2
import os
from matplotlib import pyplot as plt

# Adjust these paths to your local dataset paths
DATA_CSV_PATH = "./data/data.csv"
IMAGES_DIR = "./data/images"

# Load the dataset
data = pd.read_csv(DATA_CSV_PATH)

def display_images_and_labels(data, root_dir, num_images=10):
    """Display a subset of images with their labels."""
    
    # Randomly sample a subset of data
    subset_data = data.sample(n=num_images, random_state=42)
    
    for index, row in subset_data.iterrows():
        img_path = os.path.join(root_dir, row['filename'] + '.png')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Display image with its label
        plt.imshow(img)
        plt.title(f"Brick Type: {row['brick_type']}")
        plt.axis('off')
        plt.show()

# Display a subset of images with their labels
display_images_and_labels(data, IMAGES_DIR, num_images=60)
