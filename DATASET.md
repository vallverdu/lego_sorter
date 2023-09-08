# Lego Synthetic Dataset Generator

## Overview

This Blender Python script is designed to generate a synthetic dataset of Lego bricks. It randomly selects Lego pieces, changes their position, rotation, color, and lighting, and then renders images of the pieces. The script is meant to be run inside Blender and can be executed directly from the terminal with specific arguments.

The primary use case is to create a dataset for training machine learning models, such as convolutional neural networks (CNNs), to recognize and classify different Lego bricks.

## Features

- Randomly selects Lego pieces from a collection.
- Adjusts piece rotation and position for varied dataset images.
- Modifies lighting conditions for each render to simulate different real-world lighting scenarios.
- Outputs rendered images and stores metadata (like piece type, rotation, and color) in a CSV file.

## Usage

### Requirements

- Blender (version used during development: 2.9x)
- Lego pieces imported into Blender with correct hierarchical structures.
- Python (used by Blender's internal scripting engine).

### Running the Script

1. Open Terminal or Command Prompt.
2. Navigate to the Blender directory or ensure that Blender's executable is in your system's PATH.
3. Execute the following command:

```bash
blender -b /path/to/yourfile.blend -P /path/to/yourscript.py -- --output-folder /path/to/output/folder --num-images 1000 --csv-filename /path/to/output/folder/data.csv --engine CYCLES
```

Replace paths with the appropriate locations on your system. The script accepts four arguments:

- `--output-folder`: Directory where the rendered images will be saved.
- `--num-images`: Number of images to generate.
- `--csv-filename`: Path to the CSV file that will store the metadata of each image.
- `--engine`: Rendering engine to use (`CYCLES` or `BLENDER_EEVEE`).

### Adjusting Render Settings

The script is configured to use the Cycles rendering engine by default, but this can be changed with the `--engine` argument. The number of render samples, lighting conditions, and other parameters can be adjusted within the script for different rendering qualities and speeds.
