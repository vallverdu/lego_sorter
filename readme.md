
## Lego Brick deep learning model

### Project Overview

This project aims to predict multiple attributes of Lego bricks from images. Given a dataset of 5000 Lego brick images, each labeled with:

- Brick type
- Rotations in XYZ axes
- Color (RGB values)

The goal is to fine-tune a ResNet model that can predict all these attributes simultaneously.

### Model Architecture

A **ResNet18** model pre-trained on ImageNet is used as the backbone. The final layer of the ResNet18 model is modified to produce multiple outputs:

1. Brick Type (Classification output)
2. Rotations in XYZ axes (3 Regression outputs)
3. RGB values of the color (3 Regression outputs)

### Data Handling

**Dataset Class (`LegoDataset`)**:
- The dataset class loads images and their associated attributes from the provided CSV file and image folder.
- Data augmentation techniques (like rotation, cropping, brightness adjustment) are applied using the `imgaug` library to enhance the dataset and make the model robust.

### Training and Evaluation

The training loop processes each batch of images and their associated attributes. For each image, the model predicts:

- The brick type
- Rotations in XYZ axes
- RGB values of the color

Three separate losses are calculated for these predictions, which are then combined to get the total loss for the batch. This combined loss is used for backpropagation and model optimization.

During validation, the model's performance is assessed using the validation set, providing insights into how well the model is generalizing.

### TensorBoard Integration

TensorBoard is integrated to track the progress during training. Scalars such as training and validation losses are logged after each epoch. To monitor the training process with TensorBoard, navigate to the directory where the logs are saved and run:

```
tensorboard --logdir=runs
```

### Configuration

Various parameters and hyperparameters related to the dataset, training, and model are defined in the `Config` class. This includes paths to the dataset, image resize dimensions, batch size, number of epochs, etc.

### Future Enhancements

1. Implement early stopping based on validation loss to prevent overfitting.
2. Experiment with other model architectures or ensemble methods to potentially improve performance.
3. Implement post-training analysis to assess model predictions and identify areas of improvement.
