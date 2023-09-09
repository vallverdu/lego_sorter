Challenge
Given a supervised problem with a dataset of 5000 lego bricks, where each one has been labeled
including the following information: type of brick, rotations in every XYZ axis and color. The goal is to build
a model that is capable of predicting these fields (type, rotations and color). Please,
avoid creating three different models and try to have all these outputs predicted from a
unique model.

The annotated data is provided in the following format:
- dataset.csv: includes the name of the filename, brick_type, rotation and color (filename,brick_type,rotation_x,rotation_y,rotation_z,color_r,color_g,color_b).
- dataset: folder that contains all the images.


You must use pytorch and imgaug and tensorboard to accomplish the challenge.
Remember to be practical: use a standard backbone, easy data-aug and focus on the problem.
You can assume that even if you use a fully-convolutional network the input images will
always be the same size (128*128).

As a result, we expect to receive your script and a small report attached descripting the
implemented solution, including at least some comments of the model architecture & design,
as well as model evaluation.

