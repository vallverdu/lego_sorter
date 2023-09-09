from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

import torch.onnx
import onnx 

from onnxruntime.quantization import quantize_dynamic, QuantType


# Define input and output names
input_names = ['input']
output_names = ['age_output', 'gender_output',  'pos_output']

# Define the model architecture
class FaceModel(nn.Module):
    
    def __init__(self, inference = True):
        super(FaceModel, self).__init__()

        self.inference = inference

        # Load ResNet18 pre-trained on ImageNet
        model = models.resnet18(pretrained=True)
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
        
        # print('resnet18 architecture',self.backbone)
        num_features = model.fc.in_features


        # Add new fully connected layers for gender, age and eye position
        self.fc_gender = nn.Linear(num_features, 2)
        self.fc_age = nn.Linear(num_features, 1)
        self.fc_eye = nn.Linear(num_features, 4)

        
    def forward(self, x):

        x = self.backbone(x)
        x = x.mean(dim=(2, 3))

        age = self.fc_age(x)
        gender = self.fc_gender(x)
        eye = self.fc_eye(x)

        if self.inference:
            age = torch.sigmoid(age)
            gender = torch.softmax(gender, dim=1)
            eye = torch.sigmoid(eye)
       
        return age, gender, eye
    
    
def quantize_onnx_model(onnx_model_path, quantized_model_path):
    onnx_opt_model = onnx.load(onnx_model_path)
    quantize_dynamic(onnx_model_path,
                     quantized_model_path,
                     weight_type=QuantType.QInt8)

    print(f"quantized model saved to:{quantized_model_path}")



if __name__ == '__main__':

    device = torch.device('cpu')

    # Initialize the model
    model = FaceModel()
    # modelPath = './ckpt/ckpt_15_2.7193858101963997.pth'
    modelPath = './ckpt/best_16_0.2764944826439023.pth'

    # Initialize model with the pretrained weights
    model.load_state_dict(torch.load(modelPath, map_location='cpu'))

    # set the model to inference mode
    model.eval().to(device)

    path = "./data/images/0.jpg"
    rgb = np.array(Image.open(path).resize((224, 224), Image.BICUBIC))[..., :3]

    x = torch.from_numpy(rgb).permute(2,0,1)[None].float() / 255.

    x = x.to(device)

    # Export the model
    torch.onnx.export(model,                    # model being run
					x,                         	# model input (or a tuple for multiple inputs)
					"./ckpt/model_best.onnx",   			# where to save the model (can be a file or file-like object)
					export_params=True,        	# store the trained parameter weights inside the model file
					opset_version=12,          	# the ONNX version to export the model to
					do_constant_folding=True,  	# whether to execute constant folding for optimization
					input_names = input_names,  # the model's input names
					output_names = output_names	# the model's output names
					
	)

    quantize_onnx_model("./ckpt/model.onnx", "./ckpt/model.quantized.onnx")



