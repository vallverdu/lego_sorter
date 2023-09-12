from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

import torch.onnx
import onnx 

from onnxruntime.quantization import quantize_dynamic, QuantType

from onnxruntime.quantization import quantize
from onnxruntime.quantization import QuantizationMode


# Define input and output names
input_names = ['input']
output_names = ['brick_type']

# Define the model architecture
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
        
        
        # Define the new final layers for our multi-output prediction
        self.fc_brick_type = nn.Linear(num_features, num_brick_types)


    def forward(self, x):
        x = self.backbone(x)
        x = x.mean(dim=(2, 3))

        brick_type = self.fc_brick_type(x)

        if self.inference:
            brick_type = torch.softmax(brick_type, dim=1)

        return brick_type

 
    
def quantize_onnx_model(onnx_model_path, quantized_model_path):
    onnx_opt_model = onnx.load(onnx_model_path)
    quantize_dynamic(onnx_model_path,
                     quantized_model_path,
                     weight_type=QuantType.QInt8)

    print(f"quantized model saved to:{quantized_model_path}")



if __name__ == '__main__':

    device = torch.device('cpu')

    # Initialize the model
    model = LegoModel()
    modelPath = './ckpt/ckpt_30_2.278373882174492.pth'

    # Initialize model with the pretrained weights
    model.load_state_dict(torch.load(modelPath, map_location='cpu'))

    # set the model to inference mode
    model.eval().to(device)

    path = "./data/images/brick_0a1ff1d5.png"
    rgb = np.array(Image.open(path).resize((256, 256), Image.BICUBIC))[..., :3]

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

    quantize_onnx_model("./ckpt/model_best.onnx", "./ckpt/model.quantized.onnx")
