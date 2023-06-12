
import torch.nn as nn
from vit_loader import vit_loader

model = vit_loader("simple") # "simple" or "efficient"

def get_all_linear_layers(model):
    linear_layers = []
    # Assuming your transformer is at the attribute "transformer" in the model
    for layer in model.transformer.layers:
        # Checking for linear layers in each subcomponent of the transformer
        for name, sub_module in layer.named_modules():
            if isinstance(sub_module, nn.Linear):
                linear_layers.append(sub_module)
    return linear_layers


linear_layers = get_all_linear_layers(model)
for layer in linear_layers:
    print(layer)