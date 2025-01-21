from torchvision.models import resnet50, ResNet50_Weights
from torch import nn

def get_resnet50(num_outputs):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_outputs)
    return model