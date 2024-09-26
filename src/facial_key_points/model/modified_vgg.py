import torch.nn as nn
import torch
from torchvision import models


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def get_model(device): # we'll be using transfer-learning
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    for layers in model.parameters():
        layers.requires_grad = False
    model.avgpool = nn.Sequential(
        nn.Conv2d(512, 512, 3),
        nn.MaxPool2d(2),
        nn.Flatten()
    )
    model.classifier = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, 136),
        nn.Sigmoid()  # sigmoid because all points are important
    )
    return model.to(device=device)
model = get_model(device=device)