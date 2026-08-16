#Adapted from:
#https://medium.com/@imabhi1216/fine-tuning-a-pre-trained-resnet-18-model-for-image-classification-on-custom-dataset-with-pytorch-02df12e83c2c
#Original Author: Abhishek Selokar. Accessed on 06/08/26
#Changes made: loaded default weights, put code into a class to train in different script

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class CnnModel(nn.Module):
    def __init__(self, num_classes):
        super(CnnModel, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        num_features = self.model.fc.in_features

        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self,x):

        return self.model(x)