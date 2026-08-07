import torch.nn as nn

class CnnModel(nn.Module):
    def __init__(self, num_classes):
        super(CnnModel, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv_layer6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.pool = nn.AdaptiveMaxPool2d((7,7))

        self.relu1 = nn.ReLU()

        self.fc1 = nn.Linear(6272,128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self,x):
        out = self.relu1(self.conv_layer1(x))
        out = self.relu1(self.conv_layer2(out))
        out = self.max_pool1(out)

        out = self.relu1(self.conv_layer3(out))
        out = self.relu1(self.conv_layer4(out))
        out = self.max_pool2(out)

        out = self.relu1(self.conv_layer5(out))
        out = self.relu1(self.conv_layer6(out))
        out = self.max_pool3(out)

        out = self.pool(out)
        out = out.reshape(out.size(0), -1)

        out = self.relu1(self.fc1(out))
        out = self.fc2(out)
        return out