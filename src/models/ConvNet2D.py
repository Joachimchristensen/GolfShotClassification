import torch
import torch.nn as nn
import math
from torchvision import transforms, models

transform1 = transforms.Compose([transforms.Resize((128, 128)), transforms.Normalize((-53.5712), (17.5148))])
transform2 = transforms.Compose([transforms.Resize((128, 128)), transforms.Normalize((-55.7262), (18.7910))])


class ConvNet2D(nn.Module):
    def __init__(self):
        super(ConvNet2D, self).__init__()
        self.input_args = {}
        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),  # 128 -> 128
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(32, 32, 5, stride=2, padding=1),  # 128 -> 75
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(32, 32, 5, stride=2, padding=1),  # 75 -> 36
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),  #
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(2, stride=2),  # 36 -> 18

            nn.Conv2d(64, 128, 5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(2, stride=2),  # 18 -> 7

            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 18 -> 2
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),  # 4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fully_connected_class = nn.Sequential(
            nn.Linear(256, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(100, 3),
            #nn.Sigmoid()
        )

    def stand(self, x1, x2):
        psd22 = transform2(x2)
        psd11 = transform1(x1)
        inputs = torch.cat((psd11, psd22), 1)
        return inputs

    def forward(self, x):
        #x = self.stand(x1, x2)
        x = self.convolutional(x)
        # reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        x_class = self.fully_connected_class(x)
        return x_class


if __name__ == "__main__":
    model = ConvNet2D()
    output = model(torch.randn(10, 3, 128, 128, device="cpu"))
    # print(output)
    # from torchsummary import summary
    from torchsummaryX import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # print(summary(model, (8, 4096)))
    print(summary(model, torch.rand([10, 3, 128, 128]).to(device)))
