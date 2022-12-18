import torch
import torch.nn as nn
import math
import torch.nn.functional as F

channel_means = torch.tensor([[9.7679e-04], [7.8133e-05], 
                              [9.8849e-04], [1.0418e-03], 
                              [8.0620e-04], [8.6011e-04],
                              [5.5876e-04], [3.3484e-05]])

channel_stds =  torch.tensor([[0.0070], [0.0069], 
                              [0.0071], [0.0071],
                              [0.0063], [0.0063], 
                              [0.0039], [0.0039]])


def standardize(inp):
    return (inp-channel_means)/channel_stds   


class ConvNet1D(nn.Module):
    def __init__(self, input_features=8, filters1=32, filters2=64, kernel1=(1, 10), kernel2=(1, 10), filters3=128,
                 kernel3=(1, 5), dropout_p=0.2):
        super(ConvNet1D, self).__init__()

        self.convolutional = nn.Sequential(
                nn.Conv2d(input_features, filters1, kernel1, stride=(1, 2), padding=(0, 1)),  # 4096 -> 1363
                nn.BatchNorm2d(filters1),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),
                nn.Conv2d(filters1, filters1, kernel1, stride=(1, 2), padding=(0, 1)),  # 1363 -> 678
                nn.BatchNorm2d(filters1),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),
            
                nn.MaxPool2d((1, 2)),  # 678 -> 339
                
                nn.Conv2d(filters1, filters2, kernel2, stride=(1, 2), padding=(0, 1)),  # 339 -> 4096
                nn.BatchNorm2d(filters2),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),
                nn.Conv2d(filters2, filters2, kernel2, stride=(1, 2), padding=(0, 1)),
                nn.BatchNorm2d(filters2),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),
            
                nn.MaxPool2d((1, 2)),  # 2048 -> 1024
            
                nn.Conv2d(filters2, filters3, kernel3, stride=(1, 2), padding=(0, 1)),
                nn.BatchNorm2d(filters3),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),
                nn.Conv2d(filters3, filters3, kernel3, stride=(1, 2), padding=(0, 1)),
                nn.BatchNorm2d(filters3),
                nn.ReLU(),
                nn.Dropout(p=dropout_p))

        self.length_after_conv = 2
        self.pool = nn.AvgPool2d(
            stride=(1, self.length_after_conv//1),
            kernel_size=(1, (self.length_after_conv-(1-1)*(self.length_after_conv//1))),
            padding=0)

        self.fully_connected_class = nn.Sequential(nn.Linear(128, 3))

    def forward(self, x):
        #x = standardize(x)
        # x.shape = torch.Size([10, 8, 4096])
        x = x[:, :, 0::4]  # torch.Size([10, 8, 1024])
        x = torch.unsqueeze(x, 2)  # torch.Size([10, 8, 1, 1024])
        x = self.convolutional(x)  # torch.Size([10, 128, 1, 2])
        #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = self.pool(x)  # torch.Size([10, 128, 1, 1])

        x = x.view(x.size(0), -1)  # torch.Size([10, 128])

        x_class = self.fully_connected_class(x)  # torch.Size([10, 3])
        return x_class

    '''self.input_args = {'input_features': input_features,
                       'filters1': filters1,
                       'filters2': filters2,
                       'kernel1': kernel1,
                       'kernel2': kernel2,
                       'filters3': filters3,
                       'kernel3': kernel3,
                       'dropout_p': dropout_p
                       }

    #         self.pool = nn.AvgPool2d(
    #             stride = (1,9//1),
    #             kernel_size=(1,(9-(1-1)*(9//1))),
    #             padding=0,
    #         )
'''
    
if __name__ == "__main__":
    model = ConvNet1D(True)
    output = model(torch.randn(10, 8, 4096, device="cpu"))
    #print(output)
    #from torchsummary import summary
    from torchsummaryX import summary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    #print(summary(model, (8, 4096)))
    print(summary(model, torch.rand([1, 8, 4096]).to(device)))
    